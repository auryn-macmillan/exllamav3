"""
The prefill pipeline with attention pinned to the split's attention device.

DeepSeek-V4.1 loads every attention layer on one device (caps["attn_device"], Model.load
attn_device) because runs of layers read the same compressed K/V pool and index selection.
Those parts are not another stage: the stage that owns the layer drives them inline, and the
pipeline serializes the pinned device with one lock per device so two chunks can't interleave
there. No GPU: the devices are only names and the block's hop is a no-op on CPU tensors.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from exllamav3.model.prefill_pipeline import PrefillPipeline
from exllamav3.modules.transformer import TransformerBlock

pytestmark = pytest.mark.nogpu

D0, D1, D2 = torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")


def pinned(device):
    return SimpleNamespace(device = device, caps = {"attn_device": True})


def module(device, *children):
    m = SimpleNamespace(device = device)

    class Mod:
        def __iter__(self):
            yield m
            yield from children

        device = m.device

    return Mod()


def plan_model(devices, **kw):
    mods = [(module(d), 0, i) for i, d in enumerate(devices)]
    return SimpleNamespace(fwd_modules = mods, loaded_tp = False, config = SimpleNamespace(), **kw)


def plan_cache(devices, **state):
    layers = {i: SimpleNamespace(device = d) for i, d in enumerate(devices)}
    return SimpleNamespace(
        recurrent_state_cls = SimpleNamespace(**state),
        get_all_recurrent_layers = lambda: layers,
    )


def shape(stages):
    return [(s.device, s.steps) for s in stages]


def test_pinned_attention_does_not_disqualify_the_split():
    m = plan_model(["cuda:0", "cuda:1", "cuda:1"])
    m.fwd_modules[1] = (module("cuda:1", pinned("cuda:0")), 0, 1)
    stages, draft, advance = PrefillPipeline.plan(m, None)
    # Not a TransformerBlock, so the pinned part stays inline and the route is plain sections
    assert shape(stages) == [(D0, [("section", 0, 1)]), (D1, [("section", 1, 3)])]
    assert draft is None and advance == 0


def test_a_part_on_another_device_that_is_not_pinned_still_rejects():
    m = plan_model(["cuda:0", "cuda:1"])
    m.fwd_modules[1] = (module("cuda:1", SimpleNamespace(device = "cuda:0")), 0, 1)
    assert PrefillPipeline.plan(m, None) is None


def test_pinned_device_owns_the_ring_states_when_the_first_stage_runs_it():
    # The states live on the pinned device, which is not a stage of its own: the first stage owns the
    # layer that runs it, so the states can advance before section 1 runs
    m = plan_model(["cuda:0", "cuda:1"])
    m.fwd_modules[0] = (module("cuda:0", pinned("cuda:2")), 0, 0)
    cache = plan_cache(["cuda:2"], guaranteed_rollback = 256)
    assert PrefillPipeline.plan(m, cache) is not None


def test_ring_states_on_a_pinned_device_the_first_stage_does_not_run_reject():
    m = plan_model(["cuda:0", "cuda:1"])
    m.fwd_modules[1] = (module("cuda:1", pinned("cuda:2")), 0, 1)
    assert PrefillPipeline.plan(m, plan_cache(["cuda:2"], guaranteed_rollback = 256)) is None
    # Without a guaranteed rollback window the advance is only position bookkeeping, and any stage may
    # leave the states to advance before its layer has run
    assert PrefillPipeline.plan(m, plan_cache(["cuda:2"])) is not None


def test_pipeline_locks_each_pinned_device():
    m = plan_model(["cuda:0", "cuda:1"])
    m.fwd_modules[1] = (module("cuda:1", pinned("cuda:2")), 0, 1)
    stages, draft, _ = PrefillPipeline.plan(m, None)
    pipe = PrefillPipeline(m, stages, draft)
    try:
        assert set(pipe.device_locks) == {D2}
        assert m.config.device_locks is pipe.device_locks
    finally:
        pipe.close()


class _Attn:
    """A hop that records whether the device lock was held when it ran"""

    def __init__(self, device, lock):
        self.device = device
        self.caps = {}
        self.lock = lock
        self.calls = []

    def prepare_for_device(self, y, params):
        return y

    def forward(self, y, params):
        self.calls.append(self.lock.locked())
        return y


def test_block_takes_the_pinned_device_lock():
    lock = threading.Lock()
    attn = _Attn(D2, lock)
    block = TransformerBlock(config = SimpleNamespace(device_locks = {D2: lock}),
                             key = "block", attn = attn, mlp = None)
    block.device = torch.device("cpu")
    x = torch.zeros(2, 4, dtype = torch.float32)
    assert block.forward(x, {}).shape == x.shape
    assert attn.calls == [True]


def test_block_hops_without_a_lock_when_nothing_is_pipelined():
    attn = _Attn(D2, threading.Lock())
    block = TransformerBlock(config = SimpleNamespace(), key = "block", attn = attn, mlp = None)
    block.device = torch.device("cpu")
    block.forward(torch.zeros(2, 4, dtype = torch.float32), {})
    assert attn.calls == [False]
