"""
The prefill pipeline splitting a block into steps: pre_attn on the block's device, the attention on the
pinned device's own stage, post_attn back on the block's device. No GPU: the devices are names, the tensors
stay on the CPU, and to_device is patched out, so what is checked is the route, the order and the handoff.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

import exllamav3.model.prefill_pipeline as pp
import exllamav3.modules.module as mod
import exllamav3.modules.transformer as tf
from exllamav3.model.prefill_pipeline import PrefillPipeline
from exllamav3.modules.transformer import TransformerBlock

pytestmark = pytest.mark.nogpu

D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")

LOG = []
LOCK = threading.Lock()


def record(device, what, params, tag):
    with LOCK:
        LOG.append((device, what, tag, params.get("chunk"), threading.current_thread().name))


class _Attn:
    """Attention pinned to D0"""

    def __init__(self, device, tag):
        self.device = device
        self.tag = tag
        self.caps = {"attn_device": True}

    def prepare_for_device(self, y, params):
        return y

    def forward(self, y, params):
        record(self.device, "attn", params, self.tag)
        return y


class _MLP:
    def __init__(self, device, tag, delay = 0.0):
        self.device = device
        self.tag = tag
        self.caps = {}
        self.num_slices = 1
        self.delay = delay

    def forward(self, y, params):
        record(self.device, "mlp", params, self.tag)
        time.sleep(self.delay)
        return y


class _Model:
    loaded_tp = False

    def __init__(self, blocks):
        self.fwd_modules = [(b, 0, i) for i, b in enumerate(blocks)]
        self.last_kv_end = len(blocks)
        self.last_kv_module_idx_instance = (len(blocks) - 1, 0)
        self._get_prefetch_layers = []
        self.config = SimpleNamespace()

    def prepare_inputs(self, input_ids, params):
        return input_ids.float()

    def run_section(self, x, params, a, b, prefill):
        record(self.fwd_modules[a][0].device, "section", params, a)
        return x


def block(idx, device, attn_device, delay = 0.0):
    b = TransformerBlock(
        config = SimpleNamespace(),
        key = f"block.{idx}",
        layer_idx = idx,
        attn = _Attn(attn_device, idx),
        mlp = _MLP(device, idx, delay),
    )
    b.device = device
    return b


@pytest.fixture(autouse = True)
def no_device_copies(monkeypatch):
    # The tensors stay on the CPU; the devices here are only names
    monkeypatch.setattr(mod, "to_device", lambda x, device, non_blocking = False: x)
    monkeypatch.setattr(pp, "to_device", lambda x, device, non_blocking = False: x)
    monkeypatch.setattr(tf, "to_device", lambda x, device, non_blocking = False: x)
    LOG.clear()


def model(delay = 0.0):
    # Two blocks on D1, both with attention on D0; the second is the last cache-writing layer
    return _Model([block(0, D1, D0, delay), block(1, D1, D0)])


def shape(stages):
    return [(s.device, s.steps) for s in stages]


def test_plan_splits_a_block_around_its_pinned_attention():
    stages, draft, advance = PrefillPipeline.plan(model(), None)
    assert shape(stages) == [
        (D1, [("pre", 0)]),
        (D0, [("attn", 0)]),
        (D1, [("post", 0, True), ("pre", 1)]),
        (D0, [("attn", 1)]),
        (D1, [("post", 1, False)]),
    ]
    assert draft is None and advance == 0


def test_a_pinned_part_outside_a_splittable_block_keeps_the_inline_route():
    m = model()
    # A module that is not a TransformerBlock but has a part pinned to D0 (a plain module here): that part
    # would keep running inline on a stage thread while D0's stage thread works, with nothing ordering the
    # two, so the plan keeps the inline route for the whole model
    plain = SimpleNamespace(device = D1, modules = [SimpleNamespace(device = D0, caps = {"attn_device": True})])
    m.fwd_modules.append((plain, 0, 2))
    m.fwd_modules.append((SimpleNamespace(device = D0, modules = []), 0, 3))
    stages, _, _ = PrefillPipeline.plan(m, None)
    assert shape(stages) == [(D1, [("section", 0, 3)]), (D0, [("section", 3, 4)])]


def test_plan_waits_for_the_attention_stage_before_the_states_advance():
    m = model()
    cache = SimpleNamespace(
        recurrent_state_cls = SimpleNamespace(guaranteed_rollback = 256),
        get_all_recurrent_layers = lambda: {"a": SimpleNamespace(device = D0)},
    )
    stages, draft, advance = PrefillPipeline.plan(m, cache)
    assert advance == 1


def test_steps_run_in_route_order_and_chunks_do_not_overtake():
    m = model(delay = 0.01)
    stages, draft, advance = PrefillPipeline.plan(m, None)
    pipe = PrefillPipeline(m, stages, draft, advance)
    try:
        for n in range(6):
            pipe.prefill(torch.zeros(1, 4), {"chunk": n, "block_table": torch.zeros(1)}, "job")
        pipe.flush()
    finally:
        pipe.close()

    # Per layer, the chunks ran in order on one thread: that is what a layer's state depends on
    for what, tag in (("attn", 0), ("mlp", 0), ("attn", 1)):
        entries = [e for e in LOG if e[1] == what and e[2] == tag]
        assert [e[3] for e in entries] == list(range(6))
        assert len({e[4] for e in entries}) == 1

    # Per chunk: attention for block 0 on its own device, block 0's MLP back on D1, then block 1. The last
    # cache-writing layer returns after its attention in a prefill, so its MLP does not run
    for n in range(6):
        kinds = [(e[0], e[1], e[2]) for e in LOG if e[3] == n]
        assert kinds == [(D0, "attn", 0), (D1, "mlp", 0), (D0, "attn", 1)]


def test_the_residual_survives_the_hop():
    m = model()
    stages, draft, advance = PrefillPipeline.plan(m, None)
    pipe = PrefillPipeline(m, stages, draft, advance)
    seen = []

    def mlp_forward(y, params):
        # post_attn gets the residual back as the block saw it, and the attention output in half
        seen.append((params["residual"].dtype, y.dtype))
        return torch.ones_like(params["residual"])

    m.fwd_modules[0][0].mlp.forward = mlp_forward
    try:
        pipe.prefill(torch.ones(1, 4), {"chunk": 0, "block_table": torch.zeros(1)}, "job")
        pipe.flush()
    finally:
        pipe.close()
    assert seen == [(torch.float32, torch.float16)]


def test_to_output_runs_the_whole_forward_through_the_phases():
    # A drafter's prefill runs past the last cache-writing layer, and the residual add it defers into the
    # next block's input norm crosses a stage boundary here
    m = model()
    stages, draft, advance = PrefillPipeline.plan(m, None)
    pipe = PrefillPipeline(m, stages, draft, advance)
    try:
        pipe.prefill(torch.ones(1, 4), {"chunk": 0, "block_table": torch.zeros(1)}, "job", to_output = True)
        pipe.flush()
    finally:
        pipe.close()
    kinds = [(e[0], e[1], e[2]) for e in LOG if e[3] == 0]
    assert kinds == [(D0, "attn", 0), (D1, "mlp", 0), (D0, "attn", 1), (D1, "mlp", 1)]


def test_a_failure_on_the_attention_stage_aborts_the_chunk():
    m = model()
    stages, draft, advance = PrefillPipeline.plan(m, None)
    pipe = PrefillPipeline(m, stages, draft, advance)

    def boom(y, params):
        if params.get("chunk") == 0:
            raise RuntimeError("boom")
        return y

    m.fwd_modules[0][0].attn.forward = boom
    try:
        pipe.prefill(torch.zeros(1, 4), {"chunk": 0, "block_table": torch.zeros(1)}, "bad")
        pipe.prefill(torch.zeros(1, 4), {"chunk": 1, "block_table": torch.zeros(1)}, "good")
        pipe.flush()
        failures = pipe.take_failures()
    finally:
        pipe.close()
    assert [(o, str(e)) for o, e in failures] == [("bad", "boom")]
    # The good chunk still ran its MLP: the failed item did not hold up the route
    assert any(e[0] == D1 and e[1] == "mlp" and e[3] == 1 for e in LOG)
