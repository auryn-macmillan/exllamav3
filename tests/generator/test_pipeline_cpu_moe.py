"""
The prefill pipeline with CPU-offloaded MoE experts.

No GPU: devices are only names, and the CPU host is a stub. What matters here is
that a host-side co-processor no longer disqualifies a layer split (its work runs
on the stage's thread, with MoeCpuHost serializing the shared ring), that a CUDA
part on the wrong device still does, and that every pipelined chunk wakes the
CPU workers once, as prefill_ls() used to do for a single-threaded prefill.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from exllamav3.model.prefill_pipeline import PrefillPipeline, Stage

pytestmark = pytest.mark.nogpu

D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")


def shape(stages):
    return [(s.device, s.steps) for s in stages]


class _Child:
    def __init__(self, device):
        self.device = device


class _Module:
    def __init__(self, device, children = ()):
        self.device = device
        self.children = children

    def __iter__(self):
        return iter(self.children)


class _Model:
    loaded_tp = False

    def __init__(self, devices, children = None, cpu_hosts = True):
        self.fwd_modules = [(_Module(d, (children or {}).get(i, ())), 0, i)
                            for i, d in enumerate(devices)]
        self.config = SimpleNamespace(moe_cpu_hosts = {"host": object()} if cpu_hosts else {})


def test_cpu_moe_host_does_not_disqualify_the_split():
    model = _Model([D0, D1, D1])
    stages, draft, _ = PrefillPipeline.plan(model, None)
    assert shape(stages) == [(D0, [("section", 0, 1)]), (D1, [("section", 1, 3)])]
    assert draft is None


def test_cpu_side_parts_stay_on_the_stage():
    model = _Model([D0, D1, D1], children = {1: (_Child(torch.device("cpu")),)})
    stages, _, _ = PrefillPipeline.plan(model, None)
    assert shape(stages) == [(D0, [("section", 0, 1)]), (D1, [("section", 1, 3)])]


def test_cuda_part_on_another_device_still_rejects():
    model = _Model([D0, D1, D1], children = {1: (_Child(D0),)})
    assert PrefillPipeline.plan(model, None) is None


def test_single_device_still_rejects():
    assert PrefillPipeline.plan(_Model([D0, D0]), None) is None


class _StubModel:
    def __init__(self):
        self.fwd_modules = [None] * 3
        self.last_kv_end = 3
        self._get_prefetch_layers = []
        self.passes = 0
        self.lock = threading.Lock()

    def begin_prefill_pass(self):
        with self.lock:
            self.passes += 1

    def prepare_inputs(self, input_ids, params):
        return input_ids

    def run_section(self, x, params, a, b, prefill):
        return x


def test_every_pipelined_chunk_wakes_the_cpu_workers():
    model = _StubModel()
    pipe = PrefillPipeline(model, [Stage(D0, [("section", 0, 1)]), Stage(D1, [("section", 1, 3)])], None)
    for n in range(3):
        pipe.prefill(torch.zeros(1, 4), {"chunk": n, "block_table": torch.zeros(1)}, "job")
    pipe.flush()
    assert model.passes == 3
    pipe.close()
