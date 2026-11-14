from __future__ import annotations
import copy
import queue
import threading
import weakref
from functools import partial

import torch

from ..cache.recurrent_util import advance_recurrent_states

"""
Prefill across a layer split, with the devices working on different chunks at once.

The stream of each device is in order and transfers between devices are ordered by stream events, so a chunk can be
queued on a device as soon as the host gets to it. What the host can't do by itself is get to it: expert routing reads
counts back to plan its dispatch, which holds the launching thread until the layer's device has caught up, so one
thread can't issue a chunk to the first device while the last device works through the previous one. Each device
gets its own thread here instead.

Everything that touches a device during a prefill is an item that walks one fixed route through the devices, running
one step at each: a chunk runs its section of the model there, a checkpoint copies the states of the layers there, and
the draft model's update for the chunk is a last step on its device. Each device runs its steps first come, first
served, on one thread. That gives the two properties everything depends on:

- Nothing else touches the device while a step runs, so the statics scratch buffers come from can be shared.
- An item can't overtake one issued before it: at the first device it runs later, so it reaches the next device later,
  and so on. A checkpoint therefore reads the states of every layer exactly as the chunks before it left them.

A device the modules are pinned to (caps["attn_device"]) has no stage of its own: the stage that owns a pinned part
drives it inline, from its own thread. Its work is the exception to the first point, so the pipeline keeps one lock
per pinned device (device_locks, installed on the config) and the hop takes it. An item still can't overtake one
issued before it: the pinned device's own queue is ordered by the lock and the stream, and the stage threads that
reach it do so in route order.

Only prefill goes through here. Model.forward and everything else that uses the devices, or reads what the items
write, waits for the route to clear first (flush) and then has them to itself.
"""


def _worker(tasks: queue.SimpleQueue, local: threading.local):
    local.is_worker = True
    with torch.inference_mode():
        while (task := tasks.get()) is not None:
            task()


class _Item:

    def __init__(self, steps: list, owner, is_pass: bool):
        self.steps = steps
        self.owner = owner
        self.is_pass = is_pass
        self.aborted = False
        self.on_done = None
        self.started = None


def _children(module) -> list:
    """The direct children of a module, or, for a module that only enumerates itself (a test double), those"""
    modules = getattr(module, "modules", None)
    if modules is not None:
        return [m for m in modules if m is not None]
    try:
        return [m for m in module if m is not module]
    except TypeError:
        return []


def _foreign_device_parts_ok(module, device, pinned: set) -> bool:
    """
    Whether every part of module on another CUDA device is a module pinned there (caps["attn_device"],
    Model.load attn_device), collecting the pinned devices in pinned. Any other part left on a foreign
    device would run on the wrong device's thread, with nothing ordering it against the stages that own
    that device. A pinned part is serialized instead (see PrefillPipeline.device_locks), so the stage
    thread can drive it where it stands.
    """
    for m in _children(module):
        md = torch.device(m.device) if m.device is not None else None
        if md is not None and md != device and md.type == "cuda":
            if not getattr(m, "caps", {}).get("attn_device"):
                return False
            pinned.add(md)
            continue
        if not _foreign_device_parts_ok(m, device, pinned):
            return False
    return True


class PrefillPipeline:

    @classmethod
    def plan(cls, model, cache, draft_model = None) -> tuple[list, int | None] | None:
        """
        The route through the devices for this model and draft model, as (stages, draft stage), each stage a
        (device, first module, end module) of the target's forward (draft stage: an index into the route, past the
        target's stages), or None if the split can't run this way.
        """
        modules = getattr(model, "fwd_modules", None)
        if not modules or model.loaded_tp:
            return None
        stages = []
        pinned_at = {}
        pinned_all = set()
        for i, (module, _, _) in enumerate(modules):
            device = torch.device(module.device)
            if device.type != "cuda":
                continue
            # Host-side parts (pinned linears, CPU-expert hosts) stay on this stage's thread: MoeCpuHost
            # serializes its job ring and staging ring across the stage threads. Attention pinned to the
            # split's attention device is serialized per device by this pipeline instead, and keeps its
            # place in the route because the first stage has to run it before the recurrent states advance
            pinned = set()
            if not _foreign_device_parts_ok(module, device, pinned):
                return None
            pinned_at[i] = pinned
            pinned_all |= pinned
            if not stages or stages[-1][0] != device:
                if any(s[0] == device for s in stages):
                    return None
                stages.append((device, i if stages else 0, None))
        if len(stages) < 2:
            return None
        ends = [s[1] for s in stages[1:]] + [len(modules)]
        stages = [(s[0], s[1], e) for s, e in zip(stages, ends)]
        devices = [s[0] for s in stages]

        # A pinned device is driven inline by the modules that own it, so the route does not list it; the
        # modules of the first stage are what the states have to be advanced by before section 1 runs
        first = set(devices[:1])
        for i in range(stages[0][2]):
            first |= pinned_at.get(i, set())

        if cache is not None and cache.recurrent_state_cls is not None:
            layers = {torch.device(l.device) for l in cache.get_all_recurrent_layers().values()}
            if not layers <= (set(devices) | pinned_all):
                return None
            # Ring states decide their shift in the forward: the first section has to, so that the state can
            # advance for the next chunk before the section after it has run. A pinned device the first
            # section drives inline counts, as its layers' attention runs there
            if getattr(cache.recurrent_state_cls, "guaranteed_rollback", 0) and not (layers & first):
                return None

        draft_stage = None
        if draft_model is not None:
            if draft_model.loaded_tp or draft_model.caps.get("load_with_target") \
                    or draft_model.caps.get("draft_uses_target_cache"):
                return None
            draft_devices = {torch.device(m.device) for m, _, _ in draft_model.fwd_modules}
            draft_devices.discard(torch.device("cpu"))
            if len(draft_devices) != 1:
                return None
            draft_stage = len(stages)
            stages.append((draft_devices.pop(), 0, 0))
        return stages, draft_stage


    @staticmethod
    def _pinned_devices(model) -> set:
        """The devices the model's modules are pinned to (caps["attn_device"]), if any"""
        pinned = set()
        for entry in getattr(model, "fwd_modules", None) or []:
            if not isinstance(entry, tuple):
                continue
            device = torch.device(entry[0].device)
            if device.type == "cuda":
                _foreign_device_parts_ok(entry[0], device, pinned)
        return pinned


    def __init__(self, model, stages: list, draft_stage: int | None, on_idle = None,
                 draft_model = None):
        self.model = model
        self.stages = stages
        self.route = [s[0] for s in stages]
        self.draft_stage = draft_stage
        self.on_idle = on_idle
        self.local = threading.local()

        # One lock per device that modules are pinned to (attn_device): every stage thread may drive it
        # (a layer's attention hop runs from whichever device the layer lives on), so the lock keeps the
        # hop and the forward of one chunk from interleaving with another's
        self.device_locks = {}
        for m in (model, draft_model):
            if m is None:
                continue
            for device in self._pinned_devices(m):
                self.device_locks.setdefault(device, threading.Lock())
        for m in (model, draft_model):
            config = getattr(m, "config", None) if m is not None else None
            if config is not None:
                config.device_locks = self.device_locks

        self.cond = threading.Condition()
        self.in_flight = 0
        self.passes = 0
        self.issued = False
        # Chunks past this wait in the caller, so that a slow device can't collect the output of the ones before it
        self.max_passes = len(self.route) + 1
        self.pending_stashes = set()
        self.failed_stashes = set()
        self.failures = []
        self.failed_owners = set()
        self.queues = {device: queue.SimpleQueue() for device in self.route}
        for device, q in self.queues.items():
            threading.Thread(target = _worker, args = (q, self.local), daemon = True, name = f"prefill-{device}").start()
        self.finalizer = weakref.finalize(self, PrefillPipeline._stop, list(self.queues.values()))


    @staticmethod
    def _stop(queues: list):
        for q in queues:
            q.put(None)


    def close(self):
        self.flush()
        self.finalizer()


    @property
    def busy(self) -> bool:
        return self.in_flight > 0


    def flush(self):
        """Wait for everything issued so far to reach the end of the route. No-op on a worker, whose own step is
        part of what would be waited for (a draft model's prefill runs on one)"""
        if not self.issued or getattr(self.local, "is_worker", False):
            return
        with self.cond:
            while self.in_flight:
                self.cond.wait()
        self.issued = False
        if self.on_idle is not None:
            self.on_idle()


    def take_failures(self) -> list:
        """(owner, exception) of the items that failed since the last call"""
        with self.cond:
            failures, self.failures = self.failures, []
            return failures


    def forget_owner(self, owner):
        self.failed_owners.discard(owner)


    def _issue(self, item: _Item):
        self.issued = True
        with self.cond:
            self.in_flight += 1
        self.queues[self.route[0]].put(partial(self._run, item, 0))


    def _run(self, item: _Item, s: int):
        step = item.steps[s]
        # An item of a job that failed is walked through without running, so that what's behind it isn't held up
        if item.owner in self.failed_owners:
            item.aborted = True
        if step is not None and not item.aborted:
            try:
                step()
            except BaseException as e:
                item.aborted = True
                with self.cond:
                    if item.owner not in self.failed_owners:
                        self.failed_owners.add(item.owner)
                        self.failures.append((item.owner, e))
        # The caller of a chunk that didn't run still has to get going again
        if item.started is not None:
            item.started.set()
        s += 1
        if s < len(self.route):
            self.queues[self.route[s]].put(partial(self._run, item, s))
            return
        item.steps = None
        if item.on_done is not None:
            item.on_done(item.aborted)
        with self.cond:
            self.in_flight -= 1
            self.passes -= item.is_pass
            self.cond.notify_all()


    def prefill(self, input_ids: torch.Tensor, params: dict, owner, to_output: bool = False, then = None):
        """
        Issue a prefill of input_ids, returning once the first device has it and the recurrent states have moved
        past it, which is what the next chunk needs. to_output runs the whole forward instead of stopping at the
        last cache-writing layer, for a drafter that reads the states after the final norm. then(params) runs at
        the draft stage with the params of the finished pass, so with the exported states.
        """
        with self.cond:
            while self.passes >= self.max_passes:
                self.cond.wait()
            self.passes += 1

        model = self.model
        # The page table can move on before later devices get to this chunk
        if "block_table" in params:
            params["block_table"] = params["block_table"].clone()
        n = self.draft_stage if self.draft_stage is not None else len(self.route)
        end = len(model.fwd_modules) if to_output else model.last_kv_end
        run = dict(x = None, params = params)
        started = threading.Event()
        error = []

        def section(stage: int):
            a, b = self.stages[stage][1], min(self.stages[stage][2], end)
            if a < b:
                run["x"] = model.run_section(run["x"], run["params"], a, b, not to_output)

        def first():
            try:
                # Wake the CPU-expert workers for this pass: chunks arrive one per stage thread
                # now, so prefill_ls() no longer runs to do it once per chunk
                begin_pass = getattr(model, "begin_prefill_pass", None)
                if begin_pass is not None:
                    begin_pass()
                run["x"] = model.prepare_inputs(input_ids, params)
                for m in model._get_prefetch_layers:
                    m.prefetch(run["x"], params)
                section(0)
                # Later devices need the states as this chunk found them; the next chunk needs them past it
                run["params"] = dict(params)
                states = params.get("recurrent_states")
                if states:
                    run["params"]["recurrent_states"] = [copy.copy(r) for r in states]
                advance_recurrent_states(input_ids, params, model)
            except BaseException as e:
                error.append(e)
                item.aborted = True
            finally:
                started.set()

        def finish():
            run["x"] = None
            then(run["params"])

        steps = [first] + [partial(section, s) for s in range(1, n)]
        if self.draft_stage is not None:
            steps.append(finish if then is not None else None)
        item = _Item(steps, owner, True)
        item.started = started
        self._issue(item)
        started.wait()
        if error:
            raise error[0]


    def stash(self, state, owner) -> dict:
        """
        A checkpoint of a paired state, as of the last chunk issued. Returns the stash structure right away; its
        contents arrive as the checkpoint walks the route behind that chunk, each device's layers being copied
        once the device has finished the chunk. The position and shape of a state can be read here, while what the
        draft side has reached is only known at its own stage, behind the chunk's update.
        """
        stashed = {"position": state.position, "checkpoint_size": state.checkpoint_size}
        steps = [None] * len(self.route)
        if state.target is not None:
            snapshot = copy.copy(state.target)
            side = snapshot.stash_header()
            stashed["target"] = side
            by_stage = {}
            for k, layer in snapshot.cache.get_all_recurrent_layers().items():
                by_stage.setdefault(self.route.index(torch.device(layer.device)), {})[k] = layer
            for s, layers in by_stage.items():
                steps[s] = lambda layers = layers: side.update(snapshot.stash_layers(layers))
        if state.draft is not None:
            steps[self.draft_stage] = lambda: stashed.__setitem__("draft", state.draft.stash())
        item = _Item(steps, owner, False)
        self.pending_stashes.add(id(stashed))

        def done(aborted: bool):
            if aborted:
                self.failed_stashes.add(id(stashed))
            self.pending_stashes.discard(id(stashed))

        item.on_done = done
        self._issue(item)
        return stashed
