from __future__ import annotations
import copy
import queue
import threading
import weakref
from functools import partial

import torch

from ..cache.recurrent_util import advance_recurrent_states
from ..util.device_copy import to_device

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

A device the modules are pinned to (caps["attn_device"]) can become a stage of its own when the modules it
belongs to can be split into phases (see plan()): a block's pre_attn runs on its own device, the attention
on the pinned device's stage, post_attn back on the block's device, so the pinned device is never idle while
the stage that owns a block is in its MLP. A device may then host more than one stage, with the other
devices' stages between them. Where that isn't possible the pinned part stays inline: the stage that owns
it drives it from its own thread, under one lock per pinned device (device_locks, installed on the config).

Either way an item can't overtake one issued before it: it walks the whole route, running one step at each
stage, and its step at each stage is queued behind the item issued before it there.

Only prefill goes through here. Model.forward and everything else that uses the devices, or reads what the items
write, waits for the route to clear first (flush) and then has them to itself.
"""


def _worker(tasks: queue.SimpleQueue, local: threading.local):
    local.is_worker = True
    with torch.inference_mode():
        while (task := tasks.get()) is not None:
            task()


class _Item:

    def __init__(self, steps: list, devices: list, owner, is_pass: bool):
        self.steps = steps
        # The device each step runs on: consecutive steps may be on the same one, and the route may return to
        # a device (a split block's pre_attn, its attention, and its post_attn)
        self.devices = devices
        self.owner = owner
        self.is_pass = is_pass
        self.aborted = False
        self.on_done = None
        self.started = None
        # Index of the step after which the caller can go on: past the recurrent states' advance. Everything
        # before it happens before the caller returns, so a caller that fails here never strands it
        self.ready_at = 0


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


def _split_block(module):
    """
    The attention device of a block the pipeline can run as three steps (pre_attn on the block's device,
    attention on the pinned device, post_attn on the block's device), or None. The steps are the block's
    forward split at the attention call, so what has to hold is that the attention really is the only part
    on another device.
    """
    from ..modules.transformer import TransformerBlock

    if type(module) is not TransformerBlock or module.attn is None:
        return None
    attn_device = torch.device(module.attn.device)
    if attn_device.type != "cuda" or attn_device == torch.device(module.device):
        return None
    return attn_device


class Stage:
    """A run of steps on one device. The route is the stages in order; a device may host more than one run so
    that steps on other devices can sit between them (a split block's pre_attn, its attention, its post_attn)"""

    __slots__ = ("device", "steps")

    def __init__(self, device, steps: list):
        self.device = device
        self.steps = steps

    def __eq__(self, other):
        return isinstance(other, Stage) and (self.device, self.steps) == (other.device, other.steps)

    def __repr__(self):
        return f"Stage({self.device}, {self.steps})"


class PrefillPipeline:

    @classmethod
    def plan(cls, model, cache, draft_model = None) -> tuple[list, int | None, int] | None:
        """
        The route through the devices for this model and draft model, as (stages, draft stage, advance stage):
        each stage is a Stage, the steps one device runs, in order; the draft stage is an index into the route,
        past the target's stages, and the advance stage is the one after which the recurrent states can move
        past the chunk. None if the split can't run this way.
        """
        modules = getattr(model, "fwd_modules", None)
        if not modules or model.loaded_tp:
            return None

        pinned_at = {}
        pinned_all = set()
        split_at = {}
        for i, entry in enumerate(modules):
            device = torch.device(entry[0].device)
            if device.type != "cuda":
                continue
            pinned = set()
            if not _foreign_device_parts_ok(entry[0], device, pinned):
                return None
            pinned_at[i] = pinned
            pinned_all |= pinned
            if pinned:
                split_at[i] = _split_block(entry[0])

        # A block whose attention is pinned to another device can run its attention on that device's own
        # stage, between the block's pre_attn and post_attn steps. Every other pinned part must not stay
        # inline then: it would run on a stage thread under a lock while that device's stage thread works,
        # with nothing ordering the two. So either all of them split, or the plan keeps the inline route
        split = bool(pinned_all) and all(
            split_at.get(i) is not None and pinned_at[i] == {split_at[i]}
            for i in pinned_at if pinned_at[i]
        )

        if not split:
            # Inline: a stage is a run of modules on one device, and the pinned parts are driven by the
            # stage that owns them, under the per-device locks (see PrefillPipeline.device_locks)
            ranged = []
            for i, entry in enumerate(modules):
                device = torch.device(entry[0].device)
                if device.type != "cuda":
                    continue
                if not ranged or ranged[-1][0] != device:
                    if any(r[0] == device for r in ranged):
                        return None
                    ranged.append((device, i if ranged else 0, None))
            if len(ranged) < 2:
                return None
            ends = [r[1] for r in ranged[1:]] + [len(modules)]
            stages = [
                Stage(device, [("section", a, e)])
                for (device, a, _), e in zip(ranged, ends)
            ]
        else:
            # Split: the pinned attention device gets stages of its own, one per block, between the steps
            # that run on the block's device. Consecutive steps on the same device share a stage
            stages = []
            start = 0
            i = 0
            while i < len(modules):
                device = torch.device(modules[i][0].device)
                if device.type != "cuda":
                    i += 1
                    continue
                attn_device = split_at.get(i) if pinned_at.get(i) else None
                if attn_device is not None:
                    nxt = modules[i + 1][0] if i + 1 < len(modules) else None
                    defer = type(nxt) is type(modules[i][0]) and torch.device(nxt.device) == device
                    cls._add(stages, device, ("section", start, i))
                    cls._add(stages, device, ("pre", i))
                    cls._add(stages, attn_device, ("attn", i))
                    cls._add(stages, device, ("post", i, defer))
                    i += 1
                    start = i
                else:
                    j = i + 1
                    while (
                        j < len(modules) and not split_at.get(j) and
                        torch.device(modules[j][0].device) in (device, torch.device("cpu"))
                    ):
                        j += 1
                    cls._add(stages, device, ("section", start, j))
                    i = j
                    start = j
            if start < len(modules) and stages:
                # Modules after the last one on a device (a final norm, the head) stay on its stage
                cls._add(stages, stages[-1].device, ("section", start, len(modules)))
            if len({s.device for s in stages}) < 2 and len(stages) < 2:
                return None

        devices = {s.device for s in stages}

        advance_after = 0
        if cache is not None and cache.recurrent_state_cls is not None:
            layers = {torch.device(l.device) for l in cache.get_all_recurrent_layers().values()}
            if not layers <= (devices | pinned_all):
                return None
            if getattr(cache.recurrent_state_cls, "guaranteed_rollback", 0):
                # Ring states decide their shift in the forward, so a stage that runs a layer on the
                # state's device has to run before the states advance for the next chunk. A pinned device
                # the first stage drives inline counts; otherwise the advance waits for the stage that
                # first reaches the states' device, and the stages behind it read the copied states
                first = {stages[0].device}
                if not split:
                    a, b = stages[0].steps[0][1], stages[0].steps[0][2]
                    for i in range(a, b):
                        first |= pinned_at.get(i, set())
                if layers & first:
                    advance_after = 0
                elif not split:
                    return None
                else:
                    covered = set()
                    advance_after = None
                    for s, stage in enumerate(stages):
                        covered.add(stage.device)
                        if layers & covered:
                            advance_after = s
                            break
                    if advance_after is None:
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
            stages.append(Stage(draft_devices.pop(), []))
        return stages, draft_stage, advance_after


    @staticmethod
    def _add(stages: list, device, step: tuple):
        """Append a step to the last stage if it is on that device, else start a stage (skipping empty sections)"""
        if step[0] == "section" and step[1] >= step[2]:
            return
        if stages and stages[-1].device == device:
            stages[-1].steps.append(step)
        else:
            stages.append(Stage(device, [step]))


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


    def __init__(self, model, stages: list, draft_stage: int | None, advance_after: int = 0,
                 on_idle = None, draft_model = None):
        self.model = model
        self.stages = stages
        self.route = [s.device for s in stages]
        self.draft_stage = draft_stage
        self.advance_after = advance_after
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
        self.max_passes = len({s.device for s in stages}) + 1
        self.pending_stashes = set()
        self.failed_stashes = set()
        self.failures = []
        self.failed_owners = set()
        self.queues = {device: queue.SimpleQueue() for device in dict.fromkeys(self.route)}
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


    def _last_stage(self, device) -> int:
        """The last stage on device: what a checkpoint of the layers there has to be ordered behind"""
        return max(i for i, d in enumerate(self.route) if d == device)


    def forget_owner(self, owner):
        self.failed_owners.discard(owner)


    def _issue(self, item: _Item):
        self.issued = True
        with self.cond:
            self.in_flight += 1
        self.queues[item.devices[0]].put(partial(self._run, item, 0))


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
        if item.started is not None and s >= item.ready_at:
            item.started.set()
        s += 1
        if s < len(item.devices):
            self.queues[item.devices[s]].put(partial(self._run, item, s))
            return
        item.steps = None
        item.devices = None
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
        prefill_only = not to_output
        mods = model.fwd_modules
        run = dict(x = None, y = None, params = params)
        started = threading.Event()
        error = []

        def section(a: int, b: int):
            b = min(b, end)
            if a < b:
                run["x"] = model.run_section(run["x"], run["params"], a, b, prefill_only)

        def pre(i: int):
            # A split block's input norm and residual fusion, on the block's device. The attention takes over
            # on its own device (attn step) and post_attn finishes the block back here
            module, instance, idx = mods[i]
            run["params"]["layer_instance"] = instance
            if prefill_only:
                run["params"]["prefill"] = (idx, instance) == getattr(model, "last_kv_module_idx_instance", None)
            x = module.prepare_for_device(run["x"], params = run["params"])
            run["x"], run["y"] = module.pre_attn(x, run["params"])

        def attn(i: int):
            module = mods[i][0]
            run["y"] = module.attn.forward(module.attn.prepare_for_device(run["y"], run["params"]), run["params"])

        def post(i: int, defer: bool):
            module, instance, idx = mods[i]
            p = run["params"]
            # A block may leave its final residual add to the next block's input norm (same device)
            p["resid_defer"] = defer and not prefill_only
            run["y"] = to_device(run["y"], module.device)
            run["x"] = module.post_attn(run["x"], run["y"], p)
            run["y"] = None

        steps = []
        devices = []
        ready_at = 0

        def add(device, fn):
            steps.append(fn)
            devices.append(device)

        def head():
            try:
                # Wake the CPU-expert workers for this pass: chunks arrive one per stage thread
                # now, so prefill_ls() no longer runs to do it once per chunk
                begin_pass = getattr(model, "begin_prefill_pass", None)
                if begin_pass is not None:
                    begin_pass()
                run["x"] = model.prepare_inputs(input_ids, params)
                for m in model._get_prefetch_layers:
                    m.prefetch(run["x"], params)
            except BaseException as e:
                error.append(e)
                item.aborted = True

        def advance():
            # Later devices need the states as this chunk found them; the next chunk needs them past it
            run["params"] = dict(params)
            states = params.get("recurrent_states")
            if states:
                run["params"]["recurrent_states"] = [copy.copy(r) for r in states]
            advance_recurrent_states(input_ids, params, model)

        add(self.stages[0].device, head)
        for s in range(n):
            for step in self.stages[s].steps:
                kind = step[0]
                if kind != "section" and prefill_only and step[1] >= end:
                    # Past the last cache-writing layer: prefill stops there (the last layer returns after
                    # its attention, which post_attn does from params["prefill"])
                    continue
                if kind == "section":
                    add(self.stages[s].device, partial(section, step[1], step[2]))
                elif kind == "pre":
                    add(self.stages[s].device, partial(pre, step[1]))
                elif kind == "attn":
                    add(self.stages[s].device, partial(attn, step[1]))
                elif kind == "post":
                    add(self.stages[s].device, partial(post, step[1], step[2]))
                else:
                    add(self.stages[s].device, None)
            if s == self.advance_after:
                add(self.stages[s].device, advance)
                ready_at = len(steps) - 1

        def finish():
            run["x"] = None
            then(run["params"])

        if self.draft_stage is not None:
            add(self.stages[self.draft_stage].device, finish if then is not None else None)

        item = _Item(steps, devices, owner, True)
        item.started = started
        item.ready_at = ready_at
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
        devices = list(self.route)
        if state.target is not None:
            snapshot = copy.copy(state.target)
            side = snapshot.stash_header()
            stashed["target"] = side
            by_stage = {}
            for k, layer in snapshot.cache.get_all_recurrent_layers().items():
                # The last stage on the layer's device: this chunk's work there is behind it by then
                by_stage.setdefault(self._last_stage(torch.device(layer.device)), {})[k] = layer
            for s, layers in by_stage.items():
                steps[s] = lambda layers = layers: side.update(snapshot.stash_layers(layers))
        if state.draft is not None:
            steps[self.draft_stage] = lambda: stashed.__setitem__("draft", state.draft.stash())
        item = _Item(steps, devices, owner, False)
        self.pending_stashes.add(id(stashed))

        def done(aborted: bool):
            if aborted:
                self.failed_stashes.add(id(stashed))
            self.pending_stashes.discard(id(stashed))

        item.on_done = done
        self._issue(item)
        return stashed
