from __future__ import annotations
from typing import Callable
import os
import torch

from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from ..util.memory import (
    set_memory_fraction_reserve,
    set_memory_fraction_use,
    unset_memory_fraction,
    free_mem,
    device_mem_info,
    gc_paused,
)
from ..util.progress import ProgressBar
from ..util.tensor import g_tensor_cache
from ..util.device_copy import needs_bounce
from ..cache.recurrent_util import advance_recurrent_states
from .config import Config
from abc import ABC, abstractmethod


# --- row pipelining (local experiment) ------------------------------------------------------------
#
# The layer-split chain is a serial dependency, so with one batch only one device is ever busy (measured:
# sum of per-card utilisation ~= 0.8 of a single card at 8 concurrent streams). EXL3_PIPE_ROWS is the
# row budget of one microbatch: the decode batch is cut at *job* boundaries into microbatches of at most
# that many rows, and the stages are then run as a wavefront so microbatches overlap across devices.
# The cut must fall between jobs, never inside one: during a speculative verify round a job's rows are
# a causal block (each draft position attends to the ones before it), so splitting them would change
# the result. Inert (0) unless set.

_PIPE_ROWS = int(os.environ.get("EXL3_PIPE_ROWS", "0") or 0)
_PIPE_MAX_QLEN = int(os.environ.get("EXL3_PIPE_MAX_QLEN", "16") or 16)

# params entries indexed by batch row
_ROW_KEYS = (
    "block_table",
    "cache_seqlens",
    "positions",
    "position_ids",
    "recurrent_slots",
    "recurrent_slots_scan",
    "recurrent_slots_scan_in",
    "indexed_embeddings",
    "recurrent_states",
    # MTP draft stage in fwd_modules asserts target_hidden.shape[:-1] == x.shape
    "target_hidden",
)

_warned = set()
_announced = set()


def _slice_row_params(params: dict, a: int, b: int, rows: int) -> dict:
    """Shallow copy of params with every row-indexed entry narrowed to rows [a, b)"""
    p = dict(params)
    p.pop("dev_cache", None)
    p.pop("_pipe_advanced", None)
    p.pop("_pipe_input_ids", None)
    p.pop("row_boundaries", None)
    for k, v in params.items():
        if k not in _ROW_KEYS:
            continue
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == rows:
            p[k] = v[a:b]
        elif isinstance(v, list) and len(v) == rows:
            p[k] = v[a:b]
    return p


def _microbatch_bounds(rows: int, boundaries: list | None) -> list[tuple[int, int]]:
    """
    Split [0, rows) at job boundaries into at most _PIPE_ROWS rows per microbatch.

    A job is never split: a job larger than the budget becomes a microbatch on its own.
    """
    if boundaries and len(boundaries) >= 2:
        jobs = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]
    else:
        jobs = [(0, rows)]
    bounds = []
    cur_a = jobs[0][0]
    for s, e in jobs:
        if s != cur_a and e - cur_a > _PIPE_ROWS:
            bounds.append((cur_a, s))
            cur_a = s
    bounds.append((cur_a, jobs[-1][1]))
    return bounds


def _pipeline_rows(x: torch.Tensor, params: dict) -> int:
    """Number of microbatches to use for this pass, or 0 to run the plain serial chain"""
    if _PIPE_ROWS < 1:
        return 0
    # Decode passes a 2-D (rows, hidden) tensor; prefill passes (rows, seqlen, hidden). Only decode
    # is pipelined: prefill chunks are already long enough to keep every stage busy.
    if x.dim() == 2:
        q_len = 1
    elif x.dim() == 3:
        q_len = x.shape[1]
    else:
        return 0
    if q_len > _PIPE_MAX_QLEN:
        if os.environ.get("EXL3_PIPE_DEBUG"):
            key = ("shape", x.dim(), q_len)
            if key not in _announced:
                _announced.add(key)
                print(f"[pipe] skip: dim={x.dim()} q_len={q_len} (max {_PIPE_MAX_QLEN})",
                      flush=True)
        return 0
    rows = x.shape[0]
    bounds = _microbatch_bounds(rows, params.get("row_boundaries"))
    if len(bounds) < 2:
        if os.environ.get("EXL3_PIPE_DEBUG"):
            key = ("rows", rows, q_len)
            if key not in _announced:
                _announced.add(key)
                print(f"[pipe] skip: rows={rows} in {len(bounds)} microbatch(es) "
                      f"(budget {_PIPE_ROWS}, q_len={q_len})", flush=True)
        return 0
    # Sanity: any tensor we do not know about that happens to be row-shaped is suspicious. Report it
    # once so it can be classified rather than silently producing wrong results.
    if _PIPE_ROWS and os.environ.get("EXL3_PIPE_DEBUG"):
        for k, v in params.items():
            if k in _ROW_KEYS or not isinstance(v, torch.Tensor):
                continue
            if v.dim() >= 1 and v.shape[0] == rows and (k, tuple(v.shape)) not in _warned:
                _warned.add((k, tuple(v.shape)))
                print(f"[pipe] unclassified row-shaped param {k!r} {tuple(v.shape)}", flush=True)
    return _PIPE_ROWS


class Model_LSMixin(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def get_layer_instances(self, layer_idx) -> tuple:
        ...


    def _load_single(
        self,
        progressbar: bool,
        device: torch.device,
        config: Config,
        modules: list,
        verbose: bool
    ):
        pin = config.infer_params.vision_pinned and getattr(self, "component", "text") == "vision"
        with ProgressBar(f"Loading" if progressbar else None, len(modules)) as progress:
            for idx, module in enumerate(modules):
                defer = module.can_defer_load()
                with gc_paused():
                    if defer:
                        # Pinned modules leave the arena alone: their slab slices would keep whole
                        # blocks resident after the weights move to host memory
                        config.stc.begin_deferred_load(arena = not pin)
                    module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
                    if defer:
                        config.stc.end_deferred_load()
                if pin:
                    # After the deferred fills have landed: linear weights move to pinned host
                    # memory (zero-copy aliases), everything else stays put
                    module.pin_linears()
                for h in getattr(config, "moe_cpu_hosts", {}).values():
                    h.commit_module(module.key)
                progress.update(idx + 1)


    def default_load_shape_dtype(self, chunk_size):
        return (1, chunk_size), torch.long


    def default_load_params(self, chunk_size):
        return {"input_ids": torch.zeros((1, chunk_size), dtype = torch.long)}


    def _load_autosplit(
        self,
        progressbar: bool,
        reserve_per_device: list[int] | None,
        use_per_device: list[int] | None,
        active_devices: list[int],
        max_chunk_size: int,
        max_output_size: int,
        max_output_factor: int,
        callback_sync: Callable[[int, int], None],
        generator: bool,
        config: Config,
        modules: list,
        verbose: bool,
        max_batch_size: int,
        cache_weakrefs: dict,
        autosplit_no_forward: bool,
    ):
        current_device_i = 0
        backup_shape, backup_dtype = self.default_load_shape_dtype(max_chunk_size)
        dummy_state = None
        prev_load_device = None
        touched_devices = []
        params = self.default_load_params(max_chunk_size)
        # The measuring forwards below exist to observe VRAM allocation; modules with CPU-side
        # compute (CPU-offloaded experts) can skip the host work when they see this flag
        params["autosplit_measure"] = True

        # Simulate cached/recurrent path while loading with cache
        cl = [m for m in self if m.caps.get("kv_cache")]
        if cl:
            c0 = cl[0].cache_layers[0] if len(cl[0].cache_layers) else None
            if c0:
                cache = cache_weakrefs[c0.cache_id]()
                params.update({
                    "attn_mode": "flash_attn",
                    "cache": cache,
                    "past_len": 0,
                    "batch_shape": (1, max_chunk_size),
                })
                prepare_for_attn(torch.zeros((1, max_chunk_size), dtype = torch.long), params)
                rl = [m for m in self if m.caps.get("recurrent_cache")]
                if rl:
                    prepare_for_recurrence(None, params, self)

        recurrent_states = params.get("recurrent_states")

        # Per device: the budget the memory fraction enforces, and the largest transient any
        # measuring forward has needed there. A module's forward only proves that ITS transient
        # still fits; every module already on the device needs its own at runtime, so a device
        # is closed when its remaining headroom no longer covers the largest one seen
        device_budget = {}
        max_transient = {}
        # Physical margin on top of the largest measured transient: the measure is per module,
        # while a real forward also carries what the caller keeps live around it (recurrent
        # test states, gathered embeddings, allocator slack), so a device packed to exactly one
        # transient of headroom fails on the first real chunk
        autosplit_margin = int(os.environ.get("EXL3_AUTOSPLIT_MARGIN_MB", 256)) << 20
        autosplit_prepare = os.environ.get("EXL3_AUTOSPLIT_PREPARE", "1") != "0"

        with ProgressBar(f"Loading (LS)" if progressbar else None, len(modules)) as progress:

            for idx, module in enumerate(modules):

                if callback_sync: callback_sync(idx, len(modules))
                if generator: yield idx, len(modules)

                # Narrow state to max_output_size for logit output layer. When a live dummy state
                # exists, refresh the backup from it: backup_shape still holds the state that
                # ENTERED the previous module, which can have a different rank (e.g. a
                # hyper-connection stream stack ahead of a final mixer)
                is_logits_layer = module.caps.get("logits_output")
                if is_logits_layer and not autosplit_no_forward:
                    if dummy_state is not None:
                        dummy_state = dummy_state[:, :max_output_size]
                        backup_shape = dummy_state.shape
                        backup_dtype = dummy_state.dtype
                    else:
                        b, c, *rest = backup_shape
                        backup_shape = (b, min(max_output_size, c), *rest)

                while True:
                    try:
                        # Select device
                        load_device = torch.device("cpu") if module.caps.get("prefer_cpu") else \
                            torch.device(active_devices[current_device_i])
                        x_device = torch.device("cpu") if module.caps.get("prefer_cpu") or module.caps.get("x_cpu") else \
                            torch.device(active_devices[current_device_i])

                        # Set VRAM limit if new device
                        if load_device != torch.device("cpu") and load_device != prev_load_device:
                            prev_load_device = load_device
                            i = active_devices[current_device_i]
                            touched_devices.append(i)
                            i = active_devices[current_device_i]
                            if reserve_per_device is not None:
                                device_budget[i] = set_memory_fraction_reserve(reserve_per_device[i], i)
                            elif use_per_device is not None:
                                device_budget[i] = set_memory_fraction_use(use_per_device[i], i)
                            else:
                                raise RuntimeError("Logic error")

                        # (Re)create or backup hidden state (metadata)
                        if dummy_state is None:
                            dummy_state = torch.zeros(backup_shape, dtype = backup_dtype, device = x_device)
                        else:
                            backup_shape = dummy_state.shape
                            backup_dtype = dummy_state.dtype

                        # Load module
                        defer = module.can_defer_load()
                        pin = config.infer_params.vision_pinned and \
                            getattr(self, "component", "text") == "vision"
                        with gc_paused():
                            if defer:
                                # Pinned modules leave the arena alone (see _load_single)
                                config.stc.begin_deferred_load(arena = not pin)
                            module.load(load_device, max_chunk_size = max_chunk_size)
                            if defer:
                                config.stc.end_deferred_load()
                        if pin:
                            # Before the measuring forward, so the VRAM accounting reflects the
                            # pinned (host-resident) weights
                            module.pin_linears()

                        # Forward dummy state through module. The forward runs the real cached
                        # attention path, so any dequant temporaries a quantized cache layer
                        # still needs are allocated (and accounted) here
                        if self.caps.get("autosplit_load_fwd", True) and not autosplit_no_forward:
                            measure = load_device.type == "cuda"
                            # Resident state the module would otherwise create inside the
                            # measuring window (and be budgeted for twice) is allocated first
                            if autosplit_prepare:
                                for sm in module:
                                    sm.autosplit_prepare(params)
                            if measure:
                                torch.cuda.reset_peak_memory_stats(load_device)
                                alloc_before = torch.cuda.memory_allocated(load_device)
                                in_bytes = dummy_state.untyped_storage().nbytes() if dummy_state.device == load_device else 0
                                statics_before = set(g_tensor_cache.cache)
                            dummy_state = module.prepare_for_device(dummy_state, params)
                            dummy_state = module.forward(dummy_state, params)
                            for sm in module:
                                sm.autosplit_extra_measure(params)
                            if measure:
                                i = load_device.index
                                transient = max(0, torch.cuda.max_memory_allocated(load_device) - alloc_before)
                                # Net growth beyond the output replacing the input: state a hook
                                # could have allocated in autosplit_prepare instead. Small
                                # remainders are expected (first-use library workspaces, bounded
                                # selection slabs, selections threaded to the shared layers)
                                out_bytes = dummy_state.untyped_storage().nbytes() if dummy_state.device == load_device else 0
                                resident = torch.cuda.memory_allocated(load_device) - alloc_before - out_bytes + in_bytes
                                if verbose and resident > (16 << 20):
                                    new_statics = [(k.split("/")[-1], v.numel() * v.element_size())
                                                   for k, (_, v) in g_tensor_cache.cache.items() if k not in statics_before]
                                    new_statics.sort(key = lambda t: -t[1])
                                    tags = ", ".join(f"{t} {n >> 20} MiB" for t, n in new_statics[:6] if n >= (1 << 20))
                                    print(f" !! autosplit: {module.key} left {resident >> 20} MiB resident during its "
                                          f"measuring forward (budgeted as transient too)" + (f": {tags}" if tags else ""))

                                max_transient[i] = max(max_transient.get(i, 0), transient)
                                # Models with mixed layer type and dramatically different transient memory requirements
                                # per layer break the assumption that if layer n fits and subsequently layer n+1 also
                                # fits, then layer n will continue to work even with the weights of layer n+1 loaded
                                if torch.cuda.memory_allocated(load_device) + max_transient[i] > device_budget[i]:
                                    raise torch.cuda.OutOfMemoryError(
                                        f"autosplit: cuda:{i} has no headroom left for the largest "
                                        f"transient measured on it ({max_transient[i] >> 20} MiB)"
                                    )
                                # The same check against the device itself: a split budget at or
                                # above the card's size plans against VRAM the CUDA context, the
                                # loaded kernels and other processes hold (several hundred MiB),
                                # and a load that passes the budget check then fails on the first
                                # real forward. What the largest transient can still draw on is the
                                # free device memory plus the allocator's reserved-but-unallocated
                                # pool
                                free_now, _ = device_mem_info(load_device)
                                reusable = free_now + torch.cuda.memory_reserved(load_device) - \
                                    torch.cuda.memory_allocated(load_device)
                                if reusable < max_transient[i] + autosplit_margin:
                                    raise torch.cuda.OutOfMemoryError(
                                        f"autosplit: cuda:{i} has {reusable >> 20} MiB of physical headroom "
                                        f"left, below the largest transient measured on it "
                                        f"({max_transient[i] >> 20} MiB) plus the {autosplit_margin >> 20} MiB "
                                        f"margin (EXL3_AUTOSPLIT_MARGIN_MB)"
                                    )

                        # Account for max_output_factor after last layer
                        extra_dummy_out_states = None
                        if is_logits_layer:
                            extra_dummy_out_states = [
                                torch.empty_like(dummy_state)
                                for _ in range(max_output_factor - 1)
                            ]

                        # Dereference extra dummy tensors
                        extra_dummy_out_states = None

                        # We're good. For CPU-offloaded MoE layers, wait for the worker process
                        # to finish loading this module's expert weights before advancing, so
                        # load progress stays honest and rollbacks can't outrun the child
                        for h in getattr(config, "moe_cpu_hosts", {}).values():
                            h.commit_module(module.key)
                        fail = False
                        progress.update(idx + 1)

                    # We're not good
                    except Exception as e:
                        config.stc.abort_deferred_load()
                        if e.__class__.__name__ == "OutOfMemoryError" or \
                            "CUDA out of memory" in str(e) or \
                            "HIP out of memory" in str(e):
                            # Exception object will hold references to tensors so we can't free them here
                            fail = True
                            if verbose:
                                print(f" -- autosplit: {module.key} does not fit on {load_device}: {str(e).splitlines()[0][:200]}")
                        else:
                            raise

                    # Module failed to load with an OoM error, so advance to the next device if possible
                    if fail:
                        module.unload()
                        dummy_state = None
                        if "recurrent_states" in params:
                            for rs in params["recurrent_states"]:
                                rs.reset()

                        free_mem()
                        current_device_i += 1
                        if current_device_i >= len(active_devices):
                            raise RuntimeError("Insufficient VRAM in split for model and cache")
                        continue

                    # On to next module
                    break

            if callback_sync: callback_sync(len(modules), len(modules))
            if generator: yield len(modules), len(modules)

            dummy_state = None
            unset_memory_fraction(touched_devices)
            if verbose:
                for i in touched_devices:
                    print(f" -- autosplit: cuda:{i} largest measured transient {max_transient.get(i, 0) >> 20} MiB, "
                          f"{torch.cuda.mem_get_info(i)[0] >> 20} MiB free after load")

        if recurrent_states is not None:
            for rs in recurrent_states:
                rs.free()

        config.stc.close()
        self.active_devices = active_devices

        # Python will not run anything in an async function without at least one yield statement
        if 'yield' in locals():
            yield


    def prefill_ls(
        self,
        x: torch.Tensor,
        params: dict,
    ):
        for h in getattr(self.config, "moe_cpu_hosts", {}).values():
            h.begin_pass()
        for module, instance, idx in self.fwd_modules:
            params["layer_instance"] = instance
            pf = (idx, instance) == self.last_kv_module_idx_instance
            params["prefill"] = pf
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if pf:
                break
        del params["prefill"]
        return None


    def forward_ls(
        self,
        x: torch.Tensor,
        params: dict,
    ):
        for h in getattr(self.config, "moe_cpu_hosts", {}).values():
            h.begin_pass()
        if _pipeline_rows(x, params):
            return self._forward_ls_pipelined(x, params)
        for module, instance, idx in self.fwd_modules:
            params["layer_instance"] = instance
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x


    def _forward_ls_pipelined(
        self,
        x: torch.Tensor,
        params: dict,
    ):
        """
        Row-pipelined layer-split forward (see _pipeline_rows).

        The layer-split chain is a serial dependency: each stage waits for the previous one, so with a
        single batch only one device is ever busy. Splitting the batch into microbatches at job
        boundaries and issuing the stages in wavefront order lets stage s work on microbatch m while
        stage s+1 works on microbatch m-1, since microbatches are independent of each other. Ordering
        across devices is explicit (one event per microbatch, re-armed at every stage boundary), so the
        device-to-device moves can stay asynchronous.
        """
        rows = x.shape[0]
        q_len = 1 if x.dim() == 2 else x.shape[1]
        bounds = _microbatch_bounds(rows, params.get("row_boundaries"))
        if len(bounds) < 2:
            for module, instance, idx in self.fwd_modules:
                params["layer_instance"] = instance
                if x.dim() == 3 and module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                    x = x[..., -num:, :].contiguous()
                x = module.prepare_for_device(x, params)
                x = module.forward(x, params)
            return x

        if os.environ.get("EXL3_PIPE_DEBUG"):
            key = (rows, q_len, len(bounds))
            if key not in _announced:
                _announced.add(key)
                print(f"[pipe] active rows={rows} q_len={q_len} microbatches={len(bounds)}",
                      flush=True)

        sub = [_slice_row_params(params, a, b, rows) for a, b in bounds]
        xs = [x[a:b] for a, b in bounds]
        evs = [None] * len(bounds)

        for module, instance, idx in self.fwd_modules:
            dev = module.device
            device = torch.device(dev) if not isinstance(dev, torch.device) else dev
            logits_out = module.caps.get("logits_output")
            for mi in range(len(bounds)):
                p = sub[mi]
                p["layer_instance"] = instance
                xi = xs[mi]
                if xi.dim() == 3 and logits_out and (num := p.get("last_tokens_only")):
                    xi = xi[..., -num:, :].contiguous()
                if device.type != "cuda":
                    xs[mi] = module.forward(module.prepare_for_device(xi, p), p)
                    continue
                dst = torch.cuda.current_stream(device)
                if evs[mi] is not None:
                    dst.wait_event(evs[mi])
                if xi.device != device and xi.device.type == "cuda" and \
                        not needs_bounce(xi.device, device):
                    # Direct peer move, kept asynchronous: issued on the source stream (so it is
                    # ordered after the producing stage) and fenced into the destination stream,
                    # instead of draining the host as a blocking copy would
                    src = torch.cuda.current_stream(xi.device)
                    if evs[mi] is not None:
                        src.wait_event(evs[mi])
                    xi = xi.to(device, non_blocking = True)
                    ev_copy = torch.cuda.Event()
                    ev_copy.record(src)
                    dst.wait_event(ev_copy)
                xi = module.prepare_for_device(xi, p)
                xi = module.forward(xi, p)
                ev = torch.cuda.Event()
                ev.record(dst)
                xs[mi] = xi
                evs[mi] = ev

        # Join: everything downstream (sampling, cache bookkeeping) runs on the output device's stream
        for mi in range(len(bounds)):
            xi = xs[mi]
            if xi.device.type == "cuda" and evs[mi] is not None:
                torch.cuda.current_stream(xi.device).wait_event(evs[mi])
        out = xs[0] if len(xs) == 1 else torch.cat(xs, dim=0)

        # Drafting reads the exported hidden states, and slices them by the row range of each job,
        # so they must be merged back into the original row order. Each microbatch accumulated its
        # own list (the params copies are independent), so concatenate entry by entry: entry i is
        # produced by the same layer in every microbatch, on the same device.
        merged = None
        for p in sub:
            s = p.get("export_states")
            if not s:
                continue
            if merged is None:
                merged = list(s)
            else:
                for i, t in enumerate(s):
                    merged[i] = torch.cat((merged[i], t), dim = 0)
        if merged is not None:
            params["export_states"] = merged

        # Recurrent state bookkeeping is per microbatch: the staged scan inputs are indexed by row
        # within the pass that wrote them
        if params.get("recurrent_states"):
            for (a, b), p in zip(bounds, sub):
                advance_recurrent_states(
                    torch.empty((b - a, q_len), dtype=torch.long), p, self)
            params["_pipe_advanced"] = True
        return out

