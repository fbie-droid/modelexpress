# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a staged XOR delta into a live vLLM model in place.

When a model's post-load processing only rearranges bits, processing the XOR
delta like weights and XORing the result into live storage gives the same bytes
as a reload of the target. Constant parts of processing cancel because zeros are
processed as well. Routed experts take a batched path from the staged bytes;
every other changed tensor goes through the layerwise reload with the final copy
replaced by an XOR.
"""

from __future__ import annotations

import logging
import math
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modelexpress.refit.reshard.slice_plan import resolve_slice
from modelexpress.refit.reshard.types import UnsupportedReshard
from modelexpress_rl import envs as rl_envs
from modelexpress_rl.inference.staging import StagedDelta, StagedTensor
from modelexpress_rl.utils import index_checkpoint_tensors

logger = logging.getLogger(__name__)

_EXPERT = re.compile(
    r"\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj|w1|w3|w2)\.([A-Za-z_]+)$"
)
_PER_EXPERT = re.compile(r"\.(?:mlp|block_sparse_moe)\.experts\.\d+\.")
# Checkpoint projection names, both the gate/up/down and the w1/w3/w2 conventions.
_PROJECTIONS = {
    "gate_proj": "gate",
    "w1": "gate",
    "up_proj": "up",
    "w3": "up",
    "down_proj": "down",
    "w2": "down",
}
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_ARRANGED_KINDS = ("weight_packed", "weight_scale", "weight")
_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class InPlaceUnsupported(RuntimeError):
    """The delta cannot be installed in place; reload the target instead."""


def raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return a byte view that shares storage with ``tensor``."""
    if not tensor.is_contiguous():
        raise InPlaceUnsupported("live tensor is not contiguous")
    return tensor.reshape(-1).view(torch.uint8)


_SAME_SIZE_INTEGER = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """Return a same-size integer view of ``tensor``, valid for any strides."""
    integer = _SAME_SIZE_INTEGER.get(tensor.element_size())
    if integer is None:
        raise InPlaceUnsupported(f"cannot view {tensor.dtype} as integers")
    return tensor.view(integer)


def same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and torch.equal(bits(left), bits(right))


def xor_into(target: torch.Tensor, value: torch.Tensor) -> None:
    """XOR the bits of ``value`` into ``target`` in place."""
    if target.shape == value.shape and target.element_size() == value.element_size():
        bits(target).bitwise_xor_(bits(value))
        return
    # Processing may change the shape of a tensor but not its storage size.
    target_bytes = raw_bytes(target)
    value_bytes = value.contiguous().reshape(-1).view(torch.uint8)
    if target_bytes.numel() != value_bytes.numel():
        raise InPlaceUnsupported("processed tensor size differs from live storage")
    target_bytes.bitwise_xor_(value_bytes)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def layer_key(name: str) -> int | None:
    """Decoder layer index of a dotted tensor or module name, if any."""
    match = _LAYER.search(f".{name}.")
    return int(match[1]) if match else None


def is_per_expert(name: str) -> bool:
    return _PER_EXPERT.search(name) is not None


@contextmanager
def _quiet_readonly_buffers() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        yield


def device_offsets(positions: np.ndarray, device: torch.device) -> torch.Tensor:
    """Staged byte offsets as int64 on ``device``, sent at their stored width."""
    if positions.dtype == np.uint32:
        return torch.from_numpy(positions.view(np.int32)).to(device).to(torch.int64) & 0xFFFFFFFF
    return torch.from_numpy(positions.view(np.int64)).to(device)


def staged_bytes(
    nbytes: int,
    staged: StagedTensor | None,
    device: torch.device,
    *,
    chunk: int = 1 << 24,
) -> torch.Tensor:
    """A tensor's full XOR delta as bytes on ``device``; zeros when unstaged."""
    if device.type == "cpu":
        # Untouched pages of a zeroed host array stay unallocated.
        data = np.zeros(nbytes, dtype=np.uint8)
        if staged is not None:
            data[staged.positions.astype(np.intp, copy=False)] = staged.values
        return torch.from_numpy(data)
    data = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    if staged is not None:
        for start in range(0, staged.positions.size, chunk):
            offsets = device_offsets(staged.positions[start : start + chunk], device)
            data[offsets] = torch.from_numpy(staged.values[start : start + chunk]).to(device)
    return data


def shard_nonzero(
    positions: np.ndarray,
    *,
    rows: int,
    row_bytes: int,
    shard_rows: bool,
    tp_size: int,
    tp_rank: int,
) -> tuple[slice, int]:
    """Return the slice of sorted offsets in this rank's row shard and its start.

    Column shards cannot be selected by one slice; callers filter them instead.
    """
    if not shard_rows:
        return slice(0, positions.size), 0
    shard_bytes = rows // tp_size * row_bytes
    begin = tp_rank * shard_bytes
    first, last = np.searchsorted(positions, [begin, begin + shard_bytes])
    return slice(int(first), int(last)), begin


def scatter_expert_shards(
    scratch: torch.Tensor,
    staged: StagedDelta,
    names: list[str | None],
    *,
    rows: int,
    cols: int,
    itemsize: int,
    shard_rows: bool,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Write this rank's shard of every expert's XOR bytes into ``scratch``.

    ``scratch`` is uint8 ``[experts, shard_bytes]`` and ``names[e]`` names expert
    ``e``'s checkpoint tensor, or ``None`` when it is unchanged.
    """
    row_bytes = cols * itemsize
    shard_bytes = scratch.shape[1]
    positions, values, experts, counts, offsets = [], [], [], [], []
    for expert, name in enumerate(names):
        if name is None:
            continue
        tensor = staged.tensor(name)
        selected, begin = shard_nonzero(
            tensor.positions,
            rows=rows,
            row_bytes=row_bytes,
            shard_rows=shard_rows,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        count = selected.stop - selected.start
        if count <= 0:
            continue
        positions.append(tensor.positions[selected])
        values.append(tensor.values[selected])
        experts.append(expert)
        counts.append(count)
        offsets.append(begin)
    if not positions:
        return
    device = scratch.device
    offset = device_offsets(np.concatenate(positions), device)
    value = torch.from_numpy(np.concatenate(values)).to(device)
    repeat = torch.tensor(counts, device=device)
    expert = torch.repeat_interleave(torch.tensor(experts, device=device), repeat)
    offset = offset - torch.repeat_interleave(torch.tensor(offsets, device=device), repeat)
    if not shard_rows:
        shard_row_bytes = row_bytes // tp_size
        column = offset % row_bytes - tp_rank * shard_row_bytes
        keep = (column >= 0) & (column < shard_row_bytes)
        offset = (offset // row_bytes) * shard_row_bytes + column
        offset, value, expert = offset[keep], value[keep], expert[keep]
    scratch.view(-1)[expert * shard_bytes + offset] = value


@dataclass(frozen=True)
class DirectCopy:
    """Where vLLM loads part of one checkpoint tensor into a live parameter.

    ``box`` bounds that part in each checkpoint dimension. ``offset`` and
    ``strides`` place it in the parameter's storage, in elements.
    """

    param: str
    box: tuple[tuple[int, int], ...]
    offset: int
    strides: tuple[int, ...]


def direct_copies(
    capture: Any,
    layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
    metadata: dict[str, dict],
    params: dict[str, torch.Tensor],
    excluded: set[str],
) -> dict[str, tuple[DirectCopy, ...]]:
    """Checkpoint tensors that vLLM loads as plain byte copies into live parameters.

    ``capture`` records the view each loader takes of a checkpoint tensor and
    where it copies that view. A tensor qualifies when every copy keeps its
    dtype, resolves to a box, and lands in a live parameter that is not
    excluded and still has its load-time layout.
    """
    unsupported = set(capture.unsupported)
    by_source: dict[str, list[Any]] = defaultdict(list)
    for copy in capture.copies:
        by_source[copy.src_name].append(copy)
    plan: dict[str, tuple[DirectCopy, ...]] = {}
    for name, copies in by_source.items():
        info = metadata.get(name)
        dtype = _TORCH_DTYPES.get(info["dtype"]) if info is not None else None
        if name in unsupported or dtype is None:
            continue
        entries = []
        for copy in copies:
            param = params.get(copy.param_name)
            if (
                param is None
                or copy.param_name in excluded
                or copy.dest_dtype != dtype
                or param.dtype != dtype
                or layout.get(copy.param_name) != (tuple(param.shape), param.dtype)
                or not param.is_contiguous()
            ):
                break
            try:
                box, strides = resolve_slice(copy, info["shape"])
            except UnsupportedReshard:
                break
            entries.append(
                DirectCopy(
                    copy.param_name,
                    tuple((int(low), int(high)) for low, high in box),
                    int(copy.dest_offset),
                    tuple(int(stride) for stride in strides),
                )
            )
        else:
            plan[name] = tuple(entries)
    return plan


def direct_region(tensor: torch.Tensor, copy: DirectCopy) -> torch.Tensor:
    """The integer view of the part of ``tensor`` that ``copy`` writes."""
    view = bits(tensor)
    return torch.as_strided(
        view,
        [high - low for low, high in copy.box],
        copy.strides,
        view.storage_offset() + copy.offset,
    )


def xor_direct(
    target: torch.Tensor,
    staged: StagedTensor,
    copy: DirectCopy,
    *,
    shape: list[int],
    dtype: torch.dtype,
    chunk: int = 1 << 24,
) -> None:
    """XOR one tensor's staged bytes into the part of ``target`` that ``copy`` writes.

    Only the checkpoint rows inside the copy's box are built, on the target's
    device.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    row_bytes = itemsize * math.prod(shape[1:])
    low, high = copy.box[0] if shape else (0, 1)
    begin, end = low * row_bytes, high * row_bytes
    first, last = (int(index) for index in np.searchsorted(staged.positions, [begin, end]))
    window = torch.zeros(end - begin, dtype=torch.uint8, device=target.device)
    for start in range(first, last, chunk):
        stop = min(start + chunk, last)
        offsets = device_offsets(staged.positions[start:stop], target.device) - begin
        window[offsets] = torch.from_numpy(staged.values[start:stop]).to(target.device)
    source = window.view(dtype).view([high - low, *shape[1:]] if shape else [])
    if shape:
        source = source[(slice(None), *(slice(low, high) for low, high in copy.box[1:]))]
    direct_region(target, copy).bitwise_xor_(bits(source))


def _is_deferred_attention(module: torch.nn.Module) -> bool:
    try:
        from vllm.model_executor.layers.attention import is_deferred_attention_layer
    except ImportError:
        return False
    return is_deferred_attention_layer(module)


def _routed_experts_class() -> type | None:
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    except ImportError:
        return None
    return RoutedExperts


def routed_expert_layers(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """Map decoder layer indices to their routed-expert modules."""
    routed = _routed_experts_class()
    found: dict[int, torch.nn.Module] = {}
    for path, module in model.named_modules():
        if routed is not None and not isinstance(module, routed):
            continue
        if routed is None and type(module).__name__ != "RoutedExperts":
            continue
        key = layer_key(path)
        if key is not None:
            found.setdefault(key, module)
    return found


def _expert_param(meta: dict, prefix: str, kind: str) -> str | None:
    """Name of the load-time parameter holding one checkpoint tensor kind."""
    # Some methods load packed weights into ``w13_weight`` rather than ``w13_weight_packed``.
    names = [f"{prefix}_{kind}"] + ([f"{prefix}_weight"] if kind == "weight_packed" else [])
    return next((name for name in names if name in meta), None)


def arrange_expert_delta(
    layer: torch.nn.Module,
    staged: StagedDelta,
    expert_names: dict[tuple[int, str, str], str],
    metadata: dict[str, dict],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Lay out one layer's expert XOR bytes as its pre-processing parameters.

    Gate and up shard their output rows, down shards its input columns, and the
    loader transposes packed weights when the parameter says so, as vLLM's
    routed-expert weight loader does.
    """
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    meta, _buffers = get_layerwise_info(layer).restore_metadata
    parallel = layer.moe_config.moe_parallel_config
    if getattr(parallel, "ep_size", 1) != 1:
        raise InPlaceUnsupported("expert parallelism is not supported")
    tp_size, tp_rank = parallel.tp_size, layer.moe_config.tp_rank
    present = {kind for (_expert, _proj, kind) in expert_names}
    params = {
        kind: (w13, w2)
        for kind in _ARRANGED_KINDS
        if kind in present
        and (w13 := _expert_param(meta, "w13", kind)) is not None
        and (w2 := _expert_param(meta, "w2", kind)) is not None
    }
    changed = {kind for (_expert, _proj, kind), name in expert_names.items() if name in staged}
    if not params or changed - set(params):
        raise InPlaceUnsupported(f"unsupported expert tensors {sorted(changed - set(params))}")
    first = next(iter(params.values()))[0]
    transposed = bool(getattr(meta[first], "is_transposed", False))
    experts = meta[first].shape[0]
    arranged: dict[str, torch.Tensor] = {}
    for kind, (w13_name, w2_name) in params.items():
        parts = {}
        for proj in ("gate", "up", "down"):
            sample = expert_names.get((0, proj, kind))
            if sample is None or len(metadata[sample]["shape"]) != 2:
                raise InPlaceUnsupported(f"expert {proj} {kind} is not a 2-D tensor")
            rows, cols = metadata[sample]["shape"]
            dtype = _TORCH_DTYPES[metadata[sample]["dtype"]]
            itemsize = torch.empty((), dtype=dtype).element_size()
            shard_rows = proj != "down"
            shard = (rows // tp_size, cols) if shard_rows else (rows, cols // tp_size)
            scratch = torch.zeros((experts, shard[0] * shard[1] * itemsize), dtype=torch.uint8, device=device)
            names = [expert_names.get((expert, proj, kind)) for expert in range(experts)]
            scatter_expert_shards(
                scratch,
                staged,
                [name if name in staged else None for name in names],
                rows=rows,
                cols=cols,
                itemsize=itemsize,
                shard_rows=shard_rows,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            tensor = scratch.view(dtype).view(experts, *shard)
            parts[proj] = tensor.transpose(1, 2) if transposed else tensor
        w13 = torch.cat([parts["gate"], parts["up"]], dim=2 if transposed else 1).contiguous()
        for name, tensor in ((w13_name, w13), (w2_name, parts["down"].contiguous())):
            if tuple(tensor.shape) != tuple(meta[name].shape) or tensor.dtype != meta[name].dtype:
                raise InPlaceUnsupported(
                    f"{name} layout {tuple(tensor.shape)} differs from {tuple(meta[name].shape)}"
                )
            arranged[name] = tensor
    return arranged


def process_like_weights(
    layer: torch.nn.Module, arranged: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Run the layer's post-load processing on ``arranged`` and restore the layer.

    Mirrors vLLM's layerwise reload of one layer, but returns the processed
    tensors instead of copying them into live storage.
    """
    from vllm.model_executor.model_loader.reload.layerwise import (
        _place_kernel_tensors,
        get_layerwise_info,
    )
    from vllm.model_executor.model_loader.reload.meta import (
        SKIP_TENSORS,
        materialize_layer,
        restore_layer_on_meta,
    )
    from vllm.model_executor.model_loader.reload.utils import (
        get_layer_params_buffers,
        get_layer_tensors,
    )

    info = get_layerwise_info(layer)
    info.kernel_tensors = get_layer_params_buffers(layer)
    info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)
    # Processing rebuilds the kernel around the tensors it processes, but those
    # hold the processed delta here. Restore the layer's and method's state so
    # the kernel keeps reading the live storage this delta is XORed into.
    containers = ("_parameters", "_buffers", "_modules")
    layer_state = {name: value for name, value in vars(layer).items() if name not in containers}
    method = layer.quant_method
    method_state = dict(vars(method))
    try:
        restore_layer_on_meta(layer, info)
        materialize_layer(layer, info)
        for name, tensor in get_layer_tensors(layer).items():
            if name in arranged:
                tensor.data.copy_(arranged[name])
            elif name not in SKIP_TENSORS:
                # Live tensors that reload never moves to meta stay untouched.
                tensor.data.zero_()
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")
        method.process_weights_after_loading(layer)
        return {name: getattr(layer, name).data for name in arranged}
    finally:
        _place_kernel_tensors(layer, info)
        for name in [name for name in vars(layer) if name not in containers]:
            if name not in layer_state:
                delattr(layer, name)
        vars(layer).update(layer_state)
        vars(method).clear()
        vars(method).update(method_state)
        info.reset()


@dataclass(frozen=True)
class CheckpointCatalog:
    """A checkpoint's tensor names by decoder layer, built once per layout."""

    layers: dict[str, int | None]
    experts: dict[int, dict[tuple[int, str, str], str]]
    per_expert: frozenset[str]
    # Per-expert tensors whose projection or kind is not recognized.
    unmatched_experts: frozenset[str]
    # Every other tensor, in checkpoint order.
    generic: tuple[str, ...]

    @classmethod
    def of(cls, metadata: dict[str, dict]) -> CheckpointCatalog:
        layers: dict[str, int | None] = {}
        experts: dict[int, dict[tuple[int, str, str], str]] = defaultdict(dict)
        per_expert, unmatched, generic = set(), set(), []
        for name in metadata:
            layers[name] = layer_key(name)
            if not is_per_expert(name):
                generic.append(name)
                continue
            per_expert.add(name)
            match = _EXPERT.search(name)
            if match is None:
                unmatched.add(name)
                continue
            key = (int(match[2]), _PROJECTIONS[match[3]], match[4])
            experts[int(match[1])][key] = name
        return cls(layers, dict(experts), frozenset(per_expert), frozenset(unmatched), tuple(generic))


class InPlaceInstaller:
    """Install staged deltas into one rank's live model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        generic_reload: Callable[[Callable[[Any], None]], None],
        capture_layout: Callable[[list[tuple[str, torch.dtype, tuple[int, ...]]]], Any]
        | None = None,
        refresh_attention: Callable[[list[torch.nn.Module]], None] | None = None,
        agree: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._generic_reload = generic_reload
        # Direct XOR needs the load layout and a way to rerun attention processing.
        self._capture_layout = capture_layout
        self._refresh_attention = refresh_attention
        # Per checkpoint layout: copies of each captured tensor, and every tensor
        # whose capture was attempted.
        self._direct_plan: (
            tuple[dict[str, dict], dict[str, tuple[DirectCopy, ...]], set[str]] | None
        ) = None
        # Tensors whose reload a direct XOR reproduced exactly on every peer rank,
        # and those this rank saw since peers last agreed.
        self._verified: set[str] = set()
        self._pending: set[str] = set()
        # Reduces flags to those every tensor-parallel peer set; ranks must agree
        # because attention processing may run collectives.
        self._agree = agree
        self._experts = routed_expert_layers(model)
        self._expert_ids = {id(module) for module in self._experts.values()}
        # Layers this rank does not serve: other pipeline stages' placeholders,
        # and checkpoint layers the model never builds, such as unused MTP layers.
        self._served_layers = {
            key
            for path, module in model.named_modules()
            if type(module).__name__ != "PPMissingLayer" and (key := layer_key(path)) is not None
        }
        # Processed zeros per expert layout; None when processing maps zero to zero.
        self._zero_terms: dict[tuple, dict[str, torch.Tensor] | None] = {}
        self._catalog: tuple[dict[str, dict], CheckpointCatalog] | None = None

    def install(self, checkpoint: str | Path, staging: Path) -> dict[str, float]:
        started = time.perf_counter()
        staged = StagedDelta(staging)
        try:
            with _quiet_readonly_buffers():
                metrics = self._install(checkpoint, staged)
        finally:
            staged.close()
        _synchronize(self._device)
        metrics["perf/mx_receive_inplace_time"] = time.perf_counter() - started
        return metrics

    def _install(self, checkpoint: str | Path, staged: StagedDelta) -> dict[str, float]:
        index_started = time.perf_counter()
        _paths, _locations, metadata = index_checkpoint_tensors(checkpoint)
        catalog = self._catalog_for(metadata)
        names = staged.names()
        if any(name not in metadata for name in names):
            raise InPlaceUnsupported("staged tensors are missing from the checkpoint")
        changed = [name for name in names if self._served(catalog.layers[name])]
        unmatched = [name for name in changed if name in catalog.unmatched_experts]
        if unmatched:
            raise InPlaceUnsupported(f"unrecognized expert tensors {unmatched[:3]}")
        expert_layers = sorted(
            {catalog.layers[name] for name in changed if name in catalog.per_expert}
        )
        missing = [index for index in expert_layers if index not in self._experts]
        if missing:
            raise InPlaceUnsupported(f"no routed-expert module for layers {missing[:5]}")
        index_seconds = time.perf_counter() - index_started
        plan_started = time.perf_counter()
        plan = self._plan(catalog, metadata, changed)
        if self._direct_enabled:
            self._adopt_verified(catalog)
        plan_seconds = time.perf_counter() - plan_started

        expert_started = time.perf_counter()
        for index in expert_layers:
            self._install_experts(
                self._experts[index], staged, catalog.experts.get(index, {}), metadata
            )
        _synchronize(self._device)
        expert_seconds = time.perf_counter() - expert_started

        generic_started = time.perf_counter()
        fed, direct = self._install_generic(staged, changed, catalog, metadata, plan)
        _synchronize(self._device)
        return {
            "perf/mx_receive_inplace_index_time": index_seconds,
            "perf/mx_receive_inplace_plan_time": plan_seconds,
            "perf/mx_receive_inplace_expert_layers": float(len(expert_layers)),
            "perf/mx_receive_inplace_expert_time": expert_seconds,
            "perf/mx_receive_inplace_generic_tensors": float(fed),
            "perf/mx_receive_inplace_direct_tensors": float(direct),
            "perf/mx_receive_inplace_verified_tensors": float(len(self._verified)),
            "perf/mx_receive_inplace_generic_time": time.perf_counter() - generic_started,
        }

    def _catalog_for(self, metadata: dict[str, dict]) -> CheckpointCatalog:
        # The checkpoint index returns the same metadata while headers are unchanged.
        if self._catalog is None or self._catalog[0] is not metadata:
            self._catalog = (metadata, CheckpointCatalog.of(metadata))
        return self._catalog[1]

    def _served(self, key: int | None) -> bool:
        return key is None or key in self._served_layers

    def _install_experts(
        self,
        layer: torch.nn.Module,
        staged: StagedDelta,
        expert_names: dict[tuple[int, str, str], str],
        metadata: dict[str, dict],
    ) -> None:
        arranged = arrange_expert_delta(layer, staged, expert_names, metadata, device=self._device)
        signature = (
            type(layer.quant_method).__name__,
            tuple((name, tuple(tensor.shape), tensor.dtype) for name, tensor in sorted(arranged.items())),
        )
        if signature not in self._zero_terms:
            zeros = process_like_weights(
                layer, {name: torch.zeros_like(tensor) for name, tensor in arranged.items()}
            )
            nonzero = any(bool(bits(tensor).any()) for tensor in zeros.values())
            self._zero_terms[signature] = zeros if nonzero else None
        processed = process_like_weights(layer, arranged)
        zero = self._zero_terms[signature]
        for name, tensor in processed.items():
            live = getattr(layer, name).data
            xor_into(live, tensor)
            if zero is not None:
                xor_into(live, zero[name])

    def _install_generic(
        self,
        staged: StagedDelta,
        changed: list[str],
        catalog: CheckpointCatalog,
        metadata: dict[str, dict],
        plan: dict[str, tuple[DirectCopy, ...]],
    ) -> tuple[int, int]:
        """Install every changed non-expert tensor; return reloaded and direct counts.

        A group whose changed tensors were all verified takes a direct XOR, and
        every other group a reload that may verify its tensors for next time.
        """
        by_group: dict[int | None, list[str]] = defaultdict(list)
        for name in changed:
            if name not in catalog.per_expert:
                by_group[catalog.layers[name]].append(name)
        if not by_group:
            return 0, 0
        direct_groups = {
            group
            for group, names in by_group.items()
            if all(name in self._verified for name in names)
        }
        direct = [name for group in direct_groups for name in by_group[group]]
        if direct:
            with torch.no_grad():
                self._xor_direct(staged, direct, plan, metadata)
                attention = [
                    module
                    for layers in self._attention_by_group(direct_groups).values()
                    for _path, module in layers
                ]
                if attention:
                    assert self._refresh_attention is not None
                    self._refresh_attention(attention)
        reload_groups = set(by_group) - direct_groups
        if not reload_groups:
            return 0, len(direct)
        reloaded = [name for group in reload_groups for name in by_group[group]]
        fed = self._reload_groups(staged, reloaded, reload_groups, catalog, metadata, plan)
        return fed, len(direct)

    def _reload_groups(
        self,
        staged: StagedDelta,
        changed: list[str],
        groups: set[int | None],
        catalog: CheckpointCatalog,
        metadata: dict[str, dict],
        plan: dict[str, tuple[DirectCopy, ...]],
    ) -> int:
        """Reload every tensor of each group with XOR restore."""
        fed = [
            name
            for name in catalog.generic
            if catalog.layers[name] in groups and self._served(catalog.layers[name])
        ]
        unknown = sorted({metadata[name]["dtype"] for name in fed} - _TORCH_DTYPES.keys())
        if unknown:
            raise InPlaceUnsupported(f"unsupported checkpoint dtypes {unknown}")
        staged_names = set(changed)
        live = self._live_tensors(groups)
        # Some loaders write live storage directly: tensors reload leaves in place,
        # such as biases, and mirrors a loader keeps, such as fp32 decode copies.
        # They then hold the loaded value rather than an XOR of it.
        originals = {name: tensor.detach().clone() for name, tensor in live.items()}

        def tensors(zero: bool) -> Iterator[tuple[str, torch.Tensor]]:
            for name in fed:
                info = metadata[name]
                delta = None if zero or name not in staged_names else staged.tensor(name)
                data = staged_bytes(info["byte_size"], delta, self._input_device(info["byte_size"]))
                yield name, data.view(_TORCH_DTYPES[info["dtype"]]).view(info["shape"])

        self._generic_reload(lambda _aliases: self._model.load_weights(tensors(zero=True)))
        # XOR the original back into every tensor the zero pass changed, after
        # both passes. A tensor a loader overwrote then ends as original ^ delta;
        # for one the XOR restore changed, the two extra XORs cancel.
        changed_live = {}
        for name, original in originals.items():
            if not same_bits(live[name].data, original):
                xor_into(live[name].data, original)
                changed_live[name] = original
        self._generic_reload(lambda _aliases: self._model.load_weights(tensors(zero=False)))
        for name, original in changed_live.items():
            xor_into(live[name].data, original)
        if self._direct_enabled:
            with torch.no_grad():
                self._verify(staged, changed, groups, catalog, metadata, plan, live, originals)
        return len(fed)

    def _verify(
        self,
        staged: StagedDelta,
        changed: list[str],
        groups: set[int | None],
        catalog: CheckpointCatalog,
        metadata: dict[str, dict],
        plan: dict[str, tuple[DirectCopy, ...]],
        live: dict[str, torch.Tensor],
        originals: dict[str, torch.Tensor],
    ) -> None:
        """Record a group's changed tensors when a direct XOR reproduces its reload.

        Every live tensor of the group must have changed exactly as the direct
        XOR predicts, and the delta must reach every region it writes. A mirror
        that a loader keeps of a changed tensor therefore fails the group, while
        a tensor a loader rewrites with its unchanged value does not. Attention
        layers derive their tensors in processing instead, which rerunning must
        leave unchanged.
        """
        writes: dict[str, list[tuple[str, DirectCopy]]] = defaultdict(list)
        for name in catalog.generic:
            if catalog.layers[name] in groups:
                for copy in plan.get(name, ()):
                    writes[copy.param].append((name, copy))
        attention = self._attention_by_group(groups)
        derived = {
            f"{path}.{name}"
            for layers in attention.values()
            for path, module in layers
            for name in {**module._parameters, **module._buffers}
        }
        staged_names = set(changed)
        # Check each live tensor once, under the name loaders copy into.
        canonical: dict[int, str] = {}
        for name, tensor in live.items():
            if id(tensor) not in canonical or name in writes:
                canonical[id(tensor)] = name
        tensor_groups: dict[int | None, list[str]] = defaultdict(list)
        for name in canonical.values():
            tensor_groups[layer_key(name.rsplit(".", 1)[0])].append(name)
        reached = {name: True for name in changed}
        consistent: dict[int | None, bool] = {}
        for group in groups:
            names = [name for name in changed if catalog.layers[name] == group]
            consistent[group] = bool(names) and all(
                name in plan and all(copy.param in live for copy in plan[name]) for name in names
            )
            if not consistent[group]:
                continue
            for name in tensor_groups.get(group, ()):
                if name in derived:
                    continue
                tensor = live[name].data
                copies = writes.get(name, [])
                predicted = torch.zeros_like(tensor)
                for source, copy in copies:
                    if source in staged_names:
                        info = metadata[source]
                        xor_direct(
                            predicted,
                            staged.tensor(source),
                            copy,
                            shape=info["shape"],
                            dtype=_TORCH_DTYPES[info["dtype"]],
                        )
                if not torch.equal(bits(predicted), bits(tensor) ^ bits(originals[name])):
                    consistent[group] = False
                    break
                for source, copy in copies:
                    if source in staged_names and not bool(direct_region(predicted, copy).any()):
                        reached[source] = False
        # Same calls on every rank, whatever its data showed.
        for group in sorted(attention, key=lambda key: -1 if key is None else key):
            identity = self._refresh_is_identity([module for _path, module in attention[group]])
            consistent[group] = consistent.get(group, False) and identity
        self._pending.update(
            name for name in changed if consistent.get(catalog.layers[name]) and reached[name]
        )

    def _adopt_verified(self, catalog: CheckpointCatalog) -> None:
        """Keep the verified tensors that every tensor-parallel peer verified."""
        candidates = self._verified | self._pending
        self._pending = set()
        if self._agree is None:
            self._verified = candidates
            return
        names = [name for name in catalog.generic if self._served(catalog.layers[name])]
        flags = torch.tensor([name in candidates for name in names], dtype=torch.int32)
        agreed = self._agree(flags).tolist()
        self._verified = {name for name, flag in zip(names, agreed) if flag}

    @property
    def _direct_enabled(self) -> bool:
        return (
            rl_envs.MX_REFIT_INPLACE_DIRECT
            and self._capture_layout is not None
            and self._refresh_attention is not None
        )

    def _refresh_is_identity(self, layers: list[torch.nn.Module]) -> bool:
        """Whether rerunning attention processing after the reload changes nothing.

        Processing derives the layers' registered tensors. Other tensor
        attributes, such as a bound KV cache, are runtime state it does not
        touch, and copying them could exhaust GPU memory.
        """
        assert self._refresh_attention is not None

        def tensors() -> dict[tuple[int, str], torch.Tensor]:
            found = {}
            for index, layer in enumerate(layers):
                values = {**layer._parameters, **layer._buffers}
                found.update(((index, name), value) for name, value in values.items() if value is not None)
            return found

        before = {key: value.detach().clone() for key, value in tensors().items()}
        self._refresh_attention(layers)
        after = tensors()
        if after.keys() == before.keys() and all(same_bits(after[key], before[key]) for key in before):
            return True
        for key, value in before.items():
            if key in after and after[key].shape == value.shape:
                after[key].data.copy_(value)
        return False

    def _plan(
        self, catalog: CheckpointCatalog, metadata: dict[str, dict], changed: list[str]
    ) -> dict[str, tuple[DirectCopy, ...]]:
        """Where vLLM copies each changed non-expert tensor, captured once per tensor.

        Capturing only tensors that change keeps loaders of frozen parts, such as
        a vision tower, out of the capture.
        """
        if not self._direct_enabled:
            return {}
        if self._direct_plan is None or self._direct_plan[0] is not metadata:
            self._direct_plan = (metadata, {}, set())
            self._verified = set()
            self._pending = set()
        _metadata, plan, attempted = self._direct_plan
        wanted = [
            name
            for name in changed
            if name not in catalog.per_expert
            and name not in attempted
            and metadata[name]["dtype"] in _TORCH_DTYPES
        ]
        if not wanted:
            return plan
        attempted.update(wanted)
        manifest = [
            (name, _TORCH_DTYPES[metadata[name]["dtype"]], tuple(metadata[name]["shape"]))
            for name in wanted
        ]
        excluded = {
            f"{path}.{name}"
            for path, module in self._model.named_modules()
            if self._is_expert_module(path, module) or _is_deferred_attention(module)
            for name in module._parameters
        }
        assert self._capture_layout is not None
        try:
            capture, layout = self._capture_layout(manifest)
            captured = direct_copies(
                capture, layout, metadata, dict(self._model.named_parameters()), excluded
            )
        except Exception as error:
            # The reload path still installs every tensor.
            logger.warning("Direct in-place capture failed: %s", error)
            captured = {}
        plan.update(captured)
        logger.info(
            "Direct in-place capture placed %d of %d tensors", len(captured), len(manifest)
        )
        return plan

    def _xor_direct(
        self,
        staged: StagedDelta,
        names: list[str],
        plan: dict[str, tuple[DirectCopy, ...]],
        metadata: dict[str, dict],
    ) -> None:
        params = dict(self._model.named_parameters())
        for name in names:
            info = metadata[name]
            tensor = staged.tensor(name)
            for copy in plan[name]:
                target = params.get(copy.param)
                if target is None:
                    raise InPlaceUnsupported(f"live parameter {copy.param} disappeared")
                xor_direct(
                    target.data,
                    tensor,
                    copy,
                    shape=info["shape"],
                    dtype=_TORCH_DTYPES[info["dtype"]],
                )

    def _attention_by_group(
        self, groups: set[int | None]
    ) -> dict[int | None, list[tuple[str, torch.nn.Module]]]:
        """Deferred attention layers of the groups, whose processing reads their weights."""
        found: dict[int | None, list[tuple[str, torch.nn.Module]]] = defaultdict(list)
        for path, module in self._model.named_modules():
            key = layer_key(path)
            if key in groups and _is_deferred_attention(module):
                found[key].append((path, module))
        return found

    def _input_device(self, nbytes: int) -> torch.device:
        """Build a reload input on the GPU when it has room to spare, else on the host.

        Host inputs cost page faults and a host-to-device copy of every byte.
        """
        if self._device.type != "cuda":
            return torch.device("cpu")
        free, _total = torch.cuda.mem_get_info(self._device)
        cached = torch.cuda.memory_reserved(self._device) - torch.cuda.memory_allocated(
            self._device
        )
        return self._device if free + cached > 2 * nbytes + (1 << 30) else torch.device("cpu")

    def _is_expert_module(self, path: str, module: torch.nn.Module) -> bool:
        return id(module) in self._expert_ids or "experts" in path.split(".")

    def _live_tensors(self, groups: set[int | None]) -> dict[str, torch.Tensor]:
        """Parameters and buffers of the changed groups, except routed experts."""
        found: dict[str, torch.Tensor] = {}
        for path, module in self._model.named_modules():
            if layer_key(path) not in groups or self._is_expert_module(path, module):
                continue
            for name, tensor in list(module.named_parameters(recurse=False)) + list(
                module.named_buffers(recurse=False)
            ):
                if tensor.device.type != "meta":
                    found.setdefault(f"{path}.{name}", tensor)
        return found


@contextmanager
def xor_restore() -> Iterator[None]:
    """Make layerwise reload XOR processed tensors into live storage instead of copying."""
    from vllm.model_executor.model_loader.reload import layerwise

    def restore(layer: torch.nn.Module, info: Any) -> None:
        parameters, buffers = info.kernel_tensors
        non_persistent = info.kernel_non_persistent_buffers
        loaded = {name for name, _bound in info.loaded_weights}
        for name, parameter in parameters.items():
            current = getattr(layer, name)
            # Tensors reload never replaced are live storage, not processed output.
            if current is not parameter:
                xor_into(parameter.data, current.data)
        for name, buffer in buffers.items():
            if name not in layer._buffers:
                continue
            if name in non_persistent and name not in loaded:
                continue
            current = getattr(layer, name)
            if current is not buffer:
                xor_into(buffer.data, current.data)
        layerwise._place_kernel_tensors(layer, info)

    original = layerwise._copy_and_restore_kernel_tensors
    layerwise._copy_and_restore_kernel_tensors = restore
    try:
        yield
    finally:
        layerwise._copy_and_restore_kernel_tensors = original


__all__ = [
    "CheckpointCatalog",
    "DirectCopy",
    "InPlaceInstaller",
    "InPlaceUnsupported",
    "arrange_expert_delta",
    "bits",
    "direct_copies",
    "direct_region",
    "process_like_weights",
    "raw_bytes",
    "routed_expert_layers",
    "scatter_expert_shards",
    "shard_nonzero",
    "staged_bytes",
    "xor_direct",
    "xor_into",
    "xor_restore",
]
