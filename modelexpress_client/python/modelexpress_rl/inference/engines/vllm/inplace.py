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
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modelexpress_rl.inference.staging import StagedDelta
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
    joined = np.concatenate(positions)
    if joined.dtype == np.uint32:
        offset = torch.from_numpy(joined.view(np.int32)).to(device).to(torch.int64) & 0xFFFFFFFF
    else:
        offset = torch.from_numpy(joined.view(np.int64)).to(device)
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
    kinds = [kind for kind in _ARRANGED_KINDS if f"w13_{kind}" in meta and f"w2_{kind}" in meta]
    changed = {kind for (_expert, _proj, kind), name in expert_names.items() if name in staged}
    if not kinds or changed - set(kinds):
        raise InPlaceUnsupported(f"unsupported expert tensors {sorted(changed - set(kinds))}")
    transposed = bool(getattr(meta[f"w13_{kinds[0]}"], "is_transposed", False))
    experts = meta[f"w13_{kinds[0]}"].shape[0]
    arranged: dict[str, torch.Tensor] = {}
    for kind in kinds:
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
        for name, tensor in ((f"w13_{kind}", w13), (f"w2_{kind}", parts["down"].contiguous())):
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
    bare = {name: value for name, value in vars(layer).items() if isinstance(value, torch.Tensor)}
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
        layer.quant_method.process_weights_after_loading(layer)
        return {name: getattr(layer, name).data for name in arranged}
    finally:
        _place_kernel_tensors(layer, info)
        for name, value in bare.items():
            setattr(layer, name, value)
        info.reset()


class InPlaceInstaller:
    """Install staged deltas into one rank's live model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        generic_reload: Callable[[Callable[[Any], None]], None],
    ) -> None:
        self._model = model
        self._device = device
        self._generic_reload = generic_reload
        self._experts = routed_expert_layers(model)
        # Pipeline-parallel ranks hold placeholders for other stages' layers.
        self._other_stage_layers = {
            key
            for path, module in model.named_modules()
            if type(module).__name__ == "PPMissingLayer" and (key := layer_key(path)) is not None
        }
        # Processed zeros per expert layout; None when processing maps zero to zero.
        self._zero_terms: dict[tuple, dict[str, torch.Tensor] | None] = {}

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
        _paths, _locations, metadata = index_checkpoint_tensors(checkpoint)
        changed = [name for name in staged.names() if name in metadata]
        if len(changed) != len(staged.names()):
            raise InPlaceUnsupported("staged tensors are missing from the checkpoint")
        expert_names: dict[int, dict[tuple[int, str, str], str]] = defaultdict(dict)
        for name in metadata:
            match = _EXPERT.search(name)
            if match:
                key = (int(match[2]), _PROJECTIONS[match[3]], match[4])
                expert_names[int(match[1])][key] = name
        expert_layers = sorted(
            index
            for index in {layer_key(name) for name in changed if is_per_expert(name)}
            if index not in self._other_stage_layers
        )
        missing = [index for index in expert_layers if index not in self._experts]
        if missing:
            raise InPlaceUnsupported(f"no routed-expert module for layers {missing[:5]}")

        expert_started = time.perf_counter()
        for index in expert_layers:
            self._install_experts(self._experts[index], staged, expert_names[index], metadata)
        _synchronize(self._device)
        expert_seconds = time.perf_counter() - expert_started

        generic_started = time.perf_counter()
        fed = self._install_generic(staged, changed, metadata)
        _synchronize(self._device)
        return {
            "perf/mx_receive_inplace_expert_layers": float(len(expert_layers)),
            "perf/mx_receive_inplace_expert_time": expert_seconds,
            "perf/mx_receive_inplace_generic_tensors": float(fed),
            "perf/mx_receive_inplace_generic_time": time.perf_counter() - generic_started,
        }

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
        self, staged: StagedDelta, changed: list[str], metadata: dict[str, dict]
    ) -> int:
        """Reload every tensor of each changed non-expert group with XOR restore."""
        groups = {layer_key(name) for name in changed if not is_per_expert(name)}
        if not groups:
            return 0
        fed = [name for name in metadata if not is_per_expert(name) and layer_key(name) in groups]
        staged_names = set(changed)
        # Some loaders write live storage directly: tensors reload leaves in place,
        # such as biases, and mirrors a loader keeps, such as fp32 decode copies.
        # They then hold the loaded value rather than an XOR of it.
        originals = {
            name: tensor.detach().clone()
            for name, tensor in self._live_tensors(groups).items()
        }

        def tensors(zero: bool) -> Iterator[tuple[str, torch.Tensor]]:
            for name in fed:
                info = metadata[name]
                data = np.zeros(info["byte_size"], dtype=np.uint8)
                if not zero and name in staged_names:
                    staged_tensor = staged.tensor(name)
                    data[staged_tensor.positions.astype(np.int64)] = staged_tensor.values
                tensor = torch.from_numpy(data)
                dtype = _TORCH_DTYPES[info["dtype"]]
                yield name, tensor.view(dtype).view(info["shape"])

        live = self._live_tensors(groups)
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
        return len(fed)

    def _live_tensors(self, groups: set[int | None]) -> dict[str, torch.Tensor]:
        """Parameters and buffers of the changed groups, except routed experts."""
        experts = {id(module) for module in self._experts.values()}
        found: dict[str, torch.Tensor] = {}
        for path, module in self._model.named_modules():
            if id(module) in experts or layer_key(path) not in groups:
                continue
            if any(part == "experts" for part in path.split(".")):
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
    "InPlaceInstaller",
    "InPlaceUnsupported",
    "arrange_expert_delta",
    "bits",
    "process_like_weights",
    "raw_bytes",
    "routed_expert_layers",
    "scatter_expert_shards",
    "shard_nonzero",
    "xor_into",
    "xor_restore",
]
