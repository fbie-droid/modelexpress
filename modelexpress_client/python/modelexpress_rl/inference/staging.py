# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nonzero XOR bytes of one delta, staged for in-place installation.

A staged delta lists, per changed tensor, the byte offsets where the XOR delta is
nonzero and the XOR values there. Engine ranks memory-map it during the pause, so
only changed bytes leave host memory.
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_FILE = "index.json"
POSITIONS_FILE = "positions.bin"
VALUES_FILE = "values.bin"
_UINT32_LIMIT = 1 << 32


def nonzero_bytes(block: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute offsets and values of the nonzero bytes in one block."""
    positions = np.flatnonzero(block)
    return positions + offset, block[positions].copy()


def join_nonzero(
    nbytes: int, parts: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate block results in order, with the narrowest offset type."""
    dtype = np.uint32 if nbytes <= _UINT32_LIMIT else np.uint64
    if not parts:
        return np.empty(0, dtype=dtype), np.empty(0, dtype=np.uint8)
    positions = np.concatenate([part[0] for part in parts]).astype(dtype, copy=False)
    values = np.concatenate([part[1] for part in parts])
    return positions, values


def write_staging(
    directory: Path,
    *,
    base_version: str,
    tensors: dict[str, tuple[int, np.ndarray, np.ndarray]],
) -> None:
    """Write tensors as {name: (tensor bytes, offsets, values)} into ``directory``."""
    entries = {}
    position_offset = value_offset = 0
    with (directory / POSITIONS_FILE).open("wb") as positions_file, (
        directory / VALUES_FILE
    ).open("wb") as values_file:
        for name in sorted(tensors):
            nbytes, positions, values = tensors[name]
            if positions.size != values.size:
                raise ValueError(f"staged delta {name!r} has mismatched lengths")
            positions_file.write(positions.tobytes())
            values_file.write(values.tobytes())
            entries[name] = {
                "nbytes": nbytes,
                "count": int(values.size),
                "position_dtype": positions.dtype.name,
                "position_offset": position_offset,
                "value_offset": value_offset,
            }
            position_offset += positions.nbytes
            value_offset += values.nbytes
    (directory / INDEX_FILE).write_text(
        json.dumps({"base_version": base_version, "tensors": entries}, sort_keys=True)
    )


@dataclass(frozen=True)
class StagedTensor:
    nbytes: int
    positions: np.ndarray
    values: np.ndarray


class StagedDelta:
    """Read-only view of a staged delta."""

    def __init__(self, directory: Path) -> None:
        index = json.loads((directory / INDEX_FILE).read_text())
        self.base_version: str = index["base_version"]
        self._entries: dict[str, dict] = index["tensors"]
        self._maps = []
        self._positions = self._map(directory / POSITIONS_FILE)
        self._values = self._map(directory / VALUES_FILE)

    def _map(self, path: Path) -> memoryview:
        if path.stat().st_size == 0:
            return memoryview(b"")
        with path.open("rb") as handle:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._maps.append(mapped)
        return memoryview(mapped)

    @staticmethod
    def base_of(directory: Path) -> str | None:
        """Return the base version a staged delta applies to, if it exists."""
        try:
            return json.loads((directory / INDEX_FILE).read_text())["base_version"]
        except (OSError, ValueError, KeyError):
            return None

    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def tensor(self, name: str) -> StagedTensor:
        entry = self._entries[name]
        dtype = np.dtype(entry["position_dtype"])
        positions = np.frombuffer(
            self._positions, dtype=dtype, count=entry["count"], offset=entry["position_offset"]
        )
        values = np.frombuffer(
            self._values, dtype=np.uint8, count=entry["count"], offset=entry["value_offset"]
        )
        return StagedTensor(entry["nbytes"], positions, values)

    def close(self) -> None:
        """Unmap the files once no array still views them."""
        try:
            self._positions.release()
            self._values.release()
            for mapped in self._maps:
                mapped.close()
        except BufferError:
            return
        self._maps.clear()


__all__ = [
    "StagedDelta",
    "StagedTensor",
    "join_nonzero",
    "nonzero_bytes",
    "write_staging",
]
