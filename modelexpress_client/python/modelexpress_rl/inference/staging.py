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
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_FILE = "index.json"
BASE_FILE = "base.json"
POSITIONS_FILE = "positions.bin"
VALUES_FILE = "values.bin"
_UINT32_LIMIT = 1 << 32


def position_dtype(nbytes: int) -> np.dtype:
    """Narrowest offset type for a tensor of ``nbytes`` bytes."""
    return np.dtype(np.uint32 if nbytes <= _UINT32_LIMIT else np.uint64)


def nonzero_bytes(block: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute offsets and values of the nonzero bytes in one block."""
    positions = np.flatnonzero(block)
    return positions + offset, block[positions].copy()


def join_nonzero(
    nbytes: int, parts: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate block results in order, with the narrowest offset type."""
    dtype = position_dtype(nbytes)
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
    names = sorted(tensors)
    sizes, counts = [], []
    with (directory / POSITIONS_FILE).open("wb") as positions_file, (
        directory / VALUES_FILE
    ).open("wb") as values_file:
        for name in names:
            nbytes, positions, values = tensors[name]
            if positions.size != values.size:
                raise ValueError(f"staged delta {name!r} has mismatched lengths")
            if positions.dtype != position_dtype(nbytes):
                raise ValueError(f"staged delta {name!r} has {positions.dtype} offsets")
            positions_file.write(positions.tobytes())
            values_file.write(values.tobytes())
            sizes.append(nbytes)
            counts.append(int(values.size))
    # Parallel lists parse much faster than one record per tensor, and offsets
    # follow from the counts. The base is separate so checking it reads no index.
    (directory / INDEX_FILE).write_text(
        json.dumps({"names": names, "nbytes": sizes, "counts": counts})
    )
    (directory / BASE_FILE).write_text(json.dumps({"base_version": base_version}))


@dataclass(frozen=True)
class StagedTensor:
    nbytes: int
    positions: np.ndarray
    values: np.ndarray


class StagedDelta:
    """Read-only view of a staged delta.

    Tensors are rows in name order, so one layer's tensors are consecutive rows
    and their bytes are one range of each file.
    """

    def __init__(self, directory: Path) -> None:
        index = json.loads((directory / INDEX_FILE).read_text())
        self.base_version: str = json.loads((directory / BASE_FILE).read_text())[
            "base_version"
        ]
        self._names: list[str] = index["names"]
        self._rows = {name: row for row, name in enumerate(self._names)}
        self._nbytes = np.asarray(index["nbytes"], dtype=np.int64)
        self._counts = np.asarray(index["counts"], dtype=np.int64)
        self._widths = np.where(self._nbytes <= _UINT32_LIMIT, 4, 8)
        self._position_offsets = _starts(self._counts * self._widths)
        self._value_offsets = _starts(self._counts)
        sizes = {
            POSITIONS_FILE: int((self._counts * self._widths).sum()),
            VALUES_FILE: int(self._counts.sum()),
        }
        for name, size in sizes.items():
            if (directory / name).stat().st_size != size:
                raise ValueError(f"staged {name} does not match its index")
        self._maps = []
        self._positions = self._map(directory / POSITIONS_FILE)
        self._values = self._map(directory / VALUES_FILE)
        self._files = {
            name: os.open(directory / name, os.O_RDONLY) for name in (POSITIONS_FILE, VALUES_FILE)
        }

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
            return json.loads((directory / BASE_FILE).read_text())["base_version"]
        except (OSError, ValueError, KeyError):
            return None

    def names(self) -> list[str]:
        return list(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._rows

    def tensor(self, name: str) -> StagedTensor:
        row = self._rows[name]
        nbytes = int(self._nbytes[row])
        count = int(self._counts[row])
        positions = np.frombuffer(
            self._positions,
            dtype=position_dtype(nbytes),
            count=count,
            offset=int(self._position_offsets[row]),
        )
        values = np.frombuffer(
            self._values, dtype=np.uint8, count=count, offset=int(self._value_offsets[row])
        )
        return StagedTensor(nbytes, positions, values)

    def row(self, name: str) -> int:
        return self._rows[name]

    def counts(self, begin: int, end: int) -> np.ndarray:
        """Staged byte counts of rows ``[begin, end)``."""
        return self._counts[begin:end]

    def offset_width(self, row: int) -> int:
        """Bytes per staged offset of one row."""
        return int(self._widths[row])

    def byte_ranges(self, begin: int, end: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Ranges of rows ``[begin, end)`` in the offsets file and the values file."""
        last = end - 1
        positions_end = self._position_offsets[last] + self._counts[last] * self._widths[last]
        values_end = self._value_offsets[last] + self._counts[last]
        return (
            (int(self._position_offsets[begin]), int(positions_end)),
            (int(self._value_offsets[begin]), int(values_end)),
        )

    def read_into(self, file: str, offset: int, out: memoryview) -> None:
        """Read ``len(out)`` bytes of ``file``, POSITIONS_FILE or VALUES_FILE, at ``offset``."""
        done = 0
        while done < len(out):
            read = os.preadv(self._files[file], [out[done:]], offset + done)
            if read <= 0:
                raise OSError(f"staged {file} ended at byte {offset + done}")
            done += read

    def close(self) -> None:
        """Unmap the files once no array still views them."""
        for descriptor in self._files.values():
            os.close(descriptor)
        self._files.clear()
        try:
            self._positions.release()
            self._values.release()
            for mapped in self._maps:
                mapped.close()
        except BufferError:
            return
        self._maps.clear()


def _starts(sizes: np.ndarray) -> np.ndarray:
    """Offset of each item when items of ``sizes`` are laid out back to back."""
    return np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)


__all__ = [
    "StagedDelta",
    "StagedTensor",
    "join_nonzero",
    "nonzero_bytes",
    "position_dtype",
    "write_staging",
]
