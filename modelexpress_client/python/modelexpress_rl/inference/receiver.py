# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical S3 checkpoint preparation for generator refit."""

from __future__ import annotations

import hashlib
import itertools
import logging
import json
import os
import shutil
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointCacheCapacityError,
    CheckpointRecord,
    CheckpointState,
    LocalCheckpointStore,
    checkpoint_files_state,
)
from modelexpress_rl.inference.staging import (
    StagedDelta,
    StagedTensor,
    join_nonzero,
    nonzero_bytes,
    write_staging,
)
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.utils import (
    Adler32Checksum,
    checksum_factory,
    index_checkpoint_tensors,
    read_safetensors_header,
    threadpool_map,
)

if TYPE_CHECKING:
    from .streaming_checkpoint import StreamedCheckpoint

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1_000_000_000
DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB = 2000


@dataclass(frozen=True)
class ObjectStorageGeneratorConfig:
    """Object-storage checkpoint settings for one generator rank."""

    storage_type: ObjectStorageType
    initial_base_version_id: str
    seed_checkpoint_path: str | Path | None
    refit_checkpoint_dir: str | Path
    refit_checkpoint_max_size_gb: int | None = DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_type, ObjectStorageType):
            raise TypeError("storage_type must be an ObjectStorageType")
        if not self.initial_base_version_id.strip():
            raise ValueError("initial_base_version_id is required")
        if (
            self.seed_checkpoint_path is not None
            and not str(self.seed_checkpoint_path).strip()
        ):
            raise ValueError("seed_checkpoint_path must be non-empty when provided")
        if not str(self.refit_checkpoint_dir).strip():
            raise ValueError("refit_checkpoint_dir is required")
        if (
            self.refit_checkpoint_max_size_gb is not None
            and self.refit_checkpoint_max_size_gb <= 0
        ):
            raise ValueError("refit_checkpoint_max_size_gb must be positive")


class ReceiverInstallError(RuntimeError):
    """An engine reload failed."""


@dataclass(frozen=True)
class DeltaChange:
    """Checkpoint tensor names one delta changes from its exact base."""

    base_version: str
    version: str
    tensor_names: frozenset[str]


@dataclass(frozen=True)
class PreparedCheckpoint:
    """One verified host-local checkpoint ready for engine installation."""

    target_version: str
    path: Path
    metrics: dict[str, float]
    streaming: StreamedCheckpoint | None = None
    # Deltas from the target's full root, oldest first; empty when unknown.
    delta_changes: tuple[DeltaChange, ...] = ()
    # Nonzero XOR bytes from ``staging_base`` to the target, when staged.
    staging: Path | None = None
    staging_base: str | None = None
    # When ``path`` holds an earlier version, ``overlay_base``, the staged deltas
    # from it to the target, oldest first: the target's bytes are its files XOR
    # all of them. Empty when ``path`` holds the target.
    overlay: tuple[Path, ...] = ()
    overlay_base: str | None = None

    @property
    def materialized(self) -> bool:
        """Whether ``path`` holds the target's own bytes."""
        return not self.overlay


@dataclass(frozen=True)
class _S3Version:
    version_id: str
    base_version_id: str | None
    payload_format: WeightPayloadFormat
    uri: str


_S3Manifest = tuple[_S3Version, bytes, dict[str, Any], dict[str, str]]


def _source_identity(version: _S3Version) -> dict[str, str]:
    return {"uri": version.uri}


def speculative_version_id(uri: str) -> str:
    """Return the cache ID of a delta prepared before its version is registered."""
    return "mx-speculative-" + hashlib.sha256(uri.encode()).hexdigest()[:24]


_Decompressor = Callable[[memoryview], Any]


# Delta bytes applied per read and write of the checkpoint file. Tensors larger
# than _SPREAD_TENSOR_BYTES spread their chunks over the apply workers.
_APPLY_CHUNK_BYTES = 16 << 20
_SPREAD_TENSOR_BYTES = 256 << 20


def _read_exactly(fd: int, out: memoryview, offset: int) -> None:
    done = 0
    while done < len(out):
        read = os.preadv(fd, [out[done:]], offset + done)
        if read <= 0:
            raise ValueError(f"checkpoint file ended at byte {offset + done}")
        done += read


def _write_all(fd: int, data: memoryview, offset: int) -> None:
    done = 0
    while done < len(data):
        done += os.pwritev(fd, [data[done:]], offset + done)


def _read_block(reader: Any, size: int) -> bytes:
    """Read ``size`` bytes from a decompressor, or fewer only at its end."""
    block = reader.read(size)
    if len(block) in (0, size):
        return block
    parts = [block]
    remaining = size - len(block)
    while remaining:
        more = reader.read(remaining)
        if not more:
            break
        parts.append(more)
        remaining -= len(more)
    return b"".join(parts)


@dataclass(frozen=True)
class _TensorPass:
    """One tensor's XORs in a pass over the checkpoint files.

    ``write`` is a compressed delta XORed into the file's bytes and written
    back; ``scatter`` holds staged deltas' nonzero bytes, used the same way.
    ``overlay`` holds staged deltas XORed in after that without writing, and
    ``check`` is XORed last, also without writing. With ``checksum``, the result
    must have that checksum. Staging records the nonzero bytes of ``check``, or
    of ``write`` when there is no check.
    """

    name: str
    write: memoryview | None = None
    check: memoryview | None = None
    checksum: str | None = None
    scatter: tuple[StagedTensor, ...] = ()
    overlay: tuple[StagedTensor, ...] = ()


_Shards = dict[str, tuple[bytes, list[str]]]


class _CopyAbandoned(Exception):
    """Ends a rewrite into a copy for a following delta, so the copy is discarded."""

    def __init__(
        self, manifest: _S3Manifest, downloaded: tuple[_Shards, float], index_seconds: float
    ) -> None:
        super().__init__(f"rewrite abandoned for {manifest[0].version_id!r}")
        self.manifest = manifest
        self.downloaded = downloaded
        self.index_seconds = index_seconds


def _xor_staged(target: np.ndarray, staged: StagedTensor, position: int) -> None:
    """XOR a staged delta's nonzero bytes into the block of its tensor at ``position``."""
    begin, end = np.searchsorted(staged.positions, (position, position + target.size))
    target[staged.positions[begin:end] - position] ^= staged.values[begin:end]


def _staged_tensors(layers: list[StagedDelta]) -> dict[str, tuple[StagedTensor, ...]]:
    """Each staged tensor's deltas, oldest first, from the layers that change it."""
    tensors: dict[str, tuple[StagedTensor, ...]] = {}
    for layer in layers:
        for name in layer.names():
            tensors[name] = (*tensors.get(name, ()), layer.tensor(name))
    return tensors


def _zstd_stream_reader(data: memoryview) -> Any:
    import zstandard

    return zstandard.ZstdDecompressor().stream_reader(data)


_DECOMPRESSORS: dict[str, _Decompressor] = {
    "zstd": _zstd_stream_reader,
}


def _is_safe_shard_basename(value: object) -> bool:
    """Return whether a shard is one file directly under its artifact root."""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
    )


def _parse_index_manifest(
    data: bytes,
    *,
    is_delta: bool,
    version: _S3Version | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse and validate an index manifest."""
    try:
        index = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ValueError("The index manifest is not valid JSON") from error

    weight_map = index.get("weight_map")
    if weight_map is None:
        raise ValueError("The index manifest has no weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("The index manifest has invalid weight_map")
    if not is_delta and not weight_map:
        raise ValueError("The index manifest has no tensors")
    for name, filename in weight_map.items():
        if (
            not isinstance(name, str)
            or not _is_safe_shard_basename(filename)
            or not filename.endswith(".safetensors")
        ):
            raise ValueError("The index manifest has invalid weight_map")

    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("The index manifest has invalid metadata")
    if is_delta:
        assert version is not None
        expected_metadata = {
            "delta_encoding": "xor",
            "checksum_format": "adler32",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"The index manifest {name} does not match revision "
                    f"{version.version_id!r}"
                )
        compression_format = metadata.get("compression_format")
        if compression_format not in _DECOMPRESSORS:
            raise ValueError(
                f"unsupported compression format {compression_format!r}"
            )
    elif "checksum_format" in metadata:
        checksum_factory(metadata["checksum_format"])
    return metadata, weight_map


def _group_tensors_by_shard(
    weight_map: dict[str, str],
) -> defaultdict[str, list[str]]:
    shard_to_tensors = defaultdict(list)
    for name, filename in weight_map.items():
        shard_to_tensors[filename].append(name)
    return shard_to_tensors


def _download_full_checkpoint(
    *,
    s3: S3Client,
    store: LocalCheckpointStore,
    target: Path,
    index_metadata: dict[str, Any],
    weight_map: dict[str, str],
    root_uri: str,
    protected_versions: set[str],
    tensor_metadata: dict[str, dict] | None = None,
) -> tuple[float, float]:
    """Download and validate one full HF checkpoint artifact."""
    checksum_format = index_metadata.get("checksum_format")
    if tensor_metadata is not None and set(weight_map) != set(tensor_metadata):
        raise ValueError("full HF checkpoint tensor set differs from local checkpoint")

    shard_to_tensors = _group_tensors_by_shard(weight_map)
    parent_uri = root_uri.rsplit("/", 1)[0]
    workers = min(rl_envs.MX_S3_DOWNLOAD_WORKERS, len(shard_to_tensors))
    shard_sizes = threadpool_map(
        shard_to_tensors,
        lambda filename: s3.size(f"{parent_uri}/{filename}"),
        max_workers=workers,
        thread_name_prefix="modelexpress-s3-head-full",
    )
    store.ensure_capacity(
        sum(shard_sizes),
        protected_versions=protected_versions,
        protected_paths={target},
    )

    def download_and_validate(filename: str) -> tuple[float, float]:
        download_started = time.perf_counter()
        try:
            data = s3.get(f"{parent_uri}/{filename}")
        except Exception as error:
            raise RuntimeError(
                f"full HF checkpoint download failed for {filename!r}"
            ) from error
        download_time = time.perf_counter() - download_started

        validation_started = time.perf_counter()
        header, data_start = read_safetensors_header(data, repr(filename))
        tensor_names = shard_to_tensors[filename]
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"full HF checkpoint shard {filename!r} has invalid metadata"
            )
        if set(header) - {"__metadata__"} != set(tensor_names):
            raise ValueError(
                f"full HF checkpoint shard {filename!r} tensor set differs "
                "from its index"
            )

        view = memoryview(data)
        for name in tensor_names:
            tensor_header = header[name]
            begin, end = tensor_header["data_offsets"]
            if tensor_metadata is not None:
                local_metadata = tensor_metadata[name]
                if (
                    tensor_header["dtype"] != local_metadata["dtype"]
                    or tensor_header["shape"] != local_metadata["shape"]
                    or end - begin != local_metadata["byte_size"]
                ):
                    raise ValueError(
                        f"full HF checkpoint metadata differs for {name!r}"
                    )
            source = np.frombuffer(
                view,
                dtype=np.uint8,
                count=end - begin,
                offset=data_start + begin,
            )
            if checksum_format is not None:
                expected_checksum = metadata.get(name)
                if expected_checksum is None:
                    raise ValueError(
                        f"full HF checkpoint shard {filename!r} is missing "
                        f"checksum for tensor {name!r}"
                    )
                checksum = checksum_factory(checksum_format)
                checksum.update(source)
                if checksum.hexdigest() != expected_checksum:
                    raise ValueError(
                        f"full HF checkpoint checksum differs for {name!r}"
                    )
        (target / filename).write_bytes(data)
        return download_time, time.perf_counter() - validation_started

    timings = list(
        threadpool_map(
            shard_to_tensors,
            download_and_validate,
            max_workers=workers,
            thread_name_prefix="modelexpress-s3-download-full",
        )
    )
    return (
        max(download for download, _ in timings),
        max(validation for _, validation in timings),
    )


def _active_version(store: LocalCheckpointStore) -> str | None:
    """Return the installed version when the store has committed one."""
    try:
        return store.active_version()
    except RuntimeError:
        return None


def _protected_versions(
    store: LocalCheckpointStore,
    *versions: str,
) -> set[str]:
    """Return versions excluded from cache eviction during this operation.

    Callers supply the operation's working set. The installed version is also
    retained so recovery can rebuild from the checkpoint backing the engine.
    """
    protected = set(versions)
    active = _active_version(store)
    if active is not None:
        protected.add(active)
    state = store.state()
    # A STAGED version's bytes are its base's files plus the staged deltas.
    if state is not None and state.base is not None and state.version in protected:
        protected.add(state.base)
        try:
            protected.update(_staged_layers(store, state))
        except RuntimeError:
            logger.warning("staged deltas of %s are not in its chain", state.version)
    return protected


def _staged_layers(store: LocalCheckpointStore, state: CheckpointRecord) -> tuple[str, ...]:
    """Versions staged on the head's checkpoint files, oldest first.

    These are the head's deltas after the version its files hold, ending with
    the head itself; a READY head has none.
    """
    if state.status is not CheckpointState.STAGED:
        return ()
    chain = store.chain(state.version)
    deltas = chain.get("deltas") if chain is not None else None
    if not isinstance(deltas, list) or not deltas or deltas[-1] != state.version:
        raise RuntimeError(f"checkpoint chain for {state.version!r} is missing")
    if state.base == chain.get("full_version"):
        return tuple(deltas)
    if state.base not in deltas:
        raise RuntimeError(f"staged base {state.base!r} is not in the chain of {state.version!r}")
    return tuple(deltas[deltas.index(state.base) + 1 :])


def _ensure_full_checkpoint(
    *,
    store: LocalCheckpointStore,
    target: Path,
    version: _S3Version,
    s3: S3Client,
    protected_versions: set[str],
) -> None:
    """Ensure one immutable, source-verified full checkpoint is cached."""
    source = _source_identity(version)
    if target.exists():
        try:
            store.verify_artifact_source(target, source)
            index_checkpoint_tensors(target)
            return
        except (OSError, RuntimeError, ValueError):
            pass

    index_data = s3.get(version.uri)
    metadata, weight_map = _parse_index_manifest(index_data, is_delta=False)
    store.ensure_capacity(
        len(index_data),
        protected_versions=protected_versions,
    )
    with store.replace_directory(target) as temporary:
        (temporary / Path(version.uri).name).write_bytes(index_data)
        _download_full_checkpoint(
            s3=s3,
            store=store,
            target=temporary,
            index_metadata=metadata,
            weight_map=weight_map,
            root_uri=version.uri,
            protected_versions=protected_versions,
        )
        store.record_artifact(temporary, source=source)
    index_checkpoint_tensors(target)


def bootstrap_s3_checkpoint(
    *,
    model_name: str,
    version: _S3Version,
    refit_checkpoint_dir: str | Path,
    s3: S3Client,
    refit_checkpoint_max_size_gb: int | None = DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB,
) -> Path:
    """Download the immutable full root needed by a cold-start replay."""
    if version.payload_format is not WeightPayloadFormat.FULL_HF_CHECKPOINT:
        raise ValueError("cold-start replay root must be FULL_HF_CHECKPOINT")
    if version.base_version_id is not None:
        raise ValueError("FULL_HF_CHECKPOINT must not have base_version_id")

    store = LocalCheckpointStore(
        root=refit_checkpoint_dir,
        model_name=model_name,
        max_size_bytes=(
            refit_checkpoint_max_size_gb * _BYTES_PER_GB
            if refit_checkpoint_max_size_gb is not None
            else None
        ),
    )
    store.initialize()
    target = store.full_path(version.version_id)
    with store.installation_locked(), store.locked():
        protected_versions = _protected_versions(store, version.version_id)
        state = store.state()
        if state is not None:
            protected_versions.add(state.version)
        _ensure_full_checkpoint(
            store=store,
            target=target,
            version=version,
            s3=s3,
            protected_versions=protected_versions,
        )

        store.write_chain(
            version.version_id,
            {
                "version": version.version_id,
                "full_version": version.version_id,
                "deltas": [],
            },
        )
        # Bootstrap only populates immutable cache data. A later TP rank may
        # arrive after another rank has prepared the target, so this path must
        # not change shared preparation or engine-installation state.
        store.enforce_capacity(protected_versions=protected_versions)
    return target


class _LocalCheckpoint:
    """Immutable local artifacts plus a materialized install checkpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        config: ObjectStorageGeneratorConfig,
        s3: S3Client,
    ) -> None:
        self.initial_version = config.initial_base_version_id
        self.s3 = s3
        self.store = LocalCheckpointStore(
            root=config.refit_checkpoint_dir,
            model_name=model_name,
            max_size_bytes=(
                config.refit_checkpoint_max_size_gb * _BYTES_PER_GB
                if config.refit_checkpoint_max_size_gb is not None
                else None
            ),
        )
        self.local_checkpoint = self.store.full_path(self.initial_version)
        self.seed_checkpoint_path = (
            Path(config.seed_checkpoint_path)
            if config.seed_checkpoint_path is not None
            else self.local_checkpoint
        )
        self.checkpoint_paths: list[Path] = []
        self.locations: dict[str, tuple[Path, int, int]] = {}
        self.tensor_metadata: dict[str, dict] = {}

    def initialize(self, *, allow_unrecorded_seed: bool = False) -> bool:
        """Initialize the disk seed, or return False if bootstrap is deferred."""
        self.store.initialize()
        with self.store.installation_locked(), self.store.locked():
            initial_checkpoint = self.store.full_path(self.initial_version)
            # An omitted seed or an explicit cached-root path reuses the cache.
            cached_seed = (
                self.seed_checkpoint_path.resolve() == initial_checkpoint.resolve()
            )
            if (
                allow_unrecorded_seed
                and cached_seed
                and not self.store.has_artifact_record(initial_checkpoint)
                and self.store.state() is None
            ):
                return False
            if cached_seed and not initial_checkpoint.is_dir():
                raise FileNotFoundError(
                    f"cached seed checkpoint for {self.initial_version!r} "
                    f"not found: {initial_checkpoint}"
                )
            version = self._initialize_checkpoint_state(
                state=self.store.state(),
                initial_checkpoint=initial_checkpoint,
                cached_seed=cached_seed,
            )
            if not cached_seed:
                self.store.activate(self.initial_version)
            self.store.enforce_capacity(
                protected_versions=_protected_versions(self.store, version),
            )

        return True

    def _initialize_checkpoint_state(
        self,
        *,
        state: CheckpointRecord | None,
        initial_checkpoint: Path,
        cached_seed: bool,
    ) -> str:
        version = self.initial_version
        if (
            cached_seed
            and state is not None
            and state.status is not CheckpointState.UPDATING
        ):
            # A co-located rank may already have reconstructed the desired
            # target. Reuse that verified cache head instead of rewinding it.
            version = state.version
        reusable = self._reuse_ready_checkpoint(state=state, version=version)
        if reusable and (
            cached_seed
            or self.store.chain(self.initial_version)
            == {
                "version": self.initial_version,
                "full_version": self.initial_version,
                "deltas": [],
            }
        ):
            return version

        self.local_checkpoint = initial_checkpoint
        if cached_seed:
            self._restore_cached_initial_checkpoint()
        else:
            self.reset_initial_checkpoint()
        return self.initial_version

    def _reuse_ready_checkpoint(
        self,
        *,
        state: CheckpointRecord | None,
        version: str,
    ) -> bool:
        if (
            state is None
            or state.status is CheckpointState.UPDATING
            or state.version != version
        ):
            return False
        try:
            checkpoint = self.store.checkpoint_path(state.files_version)
            if not any(checkpoint.glob("*.safetensors")):
                return False
            self._set_local_checkpoint(checkpoint)
        except (OSError, RuntimeError, ValueError):
            return False
        if state.files != checkpoint_files_state(self.checkpoint_paths):
            return False
        return True

    def _restore_cached_initial_checkpoint(self) -> None:
        """Recover incomplete preparation without copying over the S3 root."""
        self.store.verify_artifact(self.local_checkpoint)
        self._set_local_checkpoint(self.local_checkpoint)
        self.store.write_chain(
            self.initial_version,
            {
                "version": self.initial_version,
                "full_version": self.initial_version,
                "deltas": [],
            },
        )
        self.store.write_state(
            status=CheckpointState.READY,
            version=self.initial_version,
            checkpoint_paths=self.checkpoint_paths,
        )

    def recover_incomplete_preparation(self) -> None:
        """Restore a usable cache head after an in-process replay failure."""
        with self.store.installation_locked(), self.store.locked():
            state = self.store.state()
            if state is None or state.status is not CheckpointState.UPDATING:
                return
            self.local_checkpoint = self.store.full_path(self.initial_version)
            self._restore_cached_initial_checkpoint()
            self.store.enforce_capacity(
                protected_versions=_protected_versions(
                    self.store,
                    self.initial_version,
                ),
            )

    def _set_local_checkpoint(self, path: Path) -> None:
        self.local_checkpoint = path
        (
            self.checkpoint_paths,
            self.locations,
            self.tensor_metadata,
        ) = index_checkpoint_tensors(path)

    def reset_initial_checkpoint(self) -> None:
        """Restore the immutable initial full checkpoint from the launch seed."""
        state = self.store.state()
        target = self.store.full_path(self.initial_version)
        self.store.ensure_capacity(
            self.store.path_size_bytes(self.seed_checkpoint_path),
            protected_versions={self.initial_version},
        )
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=(
                state.version
                if state is not None
                else self.initial_version
            ),
            checkpoint_paths=self.checkpoint_paths,
        )
        with self.store.replace_directory(target) as temporary:
            if self.seed_checkpoint_path.is_file():
                shutil.copy2(
                    self.seed_checkpoint_path,
                    temporary / "model.safetensors",
                )
            elif self.seed_checkpoint_path.is_dir():
                for entry in self.seed_checkpoint_path.iterdir():
                    if entry.is_file():
                        shutil.copy2(entry, temporary / entry.name)
                    elif entry.is_dir():
                        shutil.copytree(entry, temporary / entry.name)
            else:
                raise FileNotFoundError(
                    f"seed checkpoint does not exist: {self.seed_checkpoint_path}"
                )
            self.store.record_artifact(temporary)
        self._set_local_checkpoint(target)
        self.store.write_chain(
            self.initial_version,
            {
                "version": self.initial_version,
                "full_version": self.initial_version,
                "deltas": [],
            },
        )
        self.store.write_state(
            status=CheckpointState.READY,
            version=self.initial_version,
            checkpoint_paths=self.checkpoint_paths,
        )

    def prepare(self, version: _S3Version) -> PreparedCheckpoint:
        return self.prepare_chain((version,))

    def prepare_chain(
        self,
        versions: tuple[_S3Version, ...],
        *,
        staged: bool = False,
        announce: bool = True,
    ) -> PreparedCheckpoint:
        """Prepare an ordered chain into one immutable target checkpoint.

        With ``staged``, the result may be STAGED, with files that still hold
        its base. ``announce`` records the target while this waits to prepare,
        so a prefetcher rewriting the cache head can switch to it.
        """
        if not versions:
            raise ValueError("canonical replay chain is empty")
        target = versions[-1]
        if announce:
            self.store.write_wanted(
                {
                    "version": target.version_id,
                    "base_version": target.base_version_id,
                    "uri": target.uri,
                    "chain_length": len(versions),
                    "staged": staged,
                }
            )
        while True:
            # Queue preparers separately so followers can reuse the target while
            # the leader installs it, without first waiting for an exclusive fence.
            with self.store.preparation_locked():
                if announce:
                    self.store.clear_wanted()
                    announce = False
                state = self.store.state()
                if (
                    state is None
                    or state.status is CheckpointState.UPDATING
                    or (state.status is CheckpointState.STAGED and not staged)
                    or state.version != target.version_id
                ):
                    return self._prepare_chain_locked(versions, staged=staged)
            # Co-located ranks verify a target that is already the cache head
            # concurrently rather than one at a time in the preparation queue.
            with self.store.installation_locked(shared=True), self.store.locked(
                shared=True
            ):
                state, _active_version = self._load_prepared_state()
                prepared = self._reuse_prepared_target(
                    state=state, target=target, staged=staged
                )
            if prepared is not None:
                return prepared

    def prepare_speculative(
        self, version: _S3Version, *, staged: bool = False
    ) -> PreparedCheckpoint:
        """Prepare a delta that is not registered yet on top of the cache head.

        The result keeps a speculative ID until a registered version with the same
        URI and base adopts it during a later preparation.
        """
        with self.store.preparation_locked():
            with self.store.installation_locked(), self.store.locked():
                state, _active_version = self._load_prepared_state()
                if state.version != version.base_version_id:
                    raise RuntimeError("speculative base is no longer the cache head")
                self.store.write_speculation(
                    version=version.version_id,
                    uri=version.uri,
                    base_version=state.version,
                    base_source=state.source,
                )
            return self._prepare_chain_locked((version,), staged=staged)

    def materialize_head(
        self,
        *,
        staged: bool,
        following: Callable[[], tuple[_S3Version, bool] | None] | None = None,
    ) -> PreparedCheckpoint | None:
        """Write a STAGED cache head's deltas into checkpoint files; return the result.

        ``following`` is asked between tensors for a delta on top of the head,
        and whether that delta is speculative. Below MX_REFIT_STAGED_LAYERS
        staged deltas, they are written into a copy of the files that replaces
        them once complete, and a delta ``following`` names discards the copy
        and is checked through the staged deltas instead. Without a copy that
        shares the files' blocks, or room for one, nothing is written. At the
        limit, the files are rewritten in place, and a delta ``following`` names
        joins that pass. Returns the READY head or the following delta, STAGED,
        and None when the head is not STAGED or nothing was prepared.
        """
        with self.store.preparation_locked():
            with self.store.installation_locked(), self.store.locked():
                state, _active_version = self._load_prepared_state()
                if state.status is not CheckpointState.STAGED:
                    return None
                head = state.version
                layers = _staged_layers(self.store, state)

                def next_delta() -> _S3Version | None:
                    found = following() if following is not None and staged else None
                    if found is None:
                        return None
                    version, speculative = found
                    if version.base_version_id != head:
                        return None
                    if speculative:
                        self.store.write_speculation(
                            version=version.version_id,
                            uri=version.uri,
                            base_version=head,
                            base_source=state.source,
                        )
                    return version

                if not staged or len(layers) >= rl_envs.MX_REFIT_STAGED_LAYERS:
                    return self._materialize_staged(state, following=next_delta)
                return self._rewrite_copy(state, following=next_delta)

    def is_prepared(self, *, version_id: str, uri: str) -> bool:
        """Return whether the cache head already holds this version."""
        state = self.store.state()
        if state is None or state.status is CheckpointState.UPDATING:
            return False
        if state.version == version_id:
            return True
        speculation = self.store.speculation()
        return (
            speculation is not None
            and speculation.get("version") == state.version
            and speculation.get("uri") == uri
        )

    def _resolve_speculation(self, versions: tuple[_S3Version, ...]) -> None:
        """Adopt a matching speculative cache head, or undo one that is not wanted.

        Callers hold the exclusive installation and cache locks. A crash while the
        head changes leaves UPDATING, which initialization rebuilds from lineage.
        """
        speculation = self.store.speculation()
        if speculation is None:
            return
        state = self.store.state()
        if state is None or state.status is CheckpointState.UPDATING:
            return
        if state.version != speculation.get("version"):
            if all(version.version_id != speculation.get("version") for version in versions):
                # The speculative preparation never finished, or the head moved on.
                self.store.clear_speculation()
            return
        registered = next(
            (
                version
                for version in versions
                if version.uri == speculation.get("uri")
                and version.base_version_id == speculation.get("base_version")
            ),
            None,
        )
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=state.version,
            checkpoint_paths=self.checkpoint_paths,
        )
        if registered is None:
            self._revert_speculation(
                speculation, files=state.base if state.status is CheckpointState.STAGED else None
            )
        else:
            self.store.rename_version(state.version, registered.version_id)
            self._set_local_checkpoint(
                self.store.checkpoint_path(
                    state.base if state.base is not None else registered.version_id
                )
            )
            self.store.write_state(
                status=state.status,
                version=registered.version_id,
                checkpoint_paths=self.checkpoint_paths,
                source=_source_identity(registered),
                base=state.base,
            )
        self.store.clear_speculation()

    def _revert_speculation(self, speculation: dict[str, Any], *, files: str | None = None) -> None:
        """Restore the registered base under a speculative head.

        XOR deltas are their own inverse, so applying the speculative delta again
        restores the base bytes exactly. A full base was never modified, and
        neither were the files under a STAGED head, which hold ``files``. The
        base is STAGED again when other deltas are staged on those files.
        """
        staged = files is not None
        version = _S3Version(
            version_id=speculation["version"],
            base_version_id=speculation["base_version"],
            payload_format=WeightPayloadFormat.XOR_DELTA,
            uri=speculation["uri"],
        )
        base_chain = self.store.chain(version.base_version_id)
        if base_chain is None:
            raise RuntimeError("speculative base chain is missing")
        speculative_checkpoint = self.store.materialized_path(version.version_id)
        if staged:
            pass
        elif base_chain.get("deltas"):
            artifact = self.store.delta_path(version.version_id)
            index_data = (artifact / Path(version.uri).name).read_bytes()
            metadata, weight_map = _parse_index_manifest(
                index_data,
                is_delta=True,
                version=version,
            )
            self._set_local_checkpoint(speculative_checkpoint)
            self._apply_delta_artifact(artifact, metadata, weight_map, verify=False)
            base_checkpoint = self.store.materialized_path(version.base_version_id)
            shutil.rmtree(base_checkpoint, ignore_errors=True)
            speculative_checkpoint.replace(base_checkpoint)
        else:
            shutil.rmtree(speculative_checkpoint, ignore_errors=True)
        self.store.chain_path(version.version_id).unlink(missing_ok=True)
        shutil.rmtree(self.store.delta_path(version.version_id), ignore_errors=True)
        shutil.rmtree(self.store.staging_path(version.version_id), ignore_errors=True)
        base = None if files in (None, version.base_version_id) else files
        self._set_local_checkpoint(self.store.checkpoint_path(base or version.base_version_id))
        self.store.write_state(
            status=CheckpointState.READY if base is None else CheckpointState.STAGED,
            version=version.base_version_id,
            checkpoint_paths=self.checkpoint_paths,
            source=speculation.get("base_source"),
            base=base,
        )

    def _prepare_chain_locked(
        self,
        versions: tuple[_S3Version, ...],
        *,
        staged: bool,
    ) -> PreparedCheckpoint:
        target = versions[-1]
        with self.store.installation_locked(shared=True), self.store.locked(
            shared=True
        ):
            state, _active_version = self._load_prepared_state()
            prepared = self._reuse_prepared_target(state=state, target=target, staged=staged)
            if prepared is not None:
                return prepared
        with self.store.installation_locked(), self.store.locked():
            self._resolve_speculation(versions)
            state, active_version = self._load_prepared_state()
            if (
                not staged
                and state.status is CheckpointState.STAGED
                and state.version == target.version_id
            ):
                # A caller that needs the target's own files waits for them.
                self._materialize_staged(state)
                state, active_version = self._load_prepared_state()
            prepared = self._reuse_prepared_target(state=state, target=target, staged=staged)
            if prepared is not None:
                return prepared
            # P2P can advance the engine without advancing this disk checkpoint.
            # Choose the replay suffix from the verified cache head under lock.
            for position, version in enumerate(versions[:-1]):
                if version.version_id != state.version:
                    continue
                source = self.store.artifact_source(self._artifact_path(version))
                if source is None:
                    break
                if source != _source_identity(version):
                    raise ValueError("prepared checkpoint has different source identity")
                versions = versions[position + 1 :]
                break
            manifests, index_download_time = self._download_replay_manifests(
                versions=versions,
                target=target,
                base_version=state.version,
            )
            return self._reconstruct_target(
                manifests=manifests,
                target=target,
                active_version=active_version,
                index_download_time=index_download_time,
                staged=staged,
            )

    def _load_prepared_state(self) -> tuple[CheckpointRecord, str]:
        """Return the READY or STAGED head and the active version, indexing its files."""
        state = self.store.state()
        if state is None:
            raise RuntimeError("local checkpoint state is missing")
        if state.status is CheckpointState.UPDATING:
            raise RuntimeError("local checkpoint update is incomplete")
        active_version = _active_version(self.store) or state.version
        state_checkpoint = self.store.checkpoint_path(state.files_version)
        # A process that attaches without initializing has indexed nothing yet.
        if self.local_checkpoint != state_checkpoint or not self.checkpoint_paths:
            self._set_local_checkpoint(state_checkpoint)
        if state.files != checkpoint_files_state(self.checkpoint_paths):
            raise RuntimeError("local checkpoint files changed outside ModelExpress")
        return state, active_version

    def _reuse_prepared_target(
        self,
        *,
        state: CheckpointRecord,
        target: _S3Version,
        staged: bool,
    ) -> PreparedCheckpoint | None:
        if state.version != target.version_id:
            return None
        if state.status is CheckpointState.STAGED and not staged:
            return None
        delta_changes = self._delta_changes(target.version_id)
        # Followers reach this after the lock holder has reconstructed the
        # target. Source identity prevents attaching to a reused UID artifact.
        self.store.verify_artifact_source(
            self._artifact_path(target),
            _source_identity(target),
        )
        staging, staging_base = self._staged_from(target.version_id)
        return PreparedCheckpoint(
            target_version=target.version_id,
            path=self.local_checkpoint,
            metrics={
                "perf/mx_receive_delta_index_download": 0.0,
                "perf/mx_receive_delta_download": 0.0,
                "perf/mx_receive_delta_apply": 0.0,
            },
            delta_changes=delta_changes,
            staging=staging,
            staging_base=staging_base,
            overlay=self._overlay(state),
            overlay_base=state.base,
        )

    def _overlay(self, state: CheckpointRecord) -> tuple[Path, ...]:
        """Staging of each delta staged on the head's files, oldest first."""
        paths = []
        base = state.files_version
        for version in _staged_layers(self.store, state):
            path = self.store.staging_path(version)
            if StagedDelta.base_of(path) != base:
                raise RuntimeError(f"staged target {state.version!r} lost its staging")
            paths.append(path)
            base = version
        return tuple(paths)

    def _open_layers(self, state: CheckpointRecord) -> list[StagedDelta]:
        """Open the deltas staged on the head's files, oldest first."""
        layers: list[StagedDelta] = []
        try:
            for path in self._overlay(state):
                try:
                    layer = StagedDelta(path)
                except (OSError, ValueError) as error:
                    raise RuntimeError(f"staging {path} is unreadable") from error
                layers.append(layer)
                unknown = [name for name in layer.names() if name not in self.locations]
                if unknown:
                    raise ValueError(f"staged tensors {unknown[:3]} are absent from the checkpoint")
        except BaseException:
            for layer in layers:
                layer.close()
            raise
        return layers

    def _delta_changes(self, version_id: str) -> tuple[DeltaChange, ...]:
        """Changed names per delta from the full root, or ``()`` if unknown.

        The chain record orders deltas by registry lineage. Index metadata names
        writer versions, which need not equal registry version IDs.
        """
        if not rl_envs.MX_REFIT_DELTA_SURGICAL:
            return ()
        chain = self.store.chain(version_id)
        if chain is None:
            return ()
        changes = []
        base = chain["full_version"]
        for delta in chain["deltas"]:
            indexes = list(self.store.delta_path(delta).glob("*.index.json"))
            if len(indexes) != 1:
                return ()
            try:
                index = json.loads(indexes[0].read_text())
            except (OSError, ValueError):
                return ()
            if not isinstance(index, dict):
                return ()
            weight_map = index.get("weight_map")
            metadata = index.get("metadata")
            if (
                not isinstance(weight_map, dict)
                or not isinstance(metadata, dict)
                or metadata.get("delta_encoding") != "xor"
            ):
                return ()
            changes.append(DeltaChange(base, delta, frozenset(weight_map)))
            base = delta
        return tuple(changes)

    def _download_replay_manifests(
        self,
        *,
        versions: tuple[_S3Version, ...],
        target: _S3Version,
        base_version: str,
    ) -> tuple[list[_S3Manifest], float]:
        """Validate the entire chain before mutating preparation state."""
        expected_base = base_version
        manifests = []
        index_download_time = 0.0
        for position, version in enumerate(versions):
            if (
                position > 0
                and version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
            ):
                raise RuntimeError(
                    f"full checkpoint revision {version.version_id!r} must be "
                    "the first replay revision"
                )
            if (
                version.payload_format is WeightPayloadFormat.XOR_DELTA
                and expected_base != version.base_version_id
            ):
                raise RuntimeError(
                    f"local checkpoint version {expected_base!r} does not match "
                    f"exact base {version.base_version_id!r} for revision "
                    f"{version.version_id!r}"
                )
            expected_base = version.version_id
            started = time.perf_counter()
            try:
                index_data = self.s3.get(version.uri)
                is_delta = version.payload_format is WeightPayloadFormat.XOR_DELTA
                metadata, weight_map = _parse_index_manifest(
                    index_data,
                    is_delta=is_delta,
                    version=version if is_delta else None,
                )
            except Exception as error:
                raise RuntimeError(
                    f"target {target.version_id!r}: replay validation failed "
                    f"at revision {version.version_id!r}: {error}"
                ) from error
            index_download_time += time.perf_counter() - started
            manifests.append((version, index_data, metadata, weight_map))
        return manifests, index_download_time

    def _reconstruct_target(
        self,
        *,
        manifests: list[_S3Manifest],
        target: _S3Version,
        active_version: str,
        index_download_time: float,
        staged: bool,
    ) -> PreparedCheckpoint:
        state = self.store.state()
        assert state is not None
        first = manifests[0][0] if manifests else None
        if (
            staged
            and rl_envs.MX_REFIT_INPLACE
            and len(manifests) == 1
            and first is not None
            and first.payload_format is WeightPayloadFormat.XOR_DELTA
        ):
            return self._stage_target(state, manifests[0], index_download_time)
        if (
            state.status is CheckpointState.STAGED
            and first is not None
            and first.payload_format is WeightPayloadFormat.XOR_DELTA
        ):
            # The first delta applies to the head's own bytes, so write them first.
            self._materialize_staged(state)
        # READY -> UPDATING is persisted before any derived checkpoint mutates.
        # A crash therefore makes the next initializer rebuild from lineage.
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=active_version,
            checkpoint_paths=self.checkpoint_paths,
        )

        failed_version = target
        try:
            download_time = 0.0
            apply_time = 0.0
            for version, index_data, metadata, weight_map in manifests:
                failed_version = version
                if version.payload_format is WeightPayloadFormat.XOR_DELTA:
                    downloaded, applied = self._prepare_delta(
                        version,
                        index_data,
                        metadata,
                        weight_map,
                    )
                else:
                    downloaded, applied = self._prepare_full(
                        version,
                        index_data,
                        metadata,
                        weight_map,
                    )
                download_time += downloaded
                apply_time += applied
        except CheckpointCacheCapacityError:
            self._set_local_checkpoint(self.store.checkpoint_path(active_version))
            self.store.write_state(
                status=CheckpointState.READY,
                version=active_version,
                checkpoint_paths=self.checkpoint_paths,
            )
            raise
        except Exception as error:
            raise RuntimeError(
                f"target {target.version_id!r}: replay failed at revision "
                f"{failed_version.version_id!r}: {error}"
            ) from error

        self.store.write_state(
            status=CheckpointState.READY,
            version=target.version_id,
            checkpoint_paths=self.checkpoint_paths,
            source=_source_identity(target),
        )
        staging, staging_base = self._staged_from(target.version_id)
        return PreparedCheckpoint(
            target_version=target.version_id,
            path=self.local_checkpoint,
            metrics={
                "perf/mx_receive_delta_index_download": index_download_time,
                "perf/mx_receive_delta_download": download_time,
                "perf/mx_receive_delta_apply": apply_time,
            },
            delta_changes=self._delta_changes(target.version_id),
            staging=staging,
            staging_base=staging_base,
        )

    def _artifact_path(self, version: _S3Version) -> Path:
        if version.payload_format is WeightPayloadFormat.XOR_DELTA:
            return self.store.delta_path(version.version_id)
        return self.store.full_path(version.version_id)

    @staticmethod
    def _copy_non_weight_files(source: Path, target: Path) -> None:
        if not source.is_dir():
            return
        for entry in source.iterdir():
            if entry.name.endswith(".safetensors") or entry.name.endswith(
                ".safetensors.index.json"
            ):
                continue
            destination = target / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination)
            elif entry.is_file():
                shutil.copy2(entry, destination)

    @staticmethod
    def _non_weight_files_size(source: Path) -> int:
        if not source.is_dir():
            return 0
        return sum(
            LocalCheckpointStore.path_size_bytes(entry)
            for entry in source.iterdir()
            if not entry.name.endswith(".safetensors")
            and not entry.name.endswith(".safetensors.index.json")
        )

    def _prepare_full(
        self,
        version: _S3Version,
        index_data: bytes,
        metadata: dict[str, Any],
        weight_map: dict[str, str],
    ) -> tuple[float, float]:
        target = self.store.full_path(version.version_id)
        if target.exists():
            source = self.store.artifact_source(target)
            if source is not None:
                self.store.enforce_capacity(
                    protected_versions=_protected_versions(
                        self.store,
                        version.version_id,
                    ),
                )
                self.store.verify_artifact_source(
                    target,
                    _source_identity(version),
                )
                self._set_local_checkpoint(target)
                self.store.write_chain(
                    version.version_id,
                    {
                        "version": version.version_id,
                        "full_version": version.version_id,
                        "deltas": [],
                    },
                )
                return 0.0, 0.0

        non_weight_source = self.seed_checkpoint_path
        if (
            non_weight_source.resolve()
            == self.store.full_path(self.initial_version).resolve()
        ):
            # Updates carry seed metadata forward; the initial root may be evicted.
            non_weight_source = self.local_checkpoint
        protected_versions = _protected_versions(self.store, version.version_id)
        self.store.ensure_capacity(
            self._non_weight_files_size(non_weight_source) + len(index_data),
            protected_versions=protected_versions,
            protected_paths={non_weight_source},
        )
        with self.store.replace_directory(target) as temporary:
            self._copy_non_weight_files(non_weight_source, temporary)
            index_name = Path(version.uri).name
            (temporary / index_name).write_bytes(index_data)
            download_time, validation_time = self._download_full_checkpoint(
                target=temporary,
                index_metadata=metadata,
                weight_map=weight_map,
                root_uri=version.uri,
                protected_versions=protected_versions,
            )
            self.store.record_artifact(
                temporary,
                source=_source_identity(version),
            )

        self._set_local_checkpoint(target)
        self.store.write_chain(
            version.version_id,
            {
                "version": version.version_id,
                "full_version": version.version_id,
                "deltas": [],
            },
        )
        return download_time, validation_time

    def _delta_chain(self, version: _S3Version) -> dict[str, Any]:
        """Return the chain record of a delta on top of its exact base."""
        assert version.base_version_id is not None
        base_chain = self.store.chain(version.base_version_id)
        if base_chain is None:
            raise RuntimeError(
                "checkpoint chain for exact base "
                f"{version.base_version_id!r} is missing"
            )
        return {
            "version": version.version_id,
            "full_version": base_chain["full_version"],
            "deltas": [*base_chain["deltas"], version.version_id],
        }

    def _verify_lineage(self, chain: dict[str, Any]) -> None:
        """Check the cached artifacts a chain's base was built from."""
        self.store.verify_artifact(self.store.full_path(chain["full_version"]))
        for delta_version in chain["deltas"][:-1]:
            self.store.verify_artifact(self.store.delta_path(delta_version))

    def _delta_shards(
        self,
        version: _S3Version,
        index_data: bytes,
        weight_map: dict[str, str],
        *,
        protected_versions: set[str],
        protected_paths: Iterable[Path] = (),
    ) -> tuple[dict[str, tuple[bytes, list[str]]], float]:
        """Return a delta's shards and the time to get them.

        A delta downloaded now is recorded as an artifact and returned from
        memory rather than read back from its copy on disk.
        """
        artifact = self.store.delta_path(version.version_id)
        started = time.perf_counter()
        if artifact.exists():
            self.store.verify_artifact_source(artifact, _source_identity(version))
            self.store.verify_artifact(artifact)
            shards = {
                filename: ((artifact / filename).read_bytes(), names)
                for filename, names in _group_tensors_by_shard(weight_map).items()
            }
        else:
            shards = self._download_deltas(weight_map, version.uri)
            self.store.ensure_capacity(
                len(index_data) + sum(len(data) for data, _names in shards.values()),
                protected_versions=protected_versions,
                protected_paths=protected_paths,
            )
            with self.store.replace_directory(artifact) as temporary:
                (temporary / Path(version.uri).name).write_bytes(index_data)
                for filename, (data, _names) in shards.items():
                    (temporary / filename).write_bytes(data)
                self.store.record_artifact(
                    temporary,
                    source=_source_identity(version),
                )
        return shards, time.perf_counter() - started

    def _prepare_delta(
        self,
        version: _S3Version,
        index_data: bytes,
        metadata: dict[str, Any],
        weight_map: dict[str, str],
    ) -> tuple[float, float]:
        assert version.base_version_id is not None
        chain = self._delta_chain(version)
        protected_versions = _protected_versions(
            self.store,
            version.base_version_id,
            version.version_id,
        )
        shards, download_time = self._delta_shards(
            version, index_data, weight_map, protected_versions=protected_versions
        )
        target = self.store.materialized_path(version.version_id)
        apply_started = time.perf_counter()
        sink: dict[str, tuple[int, np.ndarray, np.ndarray]] | None = (
            {} if rl_envs.MX_REFIT_INPLACE else None
        )
        self._verify_lineage(chain)
        base_checkpoint = self.store.checkpoint_path(version.base_version_id)
        reuse_materialized = base_checkpoint.parent == self.store.materialized_cache
        self.store.ensure_capacity(
            0 if reuse_materialized else self.store.path_size_bytes(base_checkpoint),
            protected_versions=protected_versions,
        )
        if reuse_materialized:
            # A sequential delta can take ownership of the derived base. Rename
            # it to the target version and mutate it without another full copy.
            shutil.rmtree(target, ignore_errors=True)
            base_checkpoint.replace(target)
            self._set_local_checkpoint(target)
            self._apply_shards(shards, index_metadata=metadata, sink=sink)
        else:
            # The first delta after a full checkpoint copies once so applying
            # it cannot modify the canonical full artifact.
            with self.store.replace_directory(
                target,
                copy_from=base_checkpoint,
            ) as temporary:
                self._set_local_checkpoint(temporary)
                self._apply_shards(shards, index_metadata=metadata, sink=sink)

        self._set_local_checkpoint(target)
        self.store.write_chain(version.version_id, chain)
        if sink is not None:
            self._write_staging(version, sink)
        return download_time, time.perf_counter() - apply_started

    def _stage_target(
        self,
        state: CheckpointRecord,
        manifest: _S3Manifest,
        index_download_time: float,
        *,
        downloaded: tuple[dict[str, tuple[bytes, list[str]]], float] | None = None,
    ) -> PreparedCheckpoint:
        """Verify and stage one delta on the cache head, leaving its files for later.

        A pass reads the head's files, XORs in the deltas already staged on them,
        and checks the delta's target checksums without writing. Once
        MX_REFIT_STAGED_LAYERS deltas are staged, the same pass first writes them
        into the files. ``downloaded`` holds the delta's shards and download time
        when a caller already has them.
        """
        version, index_data, metadata, weight_map = manifest
        try:
            if (
                state.status is CheckpointState.STAGED
                and len(_staged_layers(self.store, state)) >= rl_envs.MX_REFIT_STAGED_LAYERS
            ):
                return self._materialize_staged(
                    state, then=manifest, index_download_time=index_download_time
                )
            chain = self._delta_chain(version)
            shards, download_time = downloaded or self._delta_shards(
                version,
                index_data,
                weight_map,
                protected_versions=_protected_versions(
                    self.store, state.version, version.version_id
                ),
            )
            started = time.perf_counter()
            self._verify_lineage(chain)
            sink: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
            layers = self._open_layers(state)
            try:
                self._run_pass(
                    [
                        _TensorPass(
                            name,
                            check=compressed,
                            checksum=checksum,
                            overlay=tuple(layer.tensor(name) for layer in layers if name in layer),
                        )
                        for name, (compressed, checksum) in self._shard_tensors(shards).items()
                    ],
                    index_metadata=metadata,
                    sink=sink,
                )
            finally:
                for layer in layers:
                    layer.close()
            self.store.write_chain(version.version_id, chain)
            self._write_staging(version, sink)
        except CheckpointCacheCapacityError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"target {version.version_id!r}: replay failed at revision "
                f"{version.version_id!r}: {error}"
            ) from error
        return self._staged_result(
            version,
            base=state.files_version,
            metrics={
                "perf/mx_receive_delta_index_download": index_download_time,
                "perf/mx_receive_delta_download": download_time,
                "perf/mx_receive_delta_apply": time.perf_counter() - started,
            },
        )

    def _staged_result(
        self,
        version: _S3Version,
        *,
        base: str,
        metrics: dict[str, float],
    ) -> PreparedCheckpoint:
        """Record a verified, staged delta as the STAGED head on files holding ``base``."""
        self.store.write_state(
            status=CheckpointState.STAGED,
            version=version.version_id,
            checkpoint_paths=self.checkpoint_paths,
            source=_source_identity(version),
            base=base,
        )
        state = self.store.state()
        assert state is not None
        overlay = self._overlay(state)
        staging, staging_base = self._staged_from(version.version_id)
        return PreparedCheckpoint(
            target_version=version.version_id,
            path=self.local_checkpoint,
            metrics={**metrics, "perf/mx_receive_staged_layers": float(len(overlay))},
            delta_changes=self._delta_changes(version.version_id),
            staging=staging,
            staging_base=staging_base,
            overlay=overlay,
            overlay_base=base,
        )

    def _materialize_staged(
        self,
        state: CheckpointRecord,
        *,
        then: _S3Manifest | None = None,
        following: Callable[[], _S3Version | None] | None = None,
        index_download_time: float = 0.0,
    ) -> PreparedCheckpoint:
        """Write the deltas staged on a STAGED head's files into them, in place.

        Alone, this makes the head READY. With ``then``, or once ``following``
        names a delta on top of the head, the rest of the rewrite and that
        delta's verification run as one pass, and that delta becomes the STAGED
        head on the rewritten files. Returns the new head's preparation.
        """
        head, head_metadata = self._staged_head(state)
        # Staging holds each delta's nonzero bytes, so the rewrite scatters
        # those instead of decompressing the deltas again.
        layers = self._open_layers(state)
        try:
            return self._rewrite_staged(
                state,
                head,
                head_metadata,
                _staged_tensors(layers),
                then=then,
                following=following,
                index_download_time=index_download_time,
            )
        finally:
            for layer in layers:
                layer.close()

    def _staged_head(self, state: CheckpointRecord) -> tuple[_S3Version, dict[str, Any]]:
        """Return a STAGED head's delta and its index metadata."""
        assert state.status is CheckpointState.STAGED and state.base is not None
        layers = _staged_layers(self.store, state)
        head = _S3Version(
            version_id=state.version,
            base_version_id=layers[-2] if len(layers) > 1 else state.base,
            payload_format=WeightPayloadFormat.XOR_DELTA,
            uri=(state.source or {}).get("uri", ""),
        )
        artifact = self.store.delta_path(head.version_id)
        self.store.verify_artifact(artifact)
        indexes = list(artifact.glob("*.index.json"))
        if len(indexes) != 1:
            raise RuntimeError(f"staged delta {head.version_id!r} has no single index")
        metadata, _weight_map = _parse_index_manifest(
            indexes[0].read_bytes(), is_delta=True, version=head
        )
        return head, metadata

    def _rewrite_staged(
        self,
        state: CheckpointRecord,
        head: _S3Version,
        head_metadata: dict[str, Any],
        pending: dict[str, tuple[StagedTensor, ...]],
        *,
        then: _S3Manifest | None,
        following: Callable[[], _S3Version | None] | None,
        index_download_time: float,
    ) -> PreparedCheckpoint:
        assert state.base is not None
        protected_versions = _protected_versions(self.store, state.base, head.version_id)
        shards = None
        download_time = 0.0
        if then is not None:
            protected_versions.add(then[0].version_id)
            shards, download_time = self._delta_shards(
                then[0], then[1], then[3], protected_versions=protected_versions
            )
        base_checkpoint = self.store.checkpoint_path(state.base)
        target = self.store.materialized_path(head.version_id)
        reuse_materialized = base_checkpoint.parent == self.store.materialized_cache
        self.store.ensure_capacity(
            0 if reuse_materialized else self.store.path_size_bytes(base_checkpoint),
            protected_versions=protected_versions,
        )
        # STAGED -> UPDATING before the files change, so a crash rebuilds from lineage.
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=head.version_id,
            checkpoint_paths=self.checkpoint_paths,
        )
        started = time.perf_counter()
        rewrite_seconds = 0.0
        staged_next: tuple[_S3Manifest, dict[str, tuple[int, np.ndarray, np.ndarray]]] | None = None

        def rewrite(protected_paths: set[Path]) -> None:
            nonlocal shards, download_time, rewrite_seconds, staged_next
            done: set[str] = set()
            manifest = then
            if manifest is None:
                found: list[_S3Version] = []

                def stop() -> bool:
                    version = following() if following is not None else None
                    if version is not None:
                        found.append(version)
                    return bool(found)

                done = self._run_pass(
                    [_TensorPass(name, scatter=staged) for name, staged in pending.items()],
                    index_metadata=head_metadata,
                    stop=stop if following is not None else None,
                )
                rewrite_seconds = time.perf_counter() - started
                if not found:
                    return
                try:
                    manifest, (shards, download_time), _index_seconds = self._fetch_following(
                        found[0],
                        head=head.version_id,
                        protected_versions=protected_versions,
                        protected_paths=protected_paths,
                    )
                except Exception as error:
                    # The head's own rewrite can still finish without the delta.
                    logger.warning(
                        "Rewrite of %s continues without %s: %s",
                        head.version_id,
                        found[0].version_id,
                        error,
                    )
                    self._run_pass(
                        [
                            _TensorPass(name, scatter=staged)
                            for name, staged in pending.items()
                            if name not in done
                        ],
                        index_metadata=head_metadata,
                    )
                    return
            metadata = manifest[2]
            assert shards is not None
            checked = self._shard_tensors(shards)
            sink: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
            self._run_pass(
                [
                    _TensorPass(
                        name,
                        scatter=() if name in done else pending.get(name, ()),
                        check=checked[name][0] if name in checked else None,
                        checksum=checked[name][1] if name in checked else None,
                    )
                    for name in (pending.keys() - done) | checked.keys()
                ],
                index_metadata=metadata,
                sink=sink,
            )
            staged_next = (manifest, sink)

        if reuse_materialized:
            shutil.rmtree(target, ignore_errors=True)
            base_checkpoint.replace(target)
            self._set_local_checkpoint(target)
            rewrite(set())
        else:
            with self.store.replace_directory(
                target,
                copy_from=base_checkpoint,
            ) as temporary:
                self._set_local_checkpoint(temporary)
                rewrite({temporary})
        self._set_local_checkpoint(target)
        apply_seconds = time.perf_counter() - started
        if staged_next is None:
            self.store.write_state(
                status=CheckpointState.READY,
                version=head.version_id,
                checkpoint_paths=self.checkpoint_paths,
                source=state.source,
            )
            staging, staging_base = self._staged_from(head.version_id)
            return PreparedCheckpoint(
                target_version=head.version_id,
                path=target,
                metrics={"perf/mx_receive_delta_rewrite": apply_seconds},
                delta_changes=self._delta_changes(head.version_id),
                staging=staging,
                staging_base=staging_base,
            )
        (version, _index_data, _metadata, _weight_map), sink = staged_next
        try:
            self.store.write_chain(version.version_id, self._delta_chain(version))
            self._write_staging(version, sink)
        except CheckpointCacheCapacityError:
            # The head's own rewrite finished, so keep it as READY.
            self.store.write_state(
                status=CheckpointState.READY,
                version=head.version_id,
                checkpoint_paths=self.checkpoint_paths,
                source=state.source,
            )
            raise
        return self._staged_result(
            version,
            base=head.version_id,
            metrics={
                "perf/mx_receive_delta_index_download": index_download_time,
                "perf/mx_receive_delta_download": download_time,
                "perf/mx_receive_delta_apply": apply_seconds - rewrite_seconds,
                "perf/mx_receive_delta_rewrite": rewrite_seconds,
            },
        )

    def _fetch_following(
        self,
        version: _S3Version,
        *,
        head: str,
        protected_versions: set[str],
        protected_paths: set[Path],
    ) -> tuple[_S3Manifest, tuple[_Shards, float], float]:
        """Download a delta on top of the head: its manifest, shards, and index time."""
        manifests, index_seconds = self._download_replay_manifests(
            versions=(version,), target=version, base_version=head
        )
        manifest = manifests[0]
        downloaded = self._delta_shards(
            manifest[0],
            manifest[1],
            manifest[3],
            protected_versions=protected_versions | {version.version_id},
            protected_paths=protected_paths,
        )
        return manifest, downloaded, index_seconds

    def _rewrite_copy(
        self,
        state: CheckpointRecord,
        *,
        following: Callable[[], _S3Version | None],
    ) -> PreparedCheckpoint | None:
        """Write the deltas staged on the head's files into a copy, then switch to it.

        The head stays STAGED on its own files until the copy is complete, so a
        delta ``following`` names between tensors can abandon the copy. That
        delta is then checked through the staged deltas instead. Returns the
        READY head, that delta STAGED, or None without a copy that shares blocks
        with the files or room for one.
        """
        assert state.base is not None
        if not self.store.clones_files():
            return None
        source = self.store.checkpoint_path(state.base)
        target = self.store.materialized_path(state.version)
        protected_versions = _protected_versions(self.store, state.base, state.version)
        try:
            self.store.ensure_capacity(
                self.store.path_size_bytes(source),
                protected_versions=protected_versions,
                protected_paths={source},
            )
        except CheckpointCacheCapacityError as error:
            logger.info("Deltas staged on %s stay unwritten: %s", state.version, error)
            return None
        _head, head_metadata = self._staged_head(state)
        started = time.perf_counter()
        layers = self._open_layers(state)
        try:
            jobs = [
                _TensorPass(name, scatter=staged)
                for name, staged in _staged_tensors(layers).items()
            ]
            found: list[_S3Version] = []

            def stop() -> bool:
                if not found and (version := following()) is not None:
                    found.append(version)
                return bool(found)

            with self.store.replace_directory(target, copy_from=source) as temporary:
                self._set_local_checkpoint(temporary)
                done = self._run_pass(jobs, index_metadata=head_metadata, stop=stop)
                if found:
                    try:
                        joined = self._fetch_following(
                            found[0],
                            head=state.version,
                            protected_versions=protected_versions,
                            protected_paths={temporary},
                        )
                    except Exception as error:
                        logger.warning(
                            "Rewrite of %s continues without %s: %s",
                            state.version,
                            found[0].version_id,
                            error,
                        )
                        self._run_pass(
                            [job for job in jobs if job.name not in done],
                            index_metadata=head_metadata,
                        )
                    else:
                        raise _CopyAbandoned(*joined)
        except _CopyAbandoned as abandoned:
            self._set_local_checkpoint(source)
            return self._stage_target(
                state,
                abandoned.manifest,
                abandoned.index_seconds,
                downloaded=abandoned.downloaded,
            )
        except BaseException:
            self._set_local_checkpoint(source)
            raise
        finally:
            for layer in layers:
                layer.close()
        self._set_local_checkpoint(target)
        self.store.write_state(
            status=CheckpointState.READY,
            version=state.version,
            checkpoint_paths=self.checkpoint_paths,
            source=state.source,
        )
        if source.parent == self.store.materialized_cache:
            shutil.rmtree(source, ignore_errors=True)
        staging, staging_base = self._staged_from(state.version)
        return PreparedCheckpoint(
            target_version=state.version,
            path=target,
            metrics={"perf/mx_receive_delta_rewrite": time.perf_counter() - started},
            delta_changes=self._delta_changes(state.version),
            staging=staging,
            staging_base=staging_base,
        )

    def _write_staging(
        self,
        version: _S3Version,
        tensors: dict[str, tuple[int, np.ndarray, np.ndarray]],
    ) -> None:
        assert version.base_version_id is not None
        self.store.ensure_capacity(
            sum(positions.nbytes + values.nbytes for _n, positions, values in tensors.values()),
            protected_versions=_protected_versions(
                self.store,
                version.base_version_id,
                version.version_id,
            ),
        )
        with self.store.replace_directory(
            self.store.staging_path(version.version_id)
        ) as temporary:
            write_staging(temporary, base_version=version.base_version_id, tensors=tensors)

    def _staged_from(self, version_id: str) -> tuple[Path | None, str | None]:
        """Return the staged delta into a version and its base, if staged."""
        if not rl_envs.MX_REFIT_INPLACE:
            return None, None
        staging = self.store.staging_path(version_id)
        base = StagedDelta.base_of(staging)
        return (staging, base) if base is not None else (None, None)

    def _apply_delta_artifact(
        self,
        artifact: Path,
        index_metadata: dict[str, Any],
        weight_map: dict[str, str],
        *,
        verify: bool = True,
        sink: dict[str, tuple[int, np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.store.verify_artifact(artifact)
        shards = {
            filename: (
                (artifact / filename).read_bytes(),
                names,
            )
            for filename, names in _group_tensors_by_shard(weight_map).items()
        }
        self._apply_shards(
            shards, index_metadata=index_metadata, verify=verify, sink=sink
        )

    def _download_deltas(
        self,
        weight_map: dict[str, str],
        root_uri: str,
    ) -> dict[str, tuple[bytes, list[str]]]:
        if not weight_map:
            return {}
        shard_to_tensors = _group_tensors_by_shard(weight_map)
        parent_uri = root_uri.rsplit("/", 1)[0]
        shards = {}

        def download(item: tuple[str, list[str]]):
            filename, names = item
            data = self.s3.get(f"{parent_uri}/{filename}")
            return filename, data, names

        for filename, data, names in threadpool_map(
            shard_to_tensors.items(),
            download,
            max_workers=min(
                rl_envs.MX_S3_DOWNLOAD_WORKERS,
                len(shard_to_tensors),
            ),
            thread_name_prefix="modelexpress-s3-download-file",
        ):
            shards[filename] = (data, names)
        return shards

    def _download_full_checkpoint(
        self,
        *,
        target: Path,
        index_metadata: dict[str, Any],
        weight_map: dict[str, str],
        root_uri: str,
        protected_versions: set[str],
    ) -> tuple[float, float]:
        return _download_full_checkpoint(
            s3=self.s3,
            store=self.store,
            target=target,
            index_metadata=index_metadata,
            weight_map=weight_map,
            root_uri=root_uri,
            protected_versions=protected_versions,
            tensor_metadata=self.tensor_metadata,
        )

    def _shard_tensors(
        self, shards: dict[str, tuple[bytes, list[str]]]
    ) -> dict[str, tuple[memoryview, str]]:
        """Return each delta tensor's compressed bytes and target checksum."""
        tensors = {}
        for filename, (data, names) in shards.items():
            header, data_start = read_safetensors_header(data, repr(filename))
            checksums = header.get("__metadata__")
            if not isinstance(checksums, dict):
                raise ValueError(
                    f"canonical delta shard {filename!r} is missing checksum metadata"
                )
            view = memoryview(data)
            for name in names:
                if name not in header:
                    raise ValueError(
                        f"canonical delta shard {filename!r} is missing tensor {name!r}"
                    )
                if name not in checksums:
                    raise ValueError(
                        f"canonical delta shard {filename!r} is missing checksum "
                        f"for tensor {name!r}"
                    )
                if name not in self.locations:
                    raise ValueError(
                        f"canonical delta tensor {name!r} from shard {filename!r} "
                        "is absent from the local checkpoint"
                    )
                begin, end = header[name]["data_offsets"]
                tensors[name] = (view[data_start + begin : data_start + end], checksums[name])
        return tensors

    def _apply_shards(
        self,
        shards: dict[str, tuple[bytes, list[str]]],
        *,
        index_metadata: dict[str, Any],
        verify: bool = True,
        sink: dict[str, tuple[int, np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """XOR delta shards into the checkpoint files, checking target checksums."""
        self._run_pass(
            [
                _TensorPass(name, write=compressed, checksum=checksum if verify else None)
                for name, (compressed, checksum) in self._shard_tensors(shards).items()
            ],
            index_metadata=index_metadata,
            sink=sink,
        )

    def _run_pass(
        self,
        tensors: list[_TensorPass],
        *,
        index_metadata: dict[str, Any],
        sink: dict[str, tuple[int, np.ndarray, np.ndarray]] | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> set[str]:
        """Run each tensor's XORs over the checkpoint files; return the names done.

        ``stop`` is asked before each tensor starts; once it returns True, the
        tensors already started finish and the rest are left as they are.
        """
        if not tensors:
            return set()

        # Take each file's tensors largest first, so a large tensor does not
        # finish last, and equal sizes in file order, so reads run forward.
        # Alternate files, since writes to one file serialize.
        by_file: defaultdict[Path, list[_TensorPass]] = defaultdict(list)
        for tensor in tensors:
            by_file[self.locations[tensor.name][0]].append(tensor)
        for group in by_file.values():
            group.sort(
                key=lambda tensor: (
                    -self.locations[tensor.name][2],
                    self.locations[tensor.name][1],
                )
            )
        order = [
            tensor
            for row in itertools.zip_longest(*by_file.values())
            for tensor in row
            if tensor is not None
        ]
        checksum_format = index_metadata["checksum_format"]
        decompress = _DECOMPRESSORS[index_metadata["compression_format"]]
        workers = min(rl_envs.MX_REFIT_DELTA_WORKERS, len(order))
        writes = any(tensor.write is not None or tensor.scatter for tensor in order)
        local = threading.local()
        started: set[str] = set()

        def apply_chunk(
            fd: int,
            offset: int,
            write: bytes | None,
            check: bytes | None,
            position: int,
            length: int,
            scatter: tuple[StagedTensor, ...],
            overlay: tuple[StagedTensor, ...],
            checksummed: bool,
        ) -> tuple[Adler32Checksum | None, int, tuple[np.ndarray, np.ndarray] | None]:
            """XOR one chunk of each delta into the file's bytes; write back the first."""
            if getattr(local, "buffer", None) is None:
                local.buffer = np.empty(_APPLY_CHUNK_BYTES, dtype=np.uint8)
            target = local.buffer[:length]
            view = memoryview(target)
            _read_exactly(fd, view, offset + position)
            if write is not None:
                np.bitwise_xor(target, np.frombuffer(write, dtype=np.uint8), out=target)
            for staged in scatter:
                _xor_staged(target, staged, position)
            if write is not None or scatter:
                _write_all(fd, view, offset + position)
            for staged in overlay:
                _xor_staged(target, staged, position)
            if check is not None:
                np.bitwise_xor(target, np.frombuffer(check, dtype=np.uint8), out=target)
            checksum = None
            if checksummed:
                checksum = checksum_factory(checksum_format)
                checksum.update(target)
            staged = check if check is not None else write
            nonzero = (
                nonzero_bytes(np.frombuffer(staged, dtype=np.uint8), position)
                if sink is not None and staged is not None
                else None
            )
            return checksum, length, nonzero

        files: dict[Path, int] = {}
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="modelexpress-delta-chunk"
        ) as chunks:

            def apply_one(tensor: _TensorPass) -> None:
                path, file_offset, size = self.locations[tensor.name]
                fd = files[path]
                checksum = (
                    checksum_factory(checksum_format) if tensor.checksum is not None else None
                )
                nonzero = []
                # A large tensor's chunks spread over the workers; others stay here.
                spread = workers > 1 and size > _SPREAD_TENSOR_BYTES
                pending: deque[Future] = deque()

                def collect(result) -> None:
                    part, length, part_nonzero = result
                    if checksum is not None:
                        checksum.extend(part, length)
                    if part_nonzero is not None:
                        nonzero.append(part_nonzero)

                readers = [
                    decompress(compressed) if compressed is not None else None
                    for compressed in (tensor.write, tensor.check)
                ]
                position = 0
                extra = False
                try:
                    while position < size:
                        length = min(_APPLY_CHUNK_BYTES, size - position)
                        write, check = (
                            _read_block(reader, length) if reader is not None else None
                            for reader in readers
                        )
                        blocks = [block for block in (write, check) if block is not None]
                        if blocks:
                            length = len(blocks[0])
                        if not length or any(len(block) != length for block in blocks):
                            break
                        arguments = (
                            fd,
                            file_offset,
                            write,
                            check,
                            position,
                            length,
                            tensor.scatter,
                            tensor.overlay,
                            checksum is not None,
                        )
                        if spread:
                            pending.append(chunks.submit(apply_chunk, *arguments))
                            if len(pending) >= 2 * workers:
                                collect(pending.popleft().result())
                        else:
                            collect(apply_chunk(*arguments))
                        position += length
                    if position == size:
                        extra = any(reader is not None and reader.read(1) for reader in readers)
                    while pending:
                        collect(pending.popleft().result())
                finally:
                    for reader in readers:
                        if reader is not None:
                            reader.close()
                    for future in pending:
                        future.cancel()
                    wait(pending)
                if position != size or extra:
                    raise ValueError(f"canonical delta byte size differs for {tensor.name!r}")
                # Checksums describe the delta's target, not a reverted base.
                if checksum is not None and checksum.hexdigest() != tensor.checksum:
                    raise ValueError(f"canonical target checksum differs for {tensor.name!r}")
                # Staging holds the delta this pass checks or applies, so tensors
                # it only writes staged bytes into stay out of it.
                if sink is not None and (tensor.check is not None or tensor.write is not None):
                    sink[tensor.name] = (size, *join_nonzero(size, nonzero))

            def scheduled() -> Iterator[_TensorPass]:
                for tensor in order:
                    if stop is not None and stop():
                        return
                    started.add(tensor.name)
                    yield tensor

            try:
                for path in by_file:
                    files[path] = os.open(path, os.O_RDWR if writes else os.O_RDONLY)
                list(
                    threadpool_map(
                        scheduled(),
                        apply_one,
                        max_workers=workers,
                        thread_name_prefix="modelexpress-delta-apply",
                    )
                )
            finally:
                for descriptor in files.values():
                    os.close(descriptor)
        return started

    def _validate_prepared(
        self,
        prepared: PreparedCheckpoint,
        *,
        message: str,
    ) -> None:
        """Ensure the prepared checkpoint still matches durable store state."""
        state = self.store.state()
        if (
            state is None
            or state.status
            is not (CheckpointState.READY if prepared.materialized else CheckpointState.STAGED)
            or state.version != prepared.target_version
            or (not prepared.materialized and state.base != prepared.overlay_base)
            or state.files != checkpoint_files_state(self.checkpoint_paths)
        ):
            raise ReceiverInstallError(message)

    def activate(self, prepared: PreparedCheckpoint) -> None:
        """Commit a prepared checkpoint after every local installer succeeds."""
        with self.store.installation_locked(), self.store.locked():
            self._validate_prepared(
                prepared,
                message="prepared checkpoint changed before activation",
            )
            self.store.activate(prepared.target_version)
            self.store.enforce_capacity(
                protected_versions=_protected_versions(self.store, prepared.target_version),
            )
        self.store.notify_prefetcher()

    @contextmanager
    def installation_context(
        self,
        prepared: PreparedCheckpoint,
        *,
        activate: bool = True,
    ):
        """Hold installation locks and optionally activate on successful exit."""
        with self.store.installation_locked(shared=True):
            with self.store.locked(shared=True):
                self._validate_prepared(
                    prepared,
                    message="prepared checkpoint changed before installation",
                )
                yield
            if activate:
                with self.store.locked():
                    self._validate_prepared(
                        prepared,
                        message="prepared checkpoint changed during installation",
                    )
                    self.store.activate(prepared.target_version)
                    self.store.enforce_capacity(
                        protected_versions=_protected_versions(
                            self.store, prepared.target_version
                        ),
                    )
                self.store.notify_prefetcher()


__all__ = [
    "DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB",
    "PreparedCheckpoint",
    "ReceiverInstallError",
    "ObjectStorageGeneratorConfig",
    "bootstrap_s3_checkpoint",
]
