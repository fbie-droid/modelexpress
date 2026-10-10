# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare object-storage refits in the background, outside the serving pause.

One prefetcher process runs per host checkpoint cache, next to the inference
engine. It watches the ModelExpress registry for READY versions and, optionally,
the S3 snapshot prefix for deltas that are not registered yet. It prepares the
newest version into the host-local cache through the same locked preparation the
engine's update uses, so a later update finds its target prepared and only
installs it.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import posixpath
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import grpc
from modelexpress import auth

from .. import refit_pb2, refit_pb2_grpc
from ..control import WeightVersion, WeightVersionState, _weight_version
from ..object_storage import ObjectStorageType
from ..s3 import S3Client
from ..train import WeightPayloadFormat
from .checkpoint_store import CheckpointState, LocalCheckpointStore
from .receiver import (
    ObjectStorageGeneratorConfig,
    PreparedCheckpoint,
    _LocalCheckpoint,
    _S3Version,
    speculative_version_id,
)
from .version_chain import resolve_replay_chain

logger = logging.getLogger("modelexpress_rl.inference.prefetch")

STATUS_FILE = "prefetch.json"
TRIGGER_FILE = "prefetch.trigger"
LOCK_FILE = ".prefetch.lock"

_REPLAY_FORMATS = frozenset(
    {WeightPayloadFormat.XOR_DELTA, WeightPayloadFormat.FULL_HF_CHECKPOINT}
)


@dataclass(frozen=True)
class PrefetchSettings:
    """Prefetcher configuration, passed to its process as JSON."""

    model_name: str
    cache_model_name: str
    server_url: str
    refit_checkpoint_dir: str
    initial_base_version_id: str
    seed_checkpoint_path: str | None
    refit_checkpoint_max_size_gb: int | None
    endpoint_url: str | None
    region_name: str | None
    max_replay_chain_length: int
    rpc_timeout_seconds: float
    poll_seconds: float
    s3_watch: bool
    parent_pid: int

    def object_storage(self) -> ObjectStorageGeneratorConfig:
        return ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id=self.initial_base_version_id,
            seed_checkpoint_path=self.seed_checkpoint_path,
            refit_checkpoint_dir=self.refit_checkpoint_dir,
            refit_checkpoint_max_size_gb=self.refit_checkpoint_max_size_gb,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
        )

    def cache_dir(self) -> Path:
        return LocalCheckpointStore(
            root=self.refit_checkpoint_dir,
            model_name=self.cache_model_name,
        ).cache


def launch_prefetcher(settings: PrefetchSettings, *, workers: int) -> subprocess.Popen:
    """Start the prefetcher process for one host checkpoint cache."""
    environment = dict(os.environ, MX_REFIT_DELTA_WORKERS=str(workers))
    return subprocess.Popen(
        [sys.executable, "-m", __name__, "--settings", json.dumps(asdict(settings))],
        env=environment,
    )


def notify_prefetcher(cache_dir: Path) -> None:
    """Ask a running prefetcher to poll now instead of at its next interval."""
    try:
        (cache_dir / TRIGGER_FILE).touch()
    except OSError:
        logger.debug("prefetch trigger is unavailable", exc_info=True)


class RefitPrefetcher:
    """Advance the host checkpoint cache to the newest version it can reach."""

    def __init__(
        self,
        settings: PrefetchSettings,
        *,
        service: Any,
        s3: S3Client,
        checkpoint: _LocalCheckpoint,
    ) -> None:
        self._settings = settings
        self._service = service
        self._s3 = s3
        self._checkpoint = checkpoint
        self._store = checkpoint.store
        self._channel: grpc.Channel | None = None
        # Servers that predate version listing still support the S3 watch.
        self._registry_listing = True

    @classmethod
    def connect(cls, settings: PrefetchSettings) -> RefitPrefetcher:
        channel = auth.with_auth(grpc.insecure_channel(settings.server_url))
        s3 = S3Client(
            endpoint_url=settings.endpoint_url,
            region_name=settings.region_name,
        )
        checkpoint = _LocalCheckpoint(
            model_name=settings.cache_model_name,
            config=settings.object_storage(),
            s3=s3,
        )
        prefetcher = cls(
            settings,
            service=refit_pb2_grpc.RefitServiceStub(channel),
            s3=s3,
            checkpoint=checkpoint,
        )
        prefetcher._channel = channel
        return prefetcher

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._s3.close()

    def poll_once(self) -> str | None:
        """Prepare at most one version; return the cache ID it prepared."""
        state = self._store.state()
        if state is None or state.status is not CheckpointState.READY:
            return None
        target = self._newest_registered() if self._registry_listing else None
        if target is not None and self._should_prepare(target, head=state.version):
            chain = self._registered_chain(target, head=state.version)
            if chain is not None:
                return self._prepare(
                    target.version_id,
                    kind="registered",
                    run=lambda: self._checkpoint.prepare_chain(chain),
                )
        if self._settings.s3_watch:
            return self._speculate(head=state.version, head_source=state.source)
        return None

    def run(self) -> None:
        """Poll until the engine process that started this prefetcher exits."""
        trigger = self._trigger_mtime()
        while self._parent_alive():
            try:
                self.poll_once()
            except grpc.RpcError as error:
                logger.warning("prefetch poll failed: %s", error)
            except Exception:
                logger.exception("prefetch poll failed")
            deadline = time.monotonic() + self._settings.poll_seconds
            while time.monotonic() < deadline and self._parent_alive():
                current = self._trigger_mtime()
                if current != trigger:
                    trigger = current
                    break
                time.sleep(min(0.2, self._settings.poll_seconds))

    def _parent_alive(self) -> bool:
        return os.getppid() == self._settings.parent_pid

    def _trigger_mtime(self) -> int | None:
        try:
            return (self._store.cache / TRIGGER_FILE).stat().st_mtime_ns
        except OSError:
            return None

    def _should_prepare(self, target: WeightVersion, *, head: str) -> bool:
        if target.version_id == head:
            return False
        speculation = self._store.speculation()
        # A speculative head is ahead of its base; preparing the base would undo it.
        return not (
            speculation is not None
            and speculation.get("version") == head
            and speculation.get("base_version") == target.version_id
        )

    def _newest_registered(self) -> WeightVersion | None:
        try:
            response = self._service.ListWeightVersions(
                refit_pb2.ListWeightVersionsRequest(model_name=self._settings.model_name),
                timeout=self._settings.rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            if error.code() is grpc.StatusCode.UNIMPLEMENTED:
                self._registry_listing = False
                logger.info("MX server cannot list versions; prefetch watches S3 only")
            else:
                logger.warning("prefetch could not list versions: %s", error)
            return None
        candidates = []
        for message in response.versions:
            try:
                version = _weight_version(message)
            except RuntimeError:
                continue
            if (
                version.state is WeightVersionState.READY
                and version.payload_format in _REPLAY_FORMATS
                and version.object_storage is not None
                and version.object_storage.storage_type is ObjectStorageType.S3
            ):
                candidates.append(version)
        return max(candidates, key=lambda version: version.created_at_unix_ms, default=None)

    def _fetch_ready_version(self, version_id: str) -> WeightVersion:
        try:
            response = self._service.GetWeightVersion(
                refit_pb2.GetWeightVersionRequest(uid=version_id),
                timeout=self._settings.rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            raise RuntimeError(f"failed to resolve revision {version_id!r}") from error
        if not response.HasField("version"):
            raise RuntimeError(f"MX returned no revision {version_id!r}")
        version = _weight_version(response.version)
        if (
            version.state is not WeightVersionState.READY
            or version.model_name != self._settings.model_name
        ):
            raise RuntimeError(f"revision {version_id!r} is not a READY model version")
        return version

    def _registered_chain(
        self, target: WeightVersion, *, head: str
    ) -> tuple[_S3Version, ...] | None:
        try:
            chain = resolve_replay_chain(
                target_version_id=target.version_id,
                fetch_ready_version=self._fetch_ready_version,
                max_chain_length=self._settings.max_replay_chain_length,
                stop_before_version_id=head,
            )
        except RuntimeError as error:
            logger.info("prefetch cannot resolve %s: %s", target.version_id, error)
            return None
        replay = []
        for version in chain:
            storage = version.object_storage
            if storage is None or storage.storage_type is not ObjectStorageType.S3:
                return None
            replay.append(
                _S3Version(
                    version_id=version.version_id,
                    base_version_id=version.base_version_id,
                    payload_format=version.payload_format,
                    uri=storage.uri,
                )
            )
        return tuple(replay)

    def _speculate(
        self, *, head: str, head_source: dict[str, str] | None
    ) -> str | None:
        """Prepare the unregistered delta whose snapshot names the head as base."""
        speculation = self._store.speculation()
        if speculation is not None and speculation.get("version") == head:
            return None
        head_uri = (head_source or {}).get("uri") or self._registered_uri(head)
        if head_uri is None:
            return None
        # Snapshots live at <prefix>/<identity>/<index>; deltas name their base identity.
        snapshot = posixpath.dirname(head_uri)
        identity = posixpath.basename(snapshot)
        prefix = posixpath.dirname(snapshot)
        index_name = posixpath.basename(head_uri)
        if not identity or not prefix.startswith("s3://"):
            return None
        for child in self._s3.list_children(prefix, start_after=identity):
            if child <= identity or child.startswith(("_", ".")):
                continue
            uri = f"{prefix}/{child}/{index_name}"
            if not self._is_complete_child_delta(uri, base_identity=identity):
                continue
            version = _S3Version(
                version_id=speculative_version_id(uri),
                base_version_id=head,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                uri=uri,
            )
            return self._prepare(
                version.version_id,
                kind="speculative",
                run=lambda: self._checkpoint.prepare_speculative(version),
                uri=uri,
            )
        return None

    def _registered_uri(self, version_id: str) -> str | None:
        try:
            version = self._fetch_ready_version(version_id)
        except RuntimeError:
            return None
        if version.object_storage is None:
            return None
        return version.object_storage.uri

    def _is_complete_child_delta(self, uri: str, *, base_identity: str) -> bool:
        try:
            index = json.loads(self._s3.get(uri))
        except Exception:
            return False
        metadata = index.get("metadata") if isinstance(index, dict) else None
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(weight_map, dict)
            or metadata.get("delta_encoding") != "xor"
            or metadata.get("base_version") != base_identity
        ):
            return False
        # Writers upload the index last, but every shard must exist before reading.
        snapshot = posixpath.dirname(uri)
        return all(
            self._s3.exists(f"{snapshot}/{shard}") for shard in set(weight_map.values())
        )

    def _prepare(
        self,
        version_id: str,
        *,
        kind: str,
        run: Any,
        uri: str | None = None,
    ) -> str | None:
        started = time.monotonic()
        self._write_status(phase="preparing", version=version_id, kind=kind, uri=uri)
        logger.info("prefetch preparing %s version %s", kind, version_id)
        try:
            prepared: PreparedCheckpoint = run()
        except Exception as error:
            logger.warning("prefetch of %s failed: %s", version_id, error)
            self._write_status(
                phase="failed", version=version_id, kind=kind, uri=uri, error=str(error)
            )
            try:
                self._checkpoint.recover_incomplete_preparation()
            except Exception:
                logger.exception("prefetch recovery after a failed preparation failed")
            return None
        duration = time.monotonic() - started
        logger.info(
            "prefetch prepared %s version %s in %.2fs", kind, prepared.target_version, duration
        )
        self._write_status(
            phase="prepared",
            version=prepared.target_version,
            kind=kind,
            uri=uri,
            duration_seconds=duration,
            metrics=prepared.metrics,
        )
        return prepared.target_version

    def _write_status(self, **fields: Any) -> None:
        status = dict(fields, pid=os.getpid(), updated_at_unix_ms=int(time.time() * 1000))
        path = self._store.cache / STATUS_FILE
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(status, sort_keys=True))
            temporary.replace(path)
        except OSError:
            logger.debug("prefetch status is unavailable", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", required=True, help="JSON PrefetchSettings")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = PrefetchSettings(**json.loads(args.settings))
    cache_dir = settings.cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / LOCK_FILE).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("another prefetcher already serves %s", cache_dir)
            return 0
        try:
            os.nice(10)
        except OSError:
            pass
        prefetcher = RefitPrefetcher.connect(settings)
        try:
            prefetcher.run()
        finally:
            prefetcher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
