# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background preparation of object-storage refits outside the serving pause."""

import dataclasses
import fcntl
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import safetensors.numpy
import torch
from safetensors.torch import load_file, save_file

from modelexpress_rl import (
    ObjectStorageGeneratorConfig,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightSource,
    WeightVersion,
    WeightVersionState,
    refit_pb2,
)
from modelexpress_rl.inference import client as client_module
from modelexpress_rl.inference import prefetch as prefetch_module
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.client import ModelExpressGeneratorClient
from modelexpress_rl.inference.methods import CanonicalDeltaUpdateMethod
from modelexpress_rl.inference.session import WeightUpdateSession
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.utils import checksum_factory, compress_delta, compute_delta

PREFIX = "s3://weights/hotload"
INDEX = "model.safetensors.index.json"
SHARD = "model-00000-of-00001.safetensors"


def _bytes(values):
    return torch.tensor(values).contiguous().reshape(-1).view(torch.uint8).numpy()


def _checksum(value):
    checksum = checksum_factory("adler32")
    checksum.update(value)
    return checksum.hexdigest()


def _uri(identity):
    return f"{PREFIX}/{identity}/{INDEX}"


def _delta_snapshot(base, target, *, identity, base_identity):
    delta, _ = compute_delta(_bytes(target), _bytes(base))
    shard = safetensors.numpy.save(
        {"weight": compress_delta(delta)},
        metadata={"weight": _checksum(_bytes(target))},
    )
    index = json.dumps(
        {
            "metadata": {
                "version": identity,
                "base_version": base_identity,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {"weight": SHARD},
        }
    ).encode()
    return {_uri(identity): index, f"{PREFIX}/{identity}/{SHARD}": shard}


class _FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.gets = []

    def get(self, uri):
        self.gets.append(uri)
        return self.objects[uri]

    def size(self, uri):
        return len(self.objects[uri])

    def exists(self, uri):
        return uri in self.objects

    def list_children(self, prefix_uri, *, start_after=None):
        prefix = prefix_uri.rstrip("/") + "/"
        names = {
            key[len(prefix) :].split("/", 1)[0]
            for key in self.objects
            if key.startswith(prefix) and "/" in key[len(prefix) :]
        }
        return sorted(name for name in names if start_after is None or name > start_after)

    def close(self):
        pass


class _FakeRefit:
    def __init__(self):
        self.versions = {}

    def add(self, uid, *, payload, uri, created, base=None):
        version = refit_pb2.WeightVersion(
            uid=uid,
            model_name="test/model",
            payload_format=payload,
            state=refit_pb2.WEIGHT_VERSION_STATE_READY,
            created_at_unix_ms=created,
            object_storage=refit_pb2.ObjectStorageSource(
                uri=uri,
                storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            ),
        )
        if base is not None:
            version.base_version_id = base
        self.versions[uid] = version

    def ListWeightVersions(self, request, timeout):
        assert request.model_name == "test/model"
        return refit_pb2.ListWeightVersionsResponse(versions=list(self.versions.values()))

    def GetWeightVersion(self, request, timeout):
        if request.uid not in self.versions:
            return refit_pb2.GetWeightVersionResponse()
        return refit_pb2.GetWeightVersionResponse(version=self.versions[request.uid])


FULL = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
XOR = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA


def _config(tmp_path):
    launch = tmp_path / "launch"
    if not launch.exists():
        launch.mkdir()
        save_file({"weight": torch.tensor([1.0, 2.0])}, launch / "model.safetensors")
    return ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base",
        seed_checkpoint_path=launch,
        refit_checkpoint_dir=tmp_path / "cache",
        refit_checkpoint_max_size_gb=1,
    )


def _checkpoint(tmp_path, s3, *, initialize=True):
    checkpoint = receiver_module._LocalCheckpoint(
        model_name="test/model",
        config=_config(tmp_path),
        s3=s3,
    )
    if initialize:
        checkpoint.initialize()
    return checkpoint


def _settings(tmp_path, *, s3_watch=False):
    return prefetch_module.PrefetchSettings(
        model_name="test/model",
        cache_model_name="test/model",
        server_url="mx:8001",
        refit_checkpoint_dir=str(tmp_path / "cache"),
        initial_base_version_id="base",
        seed_checkpoint_path=str(tmp_path / "launch"),
        refit_checkpoint_max_size_gb=1,
        endpoint_url=None,
        region_name=None,
        max_replay_chain_length=64,
        rpc_timeout_seconds=1.0,
        poll_seconds=0.01,
        s3_watch=s3_watch,
        parent_pid=os.getpid(),
    )


def _delta(version_id, *, base, identity):
    return receiver_module._S3Version(
        version_id=version_id,
        base_version_id=base,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        uri=_uri(identity),
    )


def _weight(prepared):
    return load_file(prepared.path / "model.safetensors")["weight"]


def _setup(tmp_path, objects, *, s3_watch=False):
    s3 = _FakeS3(objects)
    checkpoint = _checkpoint(tmp_path, s3)
    refit = _FakeRefit()
    refit.add("base", payload=FULL, uri=_uri("run-v000000"), created=1)
    prefetcher = prefetch_module.RefitPrefetcher(
        _settings(tmp_path, s3_watch=s3_watch),
        service=refit,
        s3=s3,
        checkpoint=checkpoint,
    )
    return s3, checkpoint, refit, prefetcher


def test_s3_client_lists_children_and_checks_existence():
    class _Paginator:
        def __init__(self):
            self.requests = []

        def paginate(self, **request):
            self.requests.append(request)
            return iter(
                [
                    {"CommonPrefixes": [{"Prefix": "hotload/run-v2/"}]},
                    {"CommonPrefixes": [{"Prefix": "hotload/run-v1/"}]},
                ]
            )

    class _Error(Exception):
        response = {"Error": {"Code": "404"}}

    class _Client:
        paginator = _Paginator()

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return self.paginator

        def head_object(self, *, Bucket, Key):
            if Key == "hotload/missing":
                raise _Error()
            return {"ContentLength": 1}

    s3 = object.__new__(S3Client)
    s3._client = _Client()

    assert s3.list_children("s3://weights/hotload", start_after="run-v0") == [
        "run-v1",
        "run-v2",
    ]
    assert s3._client.paginator.requests == [
        {
            "Bucket": "weights",
            "Prefix": "hotload/",
            "Delimiter": "/",
            "StartAfter": "hotload/run-v0",
        }
    ]
    assert s3.exists("s3://weights/hotload/present")
    assert not s3.exists("s3://weights/hotload/missing")


def test_store_rename_version_moves_artifacts_and_chain_references(tmp_path):
    checkpoint = _checkpoint(tmp_path, _FakeS3({}))
    store = checkpoint.store
    for path in (store.delta_path("spec"), store.materialized_path("spec")):
        path.mkdir(parents=True)
        (path / "file").write_text("payload")
    store.write_chain("spec", {"version": "spec", "full_version": "base", "deltas": ["spec"]})
    store.write_chain("next", {"version": "next", "full_version": "base", "deltas": ["spec", "next"]})

    store.rename_version("spec", "v1")

    assert (store.delta_path("v1") / "file").read_text() == "payload"
    assert (store.materialized_path("v1") / "file").read_text() == "payload"
    assert not store.delta_path("spec").exists()
    assert store.chain("v1") == {"version": "v1", "full_version": "base", "deltas": ["v1"]}
    assert store.chain("next")["deltas"] == ["v1", "next"]
    assert not store.chain_path("spec").exists()


def test_prefetcher_prepares_the_newest_registered_version(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="run-v000001", base_identity="run-v000000"
    )
    s3, checkpoint, refit, prefetcher = _setup(tmp_path, objects)
    refit.add("v1", payload=XOR, base="base", uri=_uri("run-v000001"), created=2)

    assert prefetcher.poll_once() == "v1"
    assert checkpoint.store.state().version == "v1"
    status = json.loads((checkpoint.store.cache / prefetch_module.STATUS_FILE).read_text())
    assert status["phase"] == "prepared" and status["version"] == "v1"

    # The engine process prepares the same target with its own cache view.
    downloads = len(s3.gets)
    worker = _checkpoint(tmp_path, s3, initialize=False)
    prepared = worker.prepare_chain((_delta("v1", base="base", identity="run-v000001"),))
    assert prepared.target_version == "v1"
    assert torch.equal(_weight(prepared), torch.tensor([1.0, 3.0]))
    assert len(s3.gets) == downloads
    assert prefetcher.poll_once() is None


def test_speculative_prefetch_is_adopted_when_the_version_registers(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="run-v000001", base_identity="run-v000000"
    )
    s3, checkpoint, refit, prefetcher = _setup(tmp_path, objects, s3_watch=True)
    speculative = receiver_module.speculative_version_id(_uri("run-v000001"))

    assert prefetcher.poll_once() == speculative
    assert checkpoint.store.state().version == speculative
    assert checkpoint.is_prepared(version_id="v1", uri=_uri("run-v000001"))

    downloads = len(s3.gets)
    refit.add("v1", payload=XOR, base="base", uri=_uri("run-v000001"), created=2)
    worker = _checkpoint(tmp_path, s3, initialize=False)
    prepared = worker.prepare_chain((_delta("v1", base="base", identity="run-v000001"),))

    assert prepared.target_version == "v1"
    assert torch.equal(_weight(prepared), torch.tensor([1.0, 3.0]))
    assert len(s3.gets) == downloads
    assert checkpoint.store.speculation() is None
    assert not checkpoint.store.chain_path(speculative).exists()
    assert prefetcher.poll_once() is None


def test_prefetcher_adopts_its_speculation_on_registration(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="run-v000001", base_identity="run-v000000"
    )
    s3, checkpoint, refit, prefetcher = _setup(tmp_path, objects, s3_watch=True)
    prefetcher.poll_once()
    downloads = len(s3.gets)

    refit.add("v1", payload=XOR, base="base", uri=_uri("run-v000001"), created=2)

    assert prefetcher.poll_once() == "v1"
    assert checkpoint.store.state().version == "v1"
    assert len(s3.gets) == downloads


def test_speculation_waits_for_every_shard(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="run-v000001", base_identity="run-v000000"
    )
    del objects[f"{PREFIX}/run-v000001/{SHARD}"]
    _s3, checkpoint, _refit, prefetcher = _setup(tmp_path, objects, s3_watch=True)

    assert prefetcher.poll_once() is None
    assert checkpoint.store.state().version == "base"
    assert checkpoint.store.speculation() is None


def test_speculation_ignores_snapshots_of_other_bases(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="run-v000002", base_identity="run-v000001"
    )
    _s3, checkpoint, _refit, prefetcher = _setup(tmp_path, objects, s3_watch=True)

    assert prefetcher.poll_once() is None
    assert checkpoint.store.state().version == "base"


def test_prefetcher_keeps_a_speculative_head_ahead_of_its_base(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="run-v000001", base_identity="run-v000000"
    )
    _s3, checkpoint, _refit, prefetcher = _setup(tmp_path, objects, s3_watch=True)
    speculative = prefetcher.poll_once()

    # The newest registered version is still the base, and speculation is depth one.
    assert prefetcher.poll_once() is None
    assert checkpoint.store.state().version == speculative


def test_unwanted_speculation_is_reverted_before_another_version(tmp_path):
    objects = {
        **_delta_snapshot(
            [1.0, 2.0], [1.0, 3.0], identity="run-v000001", base_identity="run-v000000"
        ),
        **_delta_snapshot(
            [1.0, 3.0], [4.0, 3.0], identity="run-v000002", base_identity="run-v000001"
        ),
        **_delta_snapshot(
            [1.0, 3.0], [1.0, 9.0], identity="run-v000002b", base_identity="run-v000001"
        ),
    }
    s3, checkpoint, refit, prefetcher = _setup(tmp_path, objects)
    refit.add("v1", payload=XOR, base="base", uri=_uri("run-v000001"), created=2)
    assert prefetcher.poll_once() == "v1"

    speculative = receiver_module.speculative_version_id(_uri("run-v000002"))
    checkpoint.prepare_speculative(
        receiver_module._S3Version(
            version_id=speculative,
            base_version_id="v1",
            payload_format=WeightPayloadFormat.XOR_DELTA,
            uri=_uri("run-v000002"),
        )
    )
    assert checkpoint.store.state().version == speculative

    worker = _checkpoint(tmp_path, s3, initialize=False)
    prepared = worker.prepare_chain((_delta("v2b", base="v1", identity="run-v000002b"),))

    # Reapplying the speculative XOR delta restored v1 exactly before v2b applied.
    assert prepared.target_version == "v2b"
    assert torch.equal(_weight(prepared), torch.tensor([1.0, 9.0]))
    assert checkpoint.store.speculation() is None
    assert not checkpoint.store.delta_path(speculative).exists()


def _version(version_id):
    return WeightVersion(
        version_id=version_id,
        model_name="test/model",
        payload_format=WeightPayloadFormat.XOR_DELTA,
        layout_signature="",
        state=WeightVersionState.READY,
        created_at_unix_ms=0,
        base_version_id="base",
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri=_uri("run-v000001"),
        ),
    )


@pytest.mark.parametrize(
    "prepared, expected",
    [(True, [WeightSource.OBJECT_STORAGE]), (False, [WeightSource.GENERATOR])],
)
def test_session_tries_a_prepared_target_from_object_storage_first(
    monkeypatch, prepared, expected
):
    session = WeightUpdateSession(
        planner=SimpleNamespace(
            source_order=(WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE)
        ),
        start_lease=lambda _version_id: None,
        prefer_prepared=lambda _version: prepared,
    )
    tried = []
    monkeypatch.setattr(
        session,
        "_stage_source",
        lambda version, *, source_kind: tried.append(source_kind) or "update",
    )

    assert session.stage(_version("v1")) == "update"
    assert tried == expected


def test_generator_client_starts_and_stops_the_prefetcher(monkeypatch, tmp_path):
    events = []

    class _Process:
        pid = 1234

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(
        client_module,
        "launch_prefetcher",
        lambda settings, *, workers: events.append((settings, workers)) or _Process(),
    )
    monkeypatch.setenv("MX_REFIT_PREFETCH_S3_WATCH", "1")
    monkeypatch.setenv("MX_REFIT_PREFETCH_WORKERS", "3")
    canonical = CanonicalDeltaUpdateMethod.__new__(CanonicalDeltaUpdateMethod)
    canonical._model_name = "cache/model"
    client = ModelExpressGeneratorClient()
    client.model_name = "registry/model"
    client.server_url = "mx:8001"
    client._max_replay_chain_length = 64
    client._rpc_timeout_seconds = 5.0
    client._runtime = SimpleNamespace(methods=(canonical,))

    client._start_prefetcher(SimpleNamespace(object_storage=_config(tmp_path)))

    settings, workers = events[0]
    assert workers == 3
    assert settings.model_name == "registry/model"
    assert settings.cache_model_name == "cache/model"
    assert settings.s3_watch is True
    assert settings.parent_pid == os.getpid()
    # The object-storage method initializes the cache before the client starts.
    settings.cache_dir().mkdir(parents=True)
    client.notify_prefetch()
    assert (settings.cache_dir() / prefetch_module.TRIGGER_FILE).exists()
    client._stop_prefetcher()
    assert events[-1] == "terminate"


def test_prefetcher_process_exits_when_another_serves_the_cache(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    cache = settings.cache_dir()
    cache.mkdir(parents=True)
    connect = Mock()
    monkeypatch.setattr(prefetch_module.RefitPrefetcher, "connect", connect)

    with (cache / prefetch_module.LOCK_FILE).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert (
            prefetch_module.main(["--settings", json.dumps(dataclasses.asdict(settings))])
            == 0
        )

    connect.assert_not_called()
