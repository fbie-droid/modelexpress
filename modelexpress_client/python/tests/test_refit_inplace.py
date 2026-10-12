# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staging of nonzero XOR bytes and in-place installation of deltas."""

import dataclasses
import functools
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import safetensors.numpy
import torch
from safetensors.torch import load_file, save_file

from modelexpress_rl import ObjectStorageGeneratorConfig, ObjectStorageType, WeightPayloadFormat
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.checkpoint_store import CheckpointState, LocalCheckpointStore
from modelexpress_rl.inference.engines.vllm import inplace as inplace_module
from modelexpress_rl.inference.engines.vllm import installer as installer_module
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from modelexpress_rl.inference.staging import (
    POSITIONS_FILE,
    VALUES_FILE,
    StagedDelta,
    join_nonzero,
    nonzero_bytes,
    write_staging,
)
from modelexpress_rl.utils import (
    checksum_factory,
    compress_delta,
    compute_delta,
    index_checkpoint_tensors,
)

PREFIX = "s3://weights/hotload"
INDEX = "model.safetensors.index.json"
SHARD = "model-00000-of-00001.safetensors"


def _stage(directory, tensors, base="base"):
    """Stage {name: (base bytes, target bytes)} as their XOR."""
    staged = {}
    for name, (before, after) in tensors.items():
        xor = np.bitwise_xor(before, after)
        staged[name] = (xor.size, *join_nonzero(xor.size, [nonzero_bytes(xor, 0)]))
    directory.mkdir(parents=True, exist_ok=True)
    write_staging(directory, base_version=base, tensors=staged)
    return StagedDelta(directory)


def test_staged_delta_round_trips_nonzero_bytes(tmp_path):
    base = np.zeros(10, dtype=np.uint8)
    target = base.copy()
    target[[1, 7]] = [5, 9]
    staged = _stage(tmp_path / "staging", {"a": (base, target), "b": (base, base)})

    assert StagedDelta.base_of(tmp_path / "staging") == "base"
    assert "a" in staged and "missing" not in staged
    tensor = staged.tensor("a")
    assert tensor.nbytes == 10
    assert tensor.positions.tolist() == [1, 7]
    assert tensor.values.tolist() == [5, 9]
    assert staged.tensor("b").positions.size == 0
    staged.close()


@pytest.mark.parametrize("density", [0.0, 0.01, 0.3, 1.0])
@pytest.mark.parametrize("size", [0, 7, 8, 17, 4097])
def test_nonzero_bytes_match_a_byte_scan_at_any_alignment(size, density):
    rng = np.random.default_rng(size)
    block = np.where(rng.random(size) < density, rng.integers(1, 256, size), 0).astype(np.uint8)
    # Start one byte into a buffer, so 8-byte words are unaligned.
    shifted = np.frombuffer(b"\0" + block.tobytes(), dtype=np.uint8)[1:]

    positions, values = nonzero_bytes(shifted, 11)

    assert positions.tolist() == (np.flatnonzero(block) + 11).tolist()
    assert values.tolist() == block[block != 0].tolist()


def test_staged_base_is_read_without_the_index(tmp_path):
    staged = _stage(
        tmp_path / "staging",
        {"a": (np.zeros(4, dtype=np.uint8), np.arange(4, dtype=np.uint8))},
        base="v7",
    )
    staged.close()
    (tmp_path / "staging" / "index.json").unlink()

    assert StagedDelta.base_of(tmp_path / "staging") == "v7"


def test_staging_rejects_offsets_of_the_wrong_width(tmp_path):
    tensors = {"a": (4, np.arange(2, dtype=np.uint64), np.ones(2, dtype=np.uint8))}

    with pytest.raises(ValueError, match="offsets"):
        write_staging(tmp_path, base_version="base", tensors=tensors)


def test_checkpoint_index_reuses_unchanged_headers(tmp_path):
    first, second = tmp_path / "v1", tmp_path / "v2"
    for directory in (first, second):
        directory.mkdir()
        save_file({"a": torch.zeros(2), "lm_head.weight": torch.ones(3)}, directory / "model.safetensors")

    _, locations, metadata = index_checkpoint_tensors(first)
    _, again, same = index_checkpoint_tensors(first)
    _, moved, reused = index_checkpoint_tensors(second)

    assert same is metadata and again is locations
    # A new version directory with the same headers keeps the parsed metadata.
    assert reused is metadata
    assert moved["a"] == (second / "model.safetensors", *locations["a"][1:])

    (second / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    _, _, tied = index_checkpoint_tensors(second)
    assert "lm_head.weight" not in tied and "lm_head.weight" in metadata

    save_file({"a": torch.zeros(4)}, second / "model.safetensors")
    _, changed, fresh = index_checkpoint_tensors(second)
    assert fresh is not metadata and fresh["a"]["shape"] == [4]
    assert list(changed) == ["a"]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
        ),
    ],
)
def test_staged_bytes_rebuild_the_dense_delta(tmp_path, device):
    after = np.random.default_rng(1).integers(0, 256, 64, dtype=np.uint8)
    after[::3] = 0
    staged = _stage(tmp_path / "staging", {"a": (np.zeros(64, dtype=np.uint8), after)})

    with inplace_module._quiet_readonly_buffers():
        data = inplace_module.staged_bytes(64, staged.tensor("a"), torch.device(device), chunk=5)

    assert data.device.type == device
    assert data.cpu().numpy().tolist() == after.tolist()
    assert not inplace_module.staged_bytes(8, None, torch.device(device)).any()
    staged.close()


def test_large_tensors_stage_wide_offsets():
    small, _ = join_nonzero(16, [(np.array([3]), np.array([1], dtype=np.uint8))])
    large, _ = join_nonzero(1 << 33, [(np.array([1 << 32]), np.array([1], dtype=np.uint8))])

    assert small.dtype == np.uint32
    assert large.dtype == np.uint64 and large.tolist() == [1 << 32]


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

    def get(self, uri):
        return self.objects[uri]

    def close(self):
        pass


def _checkpoint(tmp_path, s3, *, initialize=True, tensors=None):
    launch = tmp_path / "launch"
    if not launch.exists():
        launch.mkdir()
        tensors = tensors or {"weight": [1.0, 2.0]}
        save_file(
            {name: torch.tensor(values) for name, values in tensors.items()},
            launch / "model.safetensors",
        )
    checkpoint = receiver_module._LocalCheckpoint(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base",
            seed_checkpoint_path=launch,
            refit_checkpoint_dir=tmp_path / "cache",
            refit_checkpoint_max_size_gb=1,
        ),
        s3=s3,
    )
    if initialize:
        checkpoint.initialize()
    return checkpoint


def _delta(version_id, *, base, identity):
    return receiver_module._S3Version(
        version_id=version_id,
        base_version_id=base,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        uri=_uri(identity),
    )


@pytest.mark.parametrize("flag", [None, "false"])
def test_delta_preparation_stages_nothing_by_default(monkeypatch, tmp_path, flag):
    if flag is None:
        monkeypatch.delenv("MX_REFIT_INPLACE", raising=False)
    else:
        monkeypatch.setenv("MX_REFIT_INPLACE", flag)
    s3 = _FakeS3(_delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base"))
    checkpoint = _checkpoint(tmp_path, s3)

    prepared = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))

    assert prepared.staging is None and prepared.staging_base is None
    assert not checkpoint.store.staging_path("v1").exists()


def test_delta_preparation_stages_the_changed_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_REFIT_INPLACE", "true")
    s3 = _FakeS3(_delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base"))
    checkpoint = _checkpoint(tmp_path, s3)

    prepared = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))

    assert prepared.staging == checkpoint.store.staging_path("v1")
    assert prepared.staging_base == "base"
    staged = StagedDelta(prepared.staging)
    xor = np.bitwise_xor(_bytes([1.0, 2.0]), _bytes([1.0, 3.0]))
    assert staged.tensor("weight").positions.tolist() == np.flatnonzero(xor).tolist()
    assert staged.tensor("weight").values.tolist() == xor[xor != 0].tolist()
    staged.close()

    # A co-located rank reusing the prepared head sees the same staging.
    reused = _checkpoint(tmp_path, s3, initialize=False).prepare_chain(
        (_delta("v1", base="base", identity="v1"),)
    )
    assert reused.staging == prepared.staging and reused.staging_base == "base"


def test_staging_follows_speculation_adoption_and_revert(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_REFIT_INPLACE", "true")
    objects = {
        **_delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base"),
        **_delta_snapshot([1.0, 3.0], [4.0, 3.0], identity="x", base_identity="v1"),
        **_delta_snapshot([1.0, 3.0], [1.0, 9.0], identity="y", base_identity="v1"),
    }
    checkpoint = _checkpoint(tmp_path, _FakeS3(objects))
    store = checkpoint.store
    speculative = receiver_module.speculative_version_id(_uri("v1"))
    checkpoint.prepare_speculative(_delta(speculative, base="base", identity="v1"))
    assert store.staging_path(speculative).exists()

    adopted = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))
    assert adopted.staging == store.staging_path("v1")
    assert not store.staging_path(speculative).exists()

    unwanted = receiver_module.speculative_version_id(_uri("x"))
    checkpoint.prepare_speculative(_delta(unwanted, base="v1", identity="x"))
    prepared = checkpoint.prepare_chain((_delta("y", base="v1", identity="y"),))

    assert prepared.staging == store.staging_path("y") and prepared.staging_base == "v1"
    assert not store.staging_path(unwanted).exists()


def test_evicting_a_version_removes_its_staging(monkeypatch, tmp_path):
    monkeypatch.setenv("MX_REFIT_INPLACE", "true")
    s3 = _FakeS3(_delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base"))
    checkpoint = _checkpoint(tmp_path, s3)
    checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))
    store = checkpoint.store
    store.activate("base")

    store.ensure_capacity(0, protected_versions=())
    assert store.staging_path("v1").exists()
    store.max_size_bytes = 0
    with pytest.raises(receiver_module.CheckpointCacheCapacityError):
        store.ensure_capacity(1, protected_versions=())
    assert not store.staging_path("v1").exists()


# Each delta changes some tensors of a two-tensor checkpoint.
_LINEAGE = {
    "base": {"a": [1.0, 2.0], "b": [3.0, 4.0]},
    "v1": {"a": [1.0, 5.0], "b": [3.0, 4.0]},
    "v2": {"a": [9.0, 5.0], "b": [6.0, 4.0]},
    "v3": {"a": [7.0, 5.0], "b": [6.0, 4.0]},
    "other": {"a": [8.0, 5.0], "b": [3.0, 4.0]},
}


def _lineage_snapshot(version, base):
    """The delta from ``base`` to ``version`` of every tensor that changes."""
    tensors, checksums = {}, {}
    for name, values in _LINEAGE[version].items():
        delta, changed = compute_delta(_bytes(values), _bytes(_LINEAGE[base][name]))
        if changed:
            tensors[name] = compress_delta(delta)
            checksums[name] = _checksum(_bytes(values))
    index = json.dumps(
        {
            "metadata": {
                "version": version,
                "base_version": base,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {name: SHARD for name in tensors},
        }
    ).encode()
    shard = safetensors.numpy.save(tensors, metadata=checksums)
    return {_uri(version): index, f"{PREFIX}/{version}/{SHARD}": shard}


def _lineage_checkpoint(tmp_path, monkeypatch, *, layers=None, clones=True):
    monkeypatch.setenv("MX_REFIT_INPLACE", "true")
    if layers is not None:
        monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", str(layers))
    # Rewrites into a copy need reflinks, which the test filesystem may lack.
    monkeypatch.setattr(LocalCheckpointStore, "clones_files", lambda self: clones)
    objects = {}
    for version, base in (("v1", "base"), ("v2", "v1"), ("v3", "v2")):
        objects.update(_lineage_snapshot(version, base))
    s3 = _FakeS3(objects)
    return _checkpoint(tmp_path, s3, tensors=_LINEAGE["base"]), s3


def _lineage_delta(version):
    base = {"v1": "base", "v2": "v1", "v3": "v2"}[version]
    return _delta(version, base=base, identity=version)


def _files(checkpoint):
    """The tensors the cache head's checkpoint files hold."""
    state = checkpoint.store.state()
    path = checkpoint.store.checkpoint_path(state.files_version) / "model.safetensors"
    return {name: tensor.tolist() for name, tensor in load_file(path).items()}


def _staged_target(prepared):
    """The target a preparation describes: its files XOR each staged delta since."""
    tensors = load_file(prepared.path / "model.safetensors")
    for staging in prepared.overlay:
        staged = StagedDelta(staging)
        for name in list(tensors):
            if name in staged:
                data = tensors[name].reshape(-1).view(torch.uint8)
                delta = staged.tensor(name)
                data[torch.from_numpy(delta.positions.astype(np.int64))] ^= torch.from_numpy(
                    np.array(delta.values)
                )
        staged.close()
    return {name: tensor.tolist() for name, tensor in tensors.items()}


def test_staged_preparation_checks_a_delta_without_writing_the_files(monkeypatch, tmp_path):
    checkpoint, s3 = _lineage_checkpoint(tmp_path, monkeypatch)

    prepared = checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True)

    state = checkpoint.store.state()
    assert (state.status, state.version, state.base) == (CheckpointState.STAGED, "v1", "base")
    assert not prepared.materialized and prepared.staging_base == "base"
    assert prepared.overlay == (checkpoint.store.staging_path("v1"),)
    assert prepared.path == checkpoint.store.full_path("base")
    assert _files(checkpoint) == _LINEAGE["base"]
    assert _staged_target(prepared) == _LINEAGE["v1"]
    # A co-located rank reuses the staged head; a caller that needs files gets them.
    follower = _checkpoint(tmp_path, s3, initialize=False)
    assert not follower.prepare_chain((_lineage_delta("v1"),), staged=True).materialized
    materialized = follower.prepare_chain((_lineage_delta("v1"),))
    assert materialized.materialized and checkpoint.store.state().status is CheckpointState.READY
    assert _files(checkpoint) == _LINEAGE["v1"]


def test_a_delta_with_a_wrong_checksum_leaves_a_ready_head_ready(monkeypatch, tmp_path):
    checkpoint, s3 = _lineage_checkpoint(tmp_path, monkeypatch)
    snapshot = _lineage_snapshot("v1", "base")
    shard = safetensors.numpy.load(snapshot[f"{PREFIX}/v1/{SHARD}"])
    s3.objects[f"{PREFIX}/v1/{SHARD}"] = safetensors.numpy.save(shard, metadata={"a": "00000001"})

    with pytest.raises(RuntimeError, match="checksum differs"):
        checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True)

    state = checkpoint.store.state()
    assert (state.status, state.version) == (CheckpointState.READY, "base")
    assert _files(checkpoint) == _LINEAGE["base"]


def test_a_delta_on_a_staged_head_is_checked_through_the_staged_deltas(monkeypatch, tmp_path):
    checkpoint, s3 = _lineage_checkpoint(tmp_path, monkeypatch)
    store = checkpoint.store
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))

    prepared = checkpoint.prepare_chain((_lineage_delta("v3"),), staged=True)

    state = store.state()
    assert (state.status, state.version, state.base) == (CheckpointState.STAGED, "v3", "base")
    assert prepared.overlay == tuple(store.staging_path(v) for v in ("v1", "v2", "v3"))
    assert (prepared.overlay_base, prepared.staging_base) == ("base", "v2")
    assert prepared.metrics["perf/mx_receive_staged_layers"] == 3.0
    assert _files(checkpoint) == _LINEAGE["base"]
    assert _staged_target(prepared) == _LINEAGE["v3"]
    follower = _checkpoint(tmp_path, s3, initialize=False)
    assert follower.prepare_chain((_lineage_delta("v3"),), staged=True).overlay == prepared.overlay


def test_at_the_layer_limit_a_delta_writes_the_staged_deltas_in_its_pass(monkeypatch, tmp_path):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch, layers=2)
    store = checkpoint.store
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))

    prepared = checkpoint.prepare_chain((_lineage_delta("v3"),), staged=True)

    state = store.state()
    assert (state.status, state.version, state.base) == (CheckpointState.STAGED, "v3", "v2")
    assert prepared.path == store.materialized_path("v2")
    assert prepared.overlay == (store.staging_path("v3"),)
    assert _files(checkpoint) == _LINEAGE["v2"]
    assert _staged_target(prepared) == _LINEAGE["v3"]
    # Below the limit again, a rewrite goes into a copy that replaces the files.
    checkpoint.activate(prepared)
    rewritten = checkpoint.materialize_head(staged=True)
    assert rewritten.target_version == "v3" and _files(checkpoint) == _LINEAGE["v3"]
    assert not store.materialized_path("v2").exists()


def test_a_staged_head_is_rewritten_once_the_engine_serves_it(monkeypatch, tmp_path):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch)
    store = checkpoint.store
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))

    rewritten = checkpoint.materialize_head(staged=True)

    assert rewritten.materialized and rewritten.target_version == "v2"
    assert "perf/mx_receive_delta_rewrite" in rewritten.metrics
    state = store.state()
    assert (state.status, state.version) == (CheckpointState.READY, "v2")
    assert _files(checkpoint) == _LINEAGE["v2"]
    base = load_file(store.full_path("base") / "model.safetensors")
    assert {name: tensor.tolist() for name, tensor in base.items()} == _LINEAGE["base"]
    assert [path.name for path in store.materialized_cache.iterdir()] == ["v2"]
    assert checkpoint.materialize_head(staged=True) is None
    # The next delta then only reads the rewritten files.
    following = checkpoint.prepare_chain((_lineage_delta("v3"),), staged=True)
    assert following.overlay == (store.staging_path("v3"),) and following.overlay_base == "v2"
    assert _staged_target(following) == _LINEAGE["v3"]


def test_a_following_delta_abandons_a_rewrite_into_a_copy(monkeypatch, tmp_path):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch)
    store = checkpoint.store
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))
    answers = iter([None, (_lineage_delta("v3"), False)])

    prepared = checkpoint.materialize_head(staged=True, following=lambda: next(answers, None))

    # The copy had "a" rewritten when v3 appeared, and is discarded instead.
    state = store.state()
    assert (state.status, state.version, state.base) == (CheckpointState.STAGED, "v3", "base")
    assert prepared.target_version == "v3" and prepared.overlay_base == "base"
    assert _files(checkpoint) == _LINEAGE["base"]
    assert _staged_target(prepared) == _LINEAGE["v3"]
    assert list(store.materialized_cache.iterdir()) == []


def test_staged_deltas_stay_unwritten_without_a_copy_until_the_limit(monkeypatch, tmp_path):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch, layers=2)
    store = checkpoint.store
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    quota = store.max_size_bytes
    store.max_size_bytes = store.cache_size_bytes()
    assert checkpoint.materialize_head(staged=True) is None
    store.max_size_bytes = quota
    monkeypatch.setattr(LocalCheckpointStore, "clones_files", lambda self: False)
    assert checkpoint.materialize_head(staged=True) is None
    state = store.state()
    assert (state.status, state.version, state.base) == (CheckpointState.STAGED, "v1", "base")

    # At the limit, the staged deltas are written in place instead.
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))
    rewritten = checkpoint.materialize_head(staged=True)

    assert rewritten.target_version == "v2" and rewritten.materialized
    assert _files(checkpoint) == _LINEAGE["v2"]


def test_a_rewrite_joins_a_delta_that_appears_while_it_runs(monkeypatch, tmp_path):
    # At the layer limit, the files are rewritten in place and the delta joins the pass.
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch, layers=1)
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))
    answers = iter([None, (_lineage_delta("v3"), False)])

    prepared = checkpoint.materialize_head(staged=True, following=lambda: next(answers, None))

    # "a" was rewritten before v3 appeared and "b" after, in the pass that checked v3.
    state = checkpoint.store.state()
    assert (state.status, state.version, state.base) == (CheckpointState.STAGED, "v3", "v2")
    assert prepared.target_version == "v3" and not prepared.materialized
    assert _files(checkpoint) == _LINEAGE["v2"]
    assert _staged_target(prepared) == _LINEAGE["v3"]
    # v3 stages only the tensor it changes, not the ones its pass rewrote.
    staged = StagedDelta(prepared.staging)
    assert staged.names() == ["a"]
    staged.close()


@pytest.mark.parametrize("layers", [1, None])
def test_a_rewrite_finishes_alone_when_the_following_delta_is_unreadable(
    monkeypatch, tmp_path, layers
):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch, layers=layers)
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True))
    answers = iter([None, (_delta("lost", base="v2", identity="lost"), False)])

    prepared = checkpoint.materialize_head(staged=True, following=lambda: next(answers, None))

    assert prepared.target_version == "v2" and prepared.materialized
    assert checkpoint.store.state().status is CheckpointState.READY
    assert _files(checkpoint) == _LINEAGE["v2"]


@pytest.mark.parametrize("layers,files", [(None, "base"), (1, "v1")])
def test_a_speculative_staged_head_reverts_without_rewriting(
    monkeypatch, tmp_path, layers, files
):
    checkpoint, s3 = _lineage_checkpoint(tmp_path, monkeypatch, layers=layers)
    s3.objects.update(_lineage_snapshot("v2", "v1"))
    store = checkpoint.store
    speculative = receiver_module.speculative_version_id(_uri("v1"))
    checkpoint.prepare_speculative(_delta(speculative, base="base", identity="v1"), staged=True)
    assert store.state().status is CheckpointState.STAGED

    adopted = checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True)
    assert adopted.staging == store.staging_path("v1") and not adopted.materialized

    unwanted = receiver_module.speculative_version_id(_uri("v2"))
    checkpoint.prepare_speculative(_delta(unwanted, base="v1", identity="v2"), staged=True)
    assert store.state().base == files and _files(checkpoint) == _LINEAGE[files]
    s3.objects.update(_lineage_snapshot("other", "v1"))
    other = checkpoint.prepare_chain((_delta("other", base="v1", identity="other"),), staged=True)

    # The revert leaves v1 as it was before the speculation, on the same files.
    assert (other.overlay_base, other.staging_base) == (files, "v1")
    assert not store.staging_path(unwanted).exists()
    assert store.speculation() is None
    assert _staged_target(other) == _LINEAGE["other"]


def test_installing_a_staged_target_activates_it_and_wakes_the_prefetcher(
    monkeypatch, tmp_path
):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch)
    prepared = checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True)
    trigger = checkpoint.store.cache / "prefetch.trigger"

    with checkpoint.installation_context(prepared):
        assert checkpoint.store.active_version() == "base"

    assert checkpoint.store.active_version() == "v1" and trigger.exists()
    # The installed version and its base's files both stay protected.
    assert {"v1", "base"} <= receiver_module._protected_versions(checkpoint.store, "v1")


def test_a_staged_version_keeps_its_files_and_staged_deltas(monkeypatch, tmp_path):
    checkpoint, _s3 = _lineage_checkpoint(tmp_path, monkeypatch)
    store = checkpoint.store
    checkpoint.activate(checkpoint.prepare_chain((_lineage_delta("v1"),), staged=True))
    checkpoint.prepare_chain((_lineage_delta("v2"),), staged=True)

    protected = receiver_module._protected_versions(store, "v2")
    assert {"base", "v1", "v2"} <= protected
    store.max_size_bytes = 0
    with pytest.raises(receiver_module.CheckpointCacheCapacityError):
        store.enforce_capacity(protected_versions=protected)
    assert store.staging_path("v1").exists() and store.staging_path("v2").exists()


def test_staged_deltas_turn_base_tensors_into_the_target(tmp_path):
    before, middle, after = [1.0, 2.0, 3.0], [1.0, 7.0, 3.0], [5.0, 7.0, 3.0]
    _stage(tmp_path / "v1", {"a": (_bytes(before), _bytes(middle))}).close()
    _stage(tmp_path / "v2", {"a": (_bytes(middle), _bytes(after))}, base="v1").close()
    tensors = [("a", torch.tensor(before)), ("c", torch.tensor([4.0]))]

    out = dict(installer_module._with_staged_deltas(tensors, (tmp_path / "v1", tmp_path / "v2")))

    assert out["a"].tolist() == after and out["c"].tolist() == [4.0]
    assert tensors[0][1].tolist() == before


def test_staged_delta_reads_row_ranges_and_checks_file_sizes(tmp_path):
    tensors = {
        "a": (16, np.array([1, 3], dtype=np.uint32), np.array([7, 8], dtype=np.uint8)),
        "b": (16, np.array([2], dtype=np.uint32), np.array([9], dtype=np.uint8)),
    }
    write_staging(tmp_path, base_version="base", tensors=tensors)
    staged = StagedDelta(tmp_path)

    assert staged.row("b") == 1
    assert staged.byte_ranges(0, 2) == ((0, 12), (0, 3))
    assert staged.byte_ranges(1, 2) == ((8, 12), (2, 3))
    out = bytearray(4)
    staged.read_into(POSITIONS_FILE, 8, memoryview(out))
    assert np.frombuffer(bytes(out), dtype=np.uint32).tolist() == [2]
    staged.close()

    with (tmp_path / VALUES_FILE).open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="does not match"):
        StagedDelta(tmp_path)
    installer = inplace_module.InPlaceInstaller(
        torch.nn.Module(), device=torch.device("cpu"), generic_reload=lambda load: None
    )
    with pytest.raises(inplace_module.InPlaceUnsupported, match="does not match"):
        installer.install(tmp_path / "checkpoint", tmp_path)


def test_staged_runs_split_by_size_and_offset_width(tmp_path):
    tensors = {
        "a": (16, np.arange(4, dtype=np.uint32), np.ones(4, dtype=np.uint8)),
        "b": (16, np.arange(3, dtype=np.uint32), np.ones(3, dtype=np.uint8)),
        "c": (1 << 33, np.array([1 << 32], dtype=np.uint64), np.ones(1, dtype=np.uint8)),
        "d": (16, np.arange(5, dtype=np.uint32), np.ones(5, dtype=np.uint8)),
    }
    write_staging(tmp_path, base_version="base", tensors=tensors)
    staged = StagedDelta(tmp_path)

    assert inplace_module.staged_runs(staged, 0, 4, limit=8) == [(0, 2), (2, 3), (3, 4)]
    assert inplace_module.staged_runs(staged, 0, 2, limit=5) == [(0, 1), (1, 2)]
    # Rows larger than the limit form runs of their own.
    assert inplace_module.staged_runs(staged, 1, 4, limit=2) == [(1, 2), (2, 3), (3, 4)]
    assert inplace_module.staged_runs(staged, 1, 1) == []
    staged.close()


class _Exchange:
    """Ranks simulated by threads that all-gather through a barrier."""

    def __init__(self, size):
        self.size = size
        self._barrier = threading.Barrier(size)
        self._parts = [None] * size

    def peers(self, rank):
        def all_gather(tensor):
            self._parts[rank] = tensor
            self._barrier.wait()
            gathered = torch.cat(self._parts)
            self._barrier.wait()
            return gathered

        return inplace_module.Peers(rank, self.size, all_gather)


@pytest.mark.parametrize("size", [2, 3])
def test_peers_split_reads_and_deliver_whole_runs(tmp_path, size):
    rng = np.random.default_rng(5)
    tensors = {}
    for index, nbytes in enumerate((64, 9, 33, 128)):
        after = rng.integers(1, 256, nbytes, dtype=np.uint8)
        after[rng.random(nbytes) < 0.6] = 0
        tensors[f"t{index}"] = (np.zeros(nbytes, dtype=np.uint8), after)
    # A row with no staged bytes forms an empty run.
    tensors["t4"] = (np.zeros(8, dtype=np.uint8), np.zeros(8, dtype=np.uint8))
    staged = _stage(tmp_path / "staging", tensors)
    runs = inplace_module.staged_runs(staged, 0, 5, limit=20)
    assert runs[-1] == (4, 5)
    cpu = torch.device("cpu")
    reader = inplace_module.StagedRunReader(staged, runs, cpu, piece=8, depth=2)
    expected = list(reader)
    reader.close()
    exchange = _Exchange(size)
    delivered = [None] * size

    def read(rank):
        peer = inplace_module.StagedRunReader(
            staged, runs, cpu, peers=exchange.peers(rank), piece=8, depth=2
        )
        delivered[rank] = list(peer)
        peer.close()

    threads = [threading.Thread(target=read, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for run in expected:
        rows = [staged.tensor(f"t{row}") for row in range(run.begin, run.end)]
        offsets = np.concatenate([row.positions for row in rows]).astype(np.int64)
        values = np.concatenate([row.values for row in rows])
        assert run.offsets.to(torch.int64).tolist() == offsets.tolist()
        assert run.values.tolist() == values.tolist()
    for runs_of_rank in delivered:
        assert [(run.begin, run.end) for run in runs_of_rank] == runs
        for got, want in zip(runs_of_rank, expected):
            assert torch.equal(got.offsets, want.offsets)
            assert torch.equal(got.values, want.values)
    staged.close()


def test_runs_place_narrow_and_wide_offsets(tmp_path):
    tensors = {
        "a": (16, np.array([2, 5], dtype=np.uint32), np.array([1, 2], dtype=np.uint8)),
        "b": (1 << 33, np.array([1, (1 << 32) + 3], dtype=np.uint64), np.array([3, 4], dtype=np.uint8)),
    }
    write_staging(tmp_path, base_version="base", tensors=tensors)
    staged = StagedDelta(tmp_path)
    runs = inplace_module.staged_runs(staged, 0, 2)
    assert runs == [(0, 1), (1, 2)]
    cpu = torch.device("cpu")
    flat = torch.zeros(32 + inplace_module._DISCARD_BYTES, dtype=torch.uint8)
    # a: two-byte columns from byte 1 of each four-byte row; b: its first 16
    # bytes as one range at 16, so its byte past 4 GiB is discarded.
    rules = {0: (0, 4, 2, 1), 1: (16, inplace_module._ONE_ROW, 16, 0)}
    placement = inplace_module.StagedPlacement(staged, (0, 2), rules, 32, cpu)

    reader = inplace_module.StagedRunReader(staged, runs, cpu)
    for run in reader:
        placement.place(flat, run, window=1)
    reader.close()

    expected = torch.zeros(32, dtype=torch.uint8)
    expected[1], expected[2], expected[17] = 1, 2, 3
    assert torch.equal(flat[:32], expected)
    staged.close()


def _fake_layerwise(monkeypatch, meta):
    module = ModuleType("vllm.model_executor.model_loader.reload.layerwise")
    module.get_layerwise_info = lambda _layer: SimpleNamespace(restore_metadata=(meta, {}))
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1)])
@pytest.mark.parametrize("transposed", [True, False])
def test_expert_layout_follows_the_weight_loader_rules(monkeypatch, tmp_path, tp_size, tp_rank, transposed):
    experts, out, inner = 2, 4, 6  # gate/up: [out, inner]; down: [inner, out]
    generator = torch.Generator().manual_seed(3)
    dense, tensors, names, metadata = {}, {}, {}, {}
    for expert in range(experts):
        for proj, shape in (("gate", (out, inner)), ("up", (out, inner)), ("down", (inner, out))):
            name = f"model.layers.5.mlp.experts.{expert}.{proj}_proj.weight"
            value = torch.randint(-(2**31), 2**31 - 1, shape, generator=generator, dtype=torch.int32)
            dense[(expert, proj)] = value
            tensors[name] = (np.zeros(value.numel() * 4, dtype=np.uint8), value.reshape(-1).view(torch.uint8).numpy())
            names[(expert, proj, "weight")] = name
            metadata[name] = {"shape": list(shape), "dtype": "I32", "byte_size": value.numel() * 4}
    staged = _stage(tmp_path / "staging", tensors)

    def loader_layout(proj):
        # Reference: the routed-expert loader's transpose, then its TP shard.
        stacked = []
        for expert in range(experts):
            value = dense[(expert, proj)]
            dim = 0 if proj != "down" else 1
            size = value.shape[dim] // tp_size
            value = value.narrow(dim, tp_rank * size, size)
            stacked.append(value.t() if transposed else value)
        return torch.stack(stacked)

    w13 = torch.cat([loader_layout("gate"), loader_layout("up")], dim=2 if transposed else 1)
    w2 = loader_layout("down")
    meta = {}
    for name, tensor in (("w13_weight", w13), ("w2_weight", w2)):
        meta_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        meta_tensor.is_transposed = transposed
        meta[name] = meta_tensor
    _fake_layerwise(monkeypatch, meta)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=tp_size, ep_size=1),
            tp_rank=tp_rank,
        )
    )

    arranged = inplace_module.arrange_expert_delta(layer, staged, names, metadata, device=torch.device("cpu"))

    assert torch.equal(arranged["w13_weight"], w13)
    assert torch.equal(arranged["w2_weight"], w2)
    staged.close()


def test_packed_experts_may_load_into_plain_weight_parameters(monkeypatch, tmp_path):
    experts, rows, cols = 2, 4, 3
    dense, tensors, names, metadata = {}, {}, {}, {}
    for expert in range(experts):
        for proj, shape in (("w1", (rows, cols)), ("w3", (rows, cols)), ("w2", (cols, rows))):
            name = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.{proj}.weight_packed"
            value = torch.randint(0, 255, shape, dtype=torch.uint8)
            dense[(expert, proj)] = value
            tensors[name] = (np.zeros(value.numel(), dtype=np.uint8), value.reshape(-1).numpy())
            metadata[name] = {"shape": list(shape), "dtype": "U8", "byte_size": value.numel()}
            match = inplace_module._EXPERT.search(name)
            names[(expert, inplace_module._PROJECTIONS[match[3]], match[4])] = name
    staged = _stage(tmp_path / "staging", tensors)
    meta = {
        "w13_weight": torch.empty((experts, 2 * rows, cols), dtype=torch.uint8, device="meta"),
        "w2_weight": torch.empty((experts, cols, rows), dtype=torch.uint8, device="meta"),
    }
    _fake_layerwise(monkeypatch, meta)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1), tp_rank=0)
    )

    arranged = inplace_module.arrange_expert_delta(layer, staged, names, metadata, device=torch.device("cpu"))

    expected = torch.stack([torch.cat([dense[(e, "w1")], dense[(e, "w3")]]) for e in range(experts)])
    assert torch.equal(arranged["w13_weight"], expected)
    assert torch.equal(arranged["w2_weight"], torch.stack([dense[(e, "w2")] for e in range(experts)]))
    staged.close()


def test_expert_layers_install_from_one_reader(monkeypatch, tmp_path):
    experts, out, inner = 2, 4, 6
    generator = torch.Generator().manual_seed(11)

    def random(shape):
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=generator, dtype=torch.int32)

    class RoutedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(random((experts, 2 * out, inner)), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(random((experts, inner, out)), requires_grad=False)
            self.moe_config = SimpleNamespace(
                moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1), tp_rank=0
            )
            self.quant_method = SimpleNamespace()

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    for layer in model.layers:
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = RoutedExperts()
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: RoutedExperts)
    _fake_layerwise(
        monkeypatch,
        {
            "w13_weight": torch.empty((experts, 2 * out, inner), dtype=torch.int32, device="meta"),
            "w2_weight": torch.empty((experts, inner, out), dtype=torch.int32, device="meta"),
        },
    )
    # Processing that keeps the loaded layout.
    monkeypatch.setattr(inplace_module, "process_like_weights", lambda _layer, arranged: arranged)
    # Runs of a few rows each, so layers span several runs of one reader.
    monkeypatch.setattr(
        inplace_module, "staged_runs", functools.partial(inplace_module.staged_runs, limit=64)
    )
    deltas, checkpoint_tensors, tensors = {}, {}, {}
    for index in range(2):
        for expert in range(experts):
            for proj, shape in (("gate", (out, inner)), ("up", (out, inner)), ("down", (inner, out))):
                name = f"model.layers.{index}.mlp.experts.{expert}.{proj}_proj.weight"
                delta = random(shape)
                delta[torch.rand(shape, generator=generator) < 0.5] = 0
                deltas[(index, expert, proj)] = delta
                checkpoint_tensors[name] = torch.zeros(shape, dtype=torch.int32)
                tensors[name] = (np.zeros(delta.numel() * 4, dtype=np.uint8), delta.reshape(-1).view(torch.uint8).numpy())
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(checkpoint_tensors, checkpoint / "model.safetensors")
    staged = _stage(tmp_path / "staging", tensors)
    before = [(layer.mlp.experts.w13_weight.clone(), layer.mlp.experts.w2_weight.clone()) for layer in model.layers]
    installer = inplace_module.InPlaceInstaller(model, device=torch.device("cpu"), generic_reload=lambda load: None)

    metrics = installer._install(checkpoint, staged)

    assert metrics["perf/mx_receive_inplace_expert_layers"] == 2
    for index, layer in enumerate(model.layers):
        w13, w2 = before[index]
        gate_up = torch.stack(
            [torch.cat([deltas[(index, e, "gate")], deltas[(index, e, "up")]]) for e in range(experts)]
        )
        down = torch.stack([deltas[(index, e, "down")] for e in range(experts)])
        assert torch.equal(layer.mlp.experts.w13_weight, w13 ^ gate_up)
        assert torch.equal(layer.mlp.experts.w2_weight, w2 ^ down)
    staged.close()


def test_layers_of_other_pipeline_stages_are_skipped(monkeypatch, tmp_path):
    class PPMissingLayer(torch.nn.Module):
        pass

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module(), PPMissingLayer()])
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    installer = inplace_module.InPlaceInstaller(model, device=torch.device("cpu"), generic_reload=lambda load: None)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    save_file({name: torch.zeros(2, dtype=torch.int32)}, checkpoint / "model.safetensors")
    staged = _stage(tmp_path / "staging", {name: (np.zeros(8, dtype=np.uint8), np.arange(8, dtype=np.uint8))})

    metrics = installer._install(checkpoint, staged)

    assert metrics["perf/mx_receive_inplace_expert_layers"] == 0
    staged.close()


def test_generic_install_handles_loaders_that_write_live_storage(monkeypatch, tmp_path):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1, 2, 3, 4], dtype=torch.int32), requires_grad=False)
            self.register_buffer("mirror", torch.tensor([9, 8, 7, 6], dtype=torch.int32), persistent=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

        def load_weights(self, weights):
            layer = self.layers[0]
            for name, tensor in weights:
                if name.endswith("attn.weight"):
                    # XOR restore: processed value XORed into live storage.
                    layer.weight.data.bitwise_xor_(tensor)
                    # A loader mirror writes the loaded value directly.
                    layer.mirror.copy_(tensor)

    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    model = Model()
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: load(None)
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    name = "model.layers.0.attn.weight"
    save_file({name: torch.zeros(4, dtype=torch.int32)}, checkpoint / "model.safetensors")
    delta = torch.tensor([0, 5, 0, 1], dtype=torch.int32)
    staged = _stage(
        tmp_path / "staging",
        {name: (np.zeros(16, dtype=np.uint8), delta.view(torch.uint8).numpy())},
    )

    metrics = installer._install(checkpoint, staged)

    layer = model.layers[0]
    assert metrics["perf/mx_receive_inplace_generic_tensors"] == 1
    assert layer.weight.tolist() == [1, 2 ^ 5, 3, 4 ^ 1]
    assert layer.mirror.tolist() == [9, 8 ^ 5, 7, 6 ^ 1]
    staged.close()


def test_checkpoint_layers_the_model_does_not_build_are_skipped(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    installer = inplace_module.InPlaceInstaller(model, device=torch.device("cpu"), generic_reload=lambda load: None)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    # Layer 1 is a speculative-decoding layer this model does not serve.
    names = ["model.layers.1.mlp.experts.0.gate_proj.weight", "model.layers.1.eh_proj.weight"]
    save_file({name: torch.zeros(2, dtype=torch.int32) for name in names}, checkpoint / "model.safetensors")
    staged = _stage(
        tmp_path / "staging",
        {name: (np.zeros(8, dtype=np.uint8), np.arange(8, dtype=np.uint8)) for name in names},
    )

    metrics = installer._install(checkpoint, staged)

    assert metrics["perf/mx_receive_inplace_expert_layers"] == 0
    assert metrics["perf/mx_receive_inplace_generic_tensors"] == 0
    staged.close()


def test_unrecognized_expert_tensors_are_not_installed_in_place(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    installer = inplace_module.InPlaceInstaller(model, device=torch.device("cpu"), generic_reload=lambda load: None)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    name = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"
    save_file({name: torch.zeros(2, dtype=torch.int32)}, checkpoint / "model.safetensors")
    staged = _stage(tmp_path / "staging", {name: (np.zeros(8, dtype=np.uint8), np.arange(8, dtype=np.uint8))})

    with pytest.raises(inplace_module.InPlaceUnsupported, match="unrecognized"):
        installer._install(checkpoint, staged)
    staged.close()


def test_checkpoint_catalog_groups_names_by_layer():
    names = [
        "model.embed_tokens.weight",
        "model.layers.3.mlp.experts.1.down_proj.weight_packed",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.3.mlp.experts.0.w9.weight",
    ]

    catalog = inplace_module.CheckpointCatalog.of({name: {} for name in names})

    assert catalog.layers == {names[0]: None, names[1]: 3, names[2]: 3, names[3]: 3}
    assert catalog.experts == {3: {(1, "down", "weight_packed"): names[1]}}
    assert catalog.per_expert == {names[1], names[3]}
    assert catalog.unmatched_experts == {names[3]}
    assert catalog.generic == (names[0], names[2])


def test_processing_a_delta_leaves_the_kernel_on_live_storage(monkeypatch):
    class Info:
        def __init__(self):
            self.kernel_tensors = None
            self.kernel_non_persistent_buffers = set()

        def reset(self):
            self.kernel_tensors = None

    info = Info()

    def params_buffers(layer):
        return dict(layer._parameters), dict(layer._buffers)

    def restore_on_meta(layer, _info):
        for name, param in list(layer._parameters.items()):
            delattr(layer, name)
            layer.register_parameter(name, torch.nn.Parameter(torch.empty_like(param), requires_grad=False))

    def place(layer, info):
        for name in list(layer._parameters):
            delattr(layer, name)
        for name, param in info.kernel_tensors[0].items():
            layer.register_parameter(name, param)

    modules = {
        "vllm.model_executor.model_loader.reload.layerwise": {"_place_kernel_tensors": place, "get_layerwise_info": lambda _layer: info},
        "vllm.model_executor.model_loader.reload.meta": {"SKIP_TENSORS": set(), "materialize_layer": lambda _l, _i: None, "restore_layer_on_meta": restore_on_meta},
        "vllm.model_executor.model_loader.reload.utils": {"get_layer_params_buffers": params_buffers, "get_layer_tensors": lambda layer: dict(layer._parameters)},
    }
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader", "vllm.model_executor.model_loader.reload"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, attributes in modules.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    class Method:
        def process_weights_after_loading(self, layer):
            # Like a MoE kernel setup, capture the tensor being processed.
            layer.w13_weight.data.add_(1)
            self.kernel_weight = layer.w13_weight
            layer.extra = "set during processing"

    layer = torch.nn.Module()
    layer.register_parameter("w13_weight", torch.nn.Parameter(torch.tensor([5, 5]), requires_grad=False))
    layer.quant_method = Method()
    live = layer.w13_weight
    layer.quant_method.kernel_weight = live

    processed = inplace_module.process_like_weights(layer, {"w13_weight": torch.tensor([1, 2])})

    assert processed["w13_weight"].tolist() == [2, 3]
    assert layer.w13_weight is live and live.tolist() == [5, 5]
    assert layer.quant_method.kernel_weight is live
    assert not hasattr(layer, "extra")


def test_xor_into_rejects_mismatched_storage():
    target = torch.zeros(4, dtype=torch.int32)
    inplace_module.xor_into(target, torch.ones(4, dtype=torch.int32))
    assert target.tolist() == [1, 1, 1, 1]
    with pytest.raises(inplace_module.InPlaceUnsupported):
        inplace_module.xor_into(target, torch.ones(3, dtype=torch.int32))


def test_xor_into_handles_strided_live_tensors():
    storage = torch.zeros(4, 4, dtype=torch.bfloat16)
    target = storage.t()
    value = torch.arange(16, dtype=torch.int16).view(torch.bfloat16).view(4, 4)

    inplace_module.xor_into(target, value)

    assert torch.equal(inplace_module.bits(target), inplace_module.bits(value))
    assert torch.equal(inplace_module.bits(storage), inplace_module.bits(value).t())


def _installer():
    return installer_module._VllmInstaller(
        model=torch.nn.Module(),
        vllm_config=SimpleNamespace(load_config=SimpleNamespace(load_format="modelexpress"), quant_config=None),
        model_config=SimpleNamespace(model="/launch", revision=None),
        device=torch.device("cpu"),
    )


class _InPlace:
    calls = []
    error = None

    def __init__(
        self, model, *, device, generic_reload, capture_layout, refresh_attention, agree, peers
    ):
        pass

    def install(self, path, staging):
        _InPlace.calls.append((path, staging))
        if _InPlace.error is not None:
            raise _InPlace.error
        return {
            "perf/mx_receive_inplace_time": 0.5,
            "perf/mx_receive_inplace_expert_layers": 2.0,
            "perf/mx_receive_inplace_generic_tensors": 3.0,
        }


@pytest.fixture
def staged_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_INPLACE", "true")
    monkeypatch.setattr(installer_module, "InPlaceInstaller", _InPlace)
    _InPlace.calls, _InPlace.error = [], None
    return PreparedCheckpoint("v2", tmp_path, {}, staging=tmp_path / "staging", staging_base="v1")


def test_in_place_install_needs_the_staged_base_to_be_live(staged_checkpoint):
    installer = _installer()

    assert installer._install_in_place(staged_checkpoint, live_version=None) is None
    assert installer._install_in_place(staged_checkpoint, live_version="v0") is None
    metrics = installer._install_in_place(staged_checkpoint, live_version="v1")

    assert metrics["perf/mx_receive_inplace_time"] == 0.5
    assert _InPlace.calls == [(staged_checkpoint.path, staged_checkpoint.staging)]


@pytest.mark.parametrize("materialized", [True, False])
def test_unsupported_in_place_install_falls_back_to_a_reload(
    staged_checkpoint, monkeypatch, materialized
):
    installer = _installer()
    _InPlace.error = inplace_module.InPlaceUnsupported("no routed experts")
    reloaded = []
    monkeypatch.setattr(
        installer, "install_checkpoint", lambda path, overlay=(): reloaded.append((path, overlay))
    )
    installer._live_version = "v1"
    overlay = () if materialized else (staged_checkpoint.staging,)
    checkpoint = dataclasses.replace(
        staged_checkpoint, overlay=overlay, overlay_base=None if materialized else "v1"
    )

    installer.install(PreparedCheckpointArtifact(checkpoint))

    # Files that hold an earlier version reload through the staged deltas since.
    assert reloaded == [(checkpoint.path, overlay)]
    assert installer._live_version == "v2"


def test_in_place_install_replaces_the_reload(staged_checkpoint, monkeypatch):
    installer = _installer()
    reloaded = []
    monkeypatch.setattr(installer, "install_checkpoint", lambda path, overlay=(): reloaded.append(path))
    installer._live_version = "v1"

    metrics = installer.install(PreparedCheckpointArtifact(staged_checkpoint))

    assert reloaded == []
    assert metrics["perf/mx_receive_inplace_expert_layers"] == 2.0
    assert installer._live_version == "v2"


def test_in_place_install_is_opt_in(staged_checkpoint, monkeypatch):
    monkeypatch.delenv("MX_REFIT_INPLACE", raising=False)
    installer = _installer()

    assert installer._install_in_place(staged_checkpoint, live_version="v1") is None
    assert _InPlace.calls == []


def _recorded(source, param, shape, *, ops=(), offset=0, stride=None, dtype=torch.int32):
    from modelexpress.refit.reshard.types import RecordedCopy

    if stride is None:
        stride = tuple(torch.empty(shape).stride())
    return RecordedCopy(
        src_name=source,
        op_chain=tuple(ops),
        param_name=param,
        dest_offset=offset,
        dest_shape=tuple(shape),
        dest_stride=tuple(stride),
        dest_dtype=dtype,
    )


def test_direct_copies_place_shards_like_the_loader(tmp_path):
    from modelexpress.refit.reshard.types import CaptureResult

    generator = torch.Generator().manual_seed(5)
    q = torch.randint(-(2**31), 2**31 - 1, (6, 4), generator=generator, dtype=torch.int32)
    k = torch.randint(-(2**31), 2**31 - 1, (2, 4), generator=generator, dtype=torch.int32)
    o = torch.randint(-(2**31), 2**31 - 1, (4, 6), generator=generator, dtype=torch.int32)
    deltas = {"q": q, "k": k, "o": o}
    staged = _stage(
        tmp_path / "staging",
        {name: (np.zeros(value.numel() * 4, dtype=np.uint8), value.reshape(-1).view(torch.uint8).numpy()) for name, value in deltas.items()},
    )
    # Rank 1 of 2: q and k fuse into one column shard, o takes a row shard.
    qkv = torch.zeros(4, 4, dtype=torch.int32)
    o_shard = torch.zeros(4, 3, dtype=torch.int32)
    capture = CaptureResult(
        copies=[
            _recorded("q", "qkv", (3, 4), ops=[("narrow", (0, 3, 3), ())]),
            _recorded("k", "qkv", (1, 4), ops=[("narrow", (0, 1, 1), ())], offset=12),
            _recorded("o", "o", (4, 3), ops=[("narrow", (1, 3, 3), ())]),
        ]
    )
    metadata = {name: {"shape": list(value.shape), "dtype": "I32"} for name, value in deltas.items()}
    params = {"qkv": qkv, "o": o_shard}
    layout = {name: (tuple(value.shape), value.dtype) for name, value in params.items()}

    plan = inplace_module.direct_copies(capture, layout, metadata, params, excluded=set())
    with inplace_module._quiet_readonly_buffers():
        for name, copies in plan.items():
            for copy in copies:
                inplace_module.xor_direct(
                    params[copy.param], staged.tensor(name), copy, shape=metadata[name]["shape"], dtype=torch.int32
                )

    assert torch.equal(qkv, torch.cat([q[3:6], k[1:2]]))
    assert torch.equal(o_shard, o[:, 3:6])
    excluded = inplace_module.direct_copies(capture, layout, metadata, params, excluded={"o"})
    assert set(excluded) == {"q", "k"}
    staged.close()


class _DirectLayer(torch.nn.Module):
    def __init__(self, mirror):
        super().__init__()
        self.proj = torch.nn.Module()
        self.proj.register_parameter("weight", torch.nn.Parameter(torch.arange(6, dtype=torch.int32).view(2, 3), requires_grad=False))
        self.attn = _DirectAttention(self.proj)
        self.mirror = mirror
        self.register_buffer("copy", torch.full((2, 3), 7, dtype=torch.int32), persistent=False)


class _DirectAttention(torch.nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.source = [proj]
        self.register_buffer("derived", proj.weight.detach() * 2, persistent=False)

    def process_weights_after_loading(self, _dtype):
        self.derived.copy_(self.source[0].weight * 2)


class _DirectModel(torch.nn.Module):
    NAME = "model.layers.0.proj.weight"
    FROZEN = "model.layers.0.frozen.bias"

    def __init__(self, mirror=False, live_written=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DirectLayer(mirror)])
        frozen = torch.nn.Module()
        frozen.register_parameter("bias", torch.nn.Parameter(torch.tensor([3, 4], dtype=torch.int32), requires_grad=False))
        self.layers[0].frozen = frozen
        self.live_written = live_written
        self.reloads = 0

    def load_weights(self, weights):
        layer = self.layers[0]
        for name, tensor in weights:
            if name == self.FROZEN:
                # A bias a loader writes in place; unchanged deltas write zeros.
                layer.frozen.bias.data.copy_(tensor)
            if name != self.NAME:
                continue
            if self.live_written:
                # Like a bias, which reload leaves in place for its loader to write.
                layer.proj.weight.data.copy_(tensor)
            else:
                # XOR restore of an unprocessed layer.
                layer.proj.weight.data.bitwise_xor_(tensor)
            if layer.mirror:
                layer.copy.copy_(tensor)


def _direct_installer(monkeypatch, model):
    from modelexpress.refit.reshard.types import CaptureResult

    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    monkeypatch.setattr(inplace_module, "_is_deferred_attention", lambda module: isinstance(module, _DirectAttention))
    monkeypatch.setenv("MX_REFIT_INPLACE_DIRECT", "true")

    def reload(load):
        model.reloads += 1
        load(None)
        # vLLM's finalize reruns deferred attention processing.
        model.layers[0].attn.process_weights_after_loading(None)

    def capture(manifest):
        assert [name for name, _dtype, _shape in manifest] == [model.NAME]
        param = "layers.0.proj.weight"
        return CaptureResult(copies=[_recorded(model.NAME, param, (2, 3))]), {param: ((2, 3), torch.int32)}

    def refresh(layers):
        for layer in layers:
            layer.process_weights_after_loading(None)

    return inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=reload, capture_layout=capture, refresh_attention=refresh
    )


def _direct_update(tmp_path, installer, step, delta):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    save_file(
        {_DirectModel.NAME: torch.zeros(2, 3, dtype=torch.int32), _DirectModel.FROZEN: torch.zeros(2, dtype=torch.int32)},
        checkpoint / "model.safetensors",
    )
    staged = _stage(
        tmp_path / f"staging{step}",
        {_DirectModel.NAME: (np.zeros(24, dtype=np.uint8), delta.reshape(-1).view(torch.uint8).numpy())},
    )
    with inplace_module._quiet_readonly_buffers():
        metrics = installer._install(checkpoint, staged)
    staged.close()
    return metrics


@pytest.mark.parametrize("live_written", [False, True])
def test_verified_tensors_install_by_direct_xor(monkeypatch, tmp_path, live_written):
    model = _DirectModel(live_written=live_written)
    installer = _direct_installer(monkeypatch, model)
    first = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.int32)
    second = torch.tensor([[0, 4, 0], [5, 0, 6]], dtype=torch.int32)
    weight = model.layers[0].proj.weight

    reloaded = _direct_update(tmp_path, installer, 1, first)
    direct = _direct_update(tmp_path, installer, 2, second)

    # The reload feeds the whole layer, including the unchanged bias.
    assert reloaded["perf/mx_receive_inplace_generic_tensors"] == 2
    # Peers agree on what each rank verified at the start of the next install.
    assert reloaded["perf/mx_receive_inplace_verified_tensors"] == 0
    assert direct["perf/mx_receive_inplace_verified_tensors"] == 1
    assert direct["perf/mx_receive_inplace_direct_tensors"] == 1
    assert direct["perf/mx_receive_inplace_generic_tensors"] == 0
    assert model.reloads == 2
    expected = torch.arange(6, dtype=torch.int32).view(2, 3) ^ first ^ second
    assert torch.equal(weight, expected)
    assert torch.equal(model.layers[0].attn.derived, expected * 2)
    assert model.layers[0].frozen.bias.tolist() == [3, 4]


def test_tensors_with_a_loader_mirror_keep_reloading(monkeypatch, tmp_path):
    model = _DirectModel(mirror=True)
    installer = _direct_installer(monkeypatch, model)
    first = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.int32)
    second = torch.tensor([[0, 4, 0], [5, 0, 6]], dtype=torch.int32)

    _direct_update(tmp_path, installer, 1, first)
    metrics = _direct_update(tmp_path, installer, 2, second)

    assert metrics["perf/mx_receive_inplace_direct_tensors"] == 0
    assert metrics["perf/mx_receive_inplace_verified_tensors"] == 0
    assert model.reloads == 4
    expected = torch.arange(6, dtype=torch.int32).view(2, 3) ^ first ^ second
    assert torch.equal(model.layers[0].proj.weight, expected)
    assert torch.equal(model.layers[0].copy, torch.full((2, 3), 7, dtype=torch.int32) ^ first ^ second)


def test_attention_refresh_keeps_graph_storage(monkeypatch):
    from contextlib import nullcontext

    config = ModuleType("vllm.config")
    config.set_current_vllm_config = lambda _config: nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale_inv", torch.ones(1), persistent=False)
            self.scale = torch.full((1,), 2.0)

        def process_weights_after_loading(self, _dtype):
            # Processing re-registers what it derives, as vLLM's MLA layers do.
            self.register_buffer("scale_inv", torch.full((1,), 0.5), persistent=False)
            self.scale = torch.full((1,), 4.0)

    layer = Attention()
    buffer, bare = layer.scale_inv, layer.scale
    installer = _installer()
    installer._model_config.dtype = torch.bfloat16

    installer._refresh_attention([layer])

    assert layer.scale_inv is buffer and buffer.tolist() == [0.5]
    assert layer.scale is bare and bare.tolist() == [4.0]
    assert "scale_inv" in layer._non_persistent_buffers_set


def test_peers_must_all_verify_a_tensor_before_it_installs_directly(monkeypatch, tmp_path):
    model = _DirectModel()
    installer = _direct_installer(monkeypatch, model)
    # A peer rank that verified nothing.
    installer._agree = lambda flags: torch.zeros_like(flags)
    first = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=torch.int32)
    second = torch.tensor([[0, 4, 0], [5, 0, 6]], dtype=torch.int32)

    _direct_update(tmp_path, installer, 1, first)
    metrics = _direct_update(tmp_path, installer, 2, second)

    assert metrics["perf/mx_receive_inplace_direct_tensors"] == 0
    assert model.reloads == 4
    assert torch.equal(model.layers[0].proj.weight, torch.arange(6, dtype=torch.int32).view(2, 3) ^ first ^ second)


def test_a_delta_that_misses_a_region_verifies_nothing(monkeypatch, tmp_path):
    model = _DirectModel()
    installer = _direct_installer(monkeypatch, model)
    untouched = torch.zeros(2, 3, dtype=torch.int32)
    untouched[0, 0] = 1
    capture = installer._capture_layout

    def two_regions(manifest):
        result, layout = capture(manifest)
        # The second row loads as its own copy, which this delta never reaches.
        row = _recorded(model.NAME, "layers.0.proj.weight", (1, 3), ops=[("narrow", (0, 1, 1), ())], offset=3)
        first = _recorded(model.NAME, "layers.0.proj.weight", (1, 3), ops=[("narrow", (0, 0, 1), ())])
        result.copies = [first, row]
        return result, layout

    installer._capture_layout = two_regions
    _direct_update(tmp_path, installer, 1, untouched)
    metrics = _direct_update(tmp_path, installer, 2, untouched)

    assert metrics["perf/mx_receive_inplace_direct_tensors"] == 0
