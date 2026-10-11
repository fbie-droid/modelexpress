# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staging of nonzero XOR bytes and in-place installation of deltas."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import safetensors.numpy
import torch
from safetensors.torch import save_file

from modelexpress_rl import ObjectStorageGeneratorConfig, ObjectStorageType, WeightPayloadFormat
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.engines.vllm import inplace as inplace_module
from modelexpress_rl.inference.engines.vllm import installer as installer_module
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from modelexpress_rl.inference.staging import (
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


def _checkpoint(tmp_path, s3, *, initialize=True):
    launch = tmp_path / "launch"
    if not launch.exists():
        launch.mkdir()
        save_file({"weight": torch.tensor([1.0, 2.0])}, launch / "model.safetensors")
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


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("shard_rows", [True, False])
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8])
def test_expert_shards_match_the_dense_delta(tmp_path, tp_size, tp_rank, shard_rows, dtype):
    generator = torch.Generator().manual_seed(7)
    rows, cols, experts = 8, 6, 3
    dense = []
    tensors = {}
    for expert in range(experts):
        delta = torch.randint(0, 255, (rows, cols * torch.empty((), dtype=dtype).element_size()), generator=generator, dtype=torch.uint8)
        delta[torch.rand(delta.shape, generator=generator) < 0.7] = 0
        dense.append(delta.view(dtype).view(rows, cols))
        tensors[f"e{expert}"] = (np.zeros(delta.numel(), dtype=np.uint8), delta.reshape(-1).numpy())
    staged = _stage(tmp_path / "staging", tensors)
    itemsize = torch.empty((), dtype=dtype).element_size()
    shard = (rows // tp_size, cols) if shard_rows else (rows, cols // tp_size)
    scratch = torch.zeros((experts, shard[0] * shard[1] * itemsize), dtype=torch.uint8)

    inplace_module.scatter_expert_shards(
        scratch,
        staged,
        ["e0", None, "e2"],
        rows=rows,
        cols=cols,
        itemsize=itemsize,
        shard_rows=shard_rows,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )

    result = scratch.view(dtype).view(experts, *shard)
    for expert in range(experts):
        if expert == 1:
            assert not result[expert].any()
            continue
        dim = 0 if shard_rows else 1
        size = dense[expert].shape[dim] // tp_size
        expected = dense[expert].narrow(dim, tp_rank * size, size)
        assert torch.equal(result[expert], expected)
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

    def __init__(self, model, *, device, generic_reload):
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


def test_unsupported_in_place_install_falls_back_to_a_reload(staged_checkpoint, monkeypatch):
    installer = _installer()
    _InPlace.error = inplace_module.InPlaceUnsupported("no routed experts")
    reloaded = []
    monkeypatch.setattr(installer, "install_checkpoint", lambda path: reloaded.append(path))
    installer._live_version = "v1"

    installer.install(PreparedCheckpointArtifact(staged_checkpoint))

    assert reloaded == [staged_checkpoint.path]
    assert installer._live_version == "v2"


def test_in_place_install_replaces_the_reload(staged_checkpoint, monkeypatch):
    installer = _installer()
    reloaded = []
    monkeypatch.setattr(installer, "install_checkpoint", lambda path: reloaded.append(path))
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
