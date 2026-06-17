import gzip
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clt_forge.attribution.circuit_tracer_features import (
    cantor_pair,
    circuit_tracer_feature_to_clt_forge_feature_dict,
    convert_circuit_tracer_feature_file,
    download_circuit_tracer_feature_from_hub,
    load_circuit_tracer_feature_data,
    write_clt_forge_feature_dict,
)
from clt_forge.attribution.conversion import build_clt_forge_attribution_result
from clt_forge.attribution.loading import load_circuit_tracer_clt_from_hub
from clt_forge.frontend.config.settings import AppConfig
from clt_forge.frontend.data.loaders import DataLoader
from clt_forge.vendor.circuit_tracer.circuit_tracer.attribution.targets import LogitTarget
from clt_forge.vendor.circuit_tracer.circuit_tracer.graph import Graph, PruneResult
from clt_forge.vendor.circuit_tracer.circuit_tracer.transcoder.cross_layer_transcoder import (
    CrossLayerTranscoder,
)
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig


class ToyTokenizer:
    def decode(self, token):
        if isinstance(token, list):
            return "".join(f"tok{int(t)}" for t in token)
        return f"tok{int(token)}"


def _toy_graph() -> Graph:
    n_layers = 2
    n_tokens = 3
    n_features = 2
    n_logits = 1
    total_nodes = n_features + n_layers * n_tokens + n_tokens + n_logits

    cfg = UnifiedConfig(
        n_layers=n_layers,
        d_model=4,
        d_head=2,
        n_heads=2,
        d_mlp=8,
        d_vocab=100,
        tokenizer_name="toy-tokenizer",
        model_name="toy-model",
        original_architecture="toy",
    )

    return Graph(
        input_string="tok10tok11tok12",
        input_tokens=torch.tensor([10, 11, 12]),
        active_features=torch.tensor(
            [
                [0, 0, 5],
                [1, 1, 7],
                [1, 2, 9],
            ],
            dtype=torch.long,
        ),
        selected_features=torch.tensor([0, 2], dtype=torch.long),
        activation_values=torch.tensor([1.0, 2.0, 3.0]),
        adjacency_matrix=torch.ones(total_nodes, total_nodes),
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="target", vocab_idx=42)],
        logit_probabilities=torch.tensor([1.0]),
        scan="mntss/clt-gemma-2-2b-426k",
    )


def _toy_prune_result(graph: Graph) -> PruneResult:
    n_nodes = graph.adjacency_matrix.shape[0]
    return PruneResult(
        node_mask=torch.ones(n_nodes, dtype=torch.bool),
        edge_mask=torch.ones(n_nodes, n_nodes, dtype=torch.bool),
        cumulative_scores=torch.ones(n_nodes),
    )


def _feature_data() -> dict:
    return {
        "index": cantor_pair(2, 17),
        "transcoder_id": "mntss/clt-gemma-2-2b-426k",
        "examples_quantiles": [
            {
                "quantile_name": "top",
                "examples": [
                    {
                        "tokens": ["The", " capital", " is"],
                        "tokens_acts_list": [0.0, 1.5, 0.2],
                        "train_token_ind": 1,
                        "is_repeated_datapoint": False,
                    }
                ],
            }
        ],
        "top_logits": [" Paris"],
        "bottom_logits": [" London"],
        "act_min": 0.0,
        "act_max": 2.0,
        "quantile_values": [0.1, 1.0],
        "histogram": [0, 2, 1],
        "activation_frequency": 0.001,
    }


def test_graph_conversion_matches_frontend_dataloader_contract(tmp_path: Path):
    graph = _toy_graph()
    result = build_clt_forge_attribution_result(
        graph=graph,
        prune_result=_toy_prune_result(graph),
        tokenizer=ToyTokenizer(),
    )

    assert result["n_layers"] == 2
    assert result["feature_indices"].tolist() == [[0, 0, 5], [2, 1, 9]]

    graph_path = tmp_path / "attribution_graph.pt"
    torch.save(result, graph_path)

    cfg = AppConfig(
        attr_graph_path=str(graph_path),
        dict_base_folder=str(tmp_path / "feature_dicts"),
        clt_checkpoint="",
        model_name="toy-model",
        model_class_name="toy",
    )
    graph_data = DataLoader(cfg).preprocess_data()

    assert graph_data.n_layers == 2
    assert graph_data.prompt_length == 3
    assert graph_data.feature_indices.tolist() == [[0, 0, 5], [2, 1, 9]]
    assert graph_data.top5_logit_tokens == ["target"]


def test_circuit_tracer_feature_conversion_writes_frontend_json(tmp_path: Path):
    feature_dict = circuit_tracer_feature_to_clt_forge_feature_dict(_feature_data())

    assert feature_dict["layer"] == 2
    assert feature_dict["feature_index"] == 17
    assert feature_dict["source"] == "circuit_tracer"
    assert "<< capital>>" in feature_dict["top_examples"][0]
    assert feature_dict["top_activating_tokens"][0]["token"] == " capital"

    path = write_clt_forge_feature_dict(feature_dict, tmp_path)
    assert path == tmp_path / "layer2" / "feature_17_complete.json"
    assert json.loads(path.read_text())["feature_index"] == 17


def test_loads_circuit_tracer_binary_feature_record(tmp_path: Path):
    payload = json.dumps(_feature_data()).encode("utf-8")
    compressed = zlib.compress(payload)
    record = len(compressed).to_bytes(4, byteorder="little") + compressed

    feature_path = tmp_path / "feature.bin"
    feature_path.write_bytes(record)

    assert load_circuit_tracer_feature_data(feature_path)["index"] == cantor_pair(2, 17)


def test_convert_circuit_tracer_feature_file_accepts_explicit_coordinates(tmp_path: Path):
    source_path = tmp_path / "feature.json"
    source_path.write_text(json.dumps({**_feature_data(), "index": 123}))

    out_path = convert_circuit_tracer_feature_file(
        source_path,
        tmp_path / "converted",
        layer=4,
        feature_idx=5,
    )

    assert out_path == tmp_path / "converted" / "layer4" / "feature_5_complete.json"
    assert json.loads(out_path.read_text())["layer"] == 4


def test_download_circuit_tracer_feature_supports_binary_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = json.dumps(_feature_data()).encode("utf-8")
    compressed = zlib.compress(payload)
    record = len(compressed).to_bytes(4, byteorder="little") + compressed

    index_path = tmp_path / "index.json.gz"
    offsets = [0] * 19
    offsets[18] = len(record)
    with gzip.open(index_path, "wt") as f:
        json.dump({"2": {"filename": "layer2.bin", "offsets": offsets}}, f)

    bin_path = tmp_path / "layer2.bin"
    bin_path.write_bytes(record)

    def fake_hf_hub_download(**kwargs):
        filename = kwargs["filename"]
        if filename.endswith(".json"):
            raise FileNotFoundError(filename)
        if filename.endswith("index.json.gz"):
            return str(index_path)
        if filename.endswith("layer2.bin"):
            return str(bin_path)
        raise AssertionError(filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    data = download_circuit_tracer_feature_from_hub(
        scan="mntss/clt-gemma-2-2b-426k",
        layer=2,
        feature_idx=17,
    )

    assert data["index"] == cantor_pair(2, 17)


def test_hub_loader_returns_cross_layer_transcoder(monkeypatch: pytest.MonkeyPatch):
    clt = CrossLayerTranscoder(
        n_layers=1,
        d_transcoder=2,
        d_model=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_decoder=True,
        lazy_encoder=False,
    )

    def fake_load_transcoder_from_hub(**kwargs):
        assert kwargs["hf_ref"] == "mntss/clt-gemma-2-2b-426k"
        assert kwargs["device"] == torch.device("cpu")
        assert kwargs["dtype"] == torch.float32
        return clt, {"model_kind": "cross_layer_transcoder"}

    def fake_import_module(name):
        assert name == "clt_forge.vendor.circuit_tracer.circuit_tracer.utils.hf_utils"
        return SimpleNamespace(load_transcoder_from_hub=fake_load_transcoder_from_hub)

    monkeypatch.setattr("clt_forge.attribution.loading.import_module", fake_import_module)

    loaded = load_circuit_tracer_clt_from_hub(
        "mntss/clt-gemma-2-2b-426k",
        device="cpu",
        dtype="float32",
    )

    assert loaded is clt
