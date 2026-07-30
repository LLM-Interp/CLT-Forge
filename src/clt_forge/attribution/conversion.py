from pathlib import Path
from typing import Any, Dict

import torch

import clt_forge.vendor.circuit_tracer  # noqa: F401
from circuit_tracer.graph import (
    Graph,
    PruneResult,
    prune_graph,
)


def _decode_token(tokenizer: Any, token_id: int | torch.Tensor) -> str:
    token_int = int(token_id)
    try:
        return tokenizer.decode([token_int])
    except TypeError:
        return tokenizer.decode(token_int)


def _logit_token_strings(graph: Graph, tokenizer: Any) -> list[str]:
    strings: list[str] = []
    for target in graph.logit_targets:
        if target.token_str:
            strings.append(target.token_str)
        elif target.vocab_idx < graph.vocab_size:
            strings.append(_decode_token(tokenizer, target.vocab_idx))
        else:
            strings.append(str(target.vocab_idx))
    return strings


def selected_feature_indices(graph: Graph) -> torch.Tensor:
    """Return selected graph features as ``(pos, layer, feature_idx)`` rows.

    circuit-tracer stores all non-zero features in ``graph.active_features`` as
    ``(layer, pos, feature_idx)``. The graph adjacency only contains
    ``graph.selected_features``. CLT-Forge's frontend expects the selected rows
    in ``(pos, layer, feature_idx)`` order.
    """
    active_features = graph.active_features
    selected_features = getattr(graph, "selected_features", None)

    if selected_features is not None:
        active_features = active_features[selected_features]

    return torch.stack(
        [
            active_features[:, 1],
            active_features[:, 0],
            active_features[:, 2],
        ],
        dim=1,
    )


def build_clt_forge_attribution_result(
    graph: Graph,
    prune_result: PruneResult,
    tokenizer: Any,
    input_string: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Convert a circuit-tracer graph into the CLT-Forge frontend ``.pt`` schema."""
    input_tokens = graph.input_tokens.reshape(-1)
    logit_tokens = graph.logit_token_ids.cpu()

    result: Dict[str, Any] = {
        "adjacency_matrix": graph.adjacency_matrix.cpu(),
        "feature_indices": selected_feature_indices(graph).cpu(),
        "sparse_pruned_adj": prune_result.edge_mask.float().cpu(),
        "feature_mask": prune_result.node_mask.cpu(),
        "edge_mask": prune_result.edge_mask.cpu(),
        "logit_tokens": logit_tokens,
        "logit_probabilities": graph.logit_probabilities.cpu(),
        "input_tokens": input_tokens.cpu(),
        "input_string": input_string if input_string is not None else graph.input_string,
        "token_string": [_decode_token(tokenizer, token) for token in input_tokens],
        "logit_token_strings": _logit_token_strings(graph, tokenizer),
        "n_layers": int(graph.cfg.n_layers),
    }

    if metadata:
        result["metadata"] = metadata

    if graph.scan is not None:
        result["circuit_tracer_scan"] = graph.scan

    return result


def convert_circuit_tracer_graph_to_clt_forge_result(
    graph: Graph,
    tokenizer: Any,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.95,
    input_string: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Prune and convert a circuit-tracer ``Graph`` for CLT-Forge visualization."""
    prune_result = prune_graph(
        graph=graph,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    return build_clt_forge_attribution_result(
        graph=graph,
        prune_result=prune_result,
        tokenizer=tokenizer,
        input_string=input_string,
        metadata=metadata,
    )


def convert_circuit_tracer_graph_file_to_clt_forge_result(
    graph_path: str | Path,
    tokenizer: Any | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.95,
    map_location: str = "cpu",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load a circuit-tracer ``Graph.to_pt`` file and convert it for CLT-Forge."""
    graph = Graph.from_pt(str(graph_path), map_location=map_location)

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)

    return convert_circuit_tracer_graph_to_clt_forge_result(
        graph=graph,
        tokenizer=tokenizer,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        metadata=metadata,
    )


def save_clt_forge_attribution_result(
    result: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a CLT-Forge-compatible attribution result dict."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return path
