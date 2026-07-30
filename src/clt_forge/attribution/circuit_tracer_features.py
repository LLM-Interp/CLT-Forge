import gzip
import json
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from clt_forge import logger


def cantor_pair(layer: int, feature_idx: int) -> int:
    """Return the feature id convention used by circuit-tracer's frontend."""
    return (layer + feature_idx) * (layer + feature_idx + 1) // 2 + feature_idx


def cantor_unpair(feature_id: int) -> tuple[int, int]:
    """Inverse of ``cantor_pair`` as ``(layer, feature_idx)``."""
    w = int(((8 * feature_id + 1) ** 0.5 - 1) // 2)
    t = (w * w + w) // 2
    feature_idx = feature_id - t
    layer = w - feature_idx
    return layer, feature_idx


def _decode_binary_feature_record(record: bytes) -> dict[str, Any]:
    if len(record) < 4:
        raise ValueError("Circuit-tracer binary feature record is too short")

    data_length = int.from_bytes(record[:4], byteorder="little", signed=False)
    compressed = record[4 : 4 + data_length]

    try:
        payload = zlib.decompress(compressed)
    except zlib.error:
        payload = gzip.decompress(compressed)

    return json.loads(payload.decode("utf-8"))


def load_circuit_tracer_feature_data(path: str | Path) -> dict[str, Any]:
    """Load a circuit-tracer feature record from JSON, gzipped JSON, or binary form."""
    feature_path = Path(path)

    if feature_path.suffix == ".json":
        with open(feature_path, "r") as f:
            return json.load(f)

    if feature_path.suffix == ".gz":
        with gzip.open(feature_path, "rt") as f:
            return json.load(f)

    with open(feature_path, "rb") as f:
        return _decode_binary_feature_record(f.read())


def _maybe_feature_coordinates(
    feature_data: Mapping[str, Any],
    layer: int | None,
    feature_idx: int | None,
) -> tuple[int, int]:
    if layer is not None and feature_idx is not None:
        return int(layer), int(feature_idx)

    if "layer" in feature_data:
        layer = int(feature_data["layer"])
    if "feature_index" in feature_data:
        feature_idx = int(feature_data["feature_index"])

    if layer is not None and feature_idx is not None:
        return int(layer), int(feature_idx)

    feature_id = feature_data.get("index")
    if feature_id is None:
        raise ValueError(
            "Could not infer feature coordinates. Pass layer and feature_idx "
            "explicitly or provide a circuit-tracer feature record with an index."
        )

    return cantor_unpair(int(feature_id))


def _highlight_tokens(tokens: list[str], acts: list[float], threshold_ratio: float) -> str:
    if not tokens:
        return ""
    if not acts or len(acts) != len(tokens):
        return "".join(tokens)

    max_act = max(acts)
    if max_act <= 0:
        return "".join(tokens)

    threshold = max_act * threshold_ratio
    parts = []
    in_highlight = False

    for token, act in zip(tokens, acts):
        should_highlight = act >= threshold and act > 0
        if should_highlight and not in_highlight:
            parts.append("<<")
            in_highlight = True
        elif not should_highlight and in_highlight:
            parts.append(">>")
            in_highlight = False

        parts.append(token)

    if in_highlight:
        parts.append(">>")

    return "".join(parts)


def _iter_examples(feature_data: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for quantile in feature_data.get("examples_quantiles", []) or []:
        quantile_name = quantile.get("quantile_name", "")
        for example in quantile.get("examples", []) or []:
            yield quantile_name, example


def _top_activating_tokens(
    examples: Iterable[Mapping[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, float]] = {}

    for example in examples:
        tokens = example.get("tokens", []) or []
        acts = example.get("tokens_acts_list", []) or []
        for token, act in zip(tokens, acts):
            act_float = float(act)
            if act_float <= 0:
                continue
            token_stats = stats.setdefault(token, {"count": 0.0, "total": 0.0})
            token_stats["count"] += 1
            token_stats["total"] += act_float

    ranking = [
        {
            "token": token,
            "frequency": int(values["count"]),
            "average_activation": values["total"] / values["count"],
        }
        for token, values in stats.items()
    ]
    ranking.sort(
        key=lambda item: (item["frequency"], item["average_activation"]),
        reverse=True,
    )
    return ranking[:top_k]


def circuit_tracer_feature_to_clt_forge_feature_dict(
    feature_data: Mapping[str, Any],
    layer: int | None = None,
    feature_idx: int | None = None,
    max_examples: int | None = None,
    top_k_tokens: int = 10,
    highlight_threshold_ratio: float = 0.6,
) -> dict[str, Any]:
    """Convert circuit-tracer feature examples to CLT-Forge frontend JSON schema."""
    layer, feature_idx = _maybe_feature_coordinates(feature_data, layer, feature_idx)

    top_examples: list[str] = []
    top_examples_tks: list[dict[str, Any]] = []
    examples_for_tokens: list[Mapping[str, Any]] = []
    positive_acts: list[float] = []

    for quantile_name, example in _iter_examples(feature_data):
        tokens = list(example.get("tokens", []) or [])
        acts = [float(act) for act in example.get("tokens_acts_list", []) or []]
        examples_for_tokens.append(example)

        if acts:
            positive_acts.extend(act for act in acts if act > 0)

        top_examples.append(
            _highlight_tokens(
                tokens=tokens,
                acts=acts,
                threshold_ratio=highlight_threshold_ratio,
            )
        )
        top_examples_tks.append(
            {
                "tokens": tokens,
                "activations": acts,
                "max_val": max(acts) if acts else 0.0,
                "quantile": quantile_name,
                "train_token_ind": example.get("train_token_ind"),
                "is_repeated_datapoint": example.get("is_repeated_datapoint", False),
            }
        )

        if max_examples is not None and len(top_examples) >= max_examples:
            break

    average_activation = (
        sum(positive_acts) / len(positive_acts) if positive_acts else 0.0
    )

    return {
        "layer": int(layer),
        "feature_index": int(feature_idx),
        "average_activation": float(
            feature_data.get("average_activation", average_activation)
        ),
        "top_examples": top_examples,
        "top_examples_tks": top_examples_tks,
        "top_activating_tokens": _top_activating_tokens(
            examples=examples_for_tokens,
            top_k=top_k_tokens,
        ),
        "description": feature_data.get("description")
        or feature_data.get("label")
        or "Unknown",
        "explanation": feature_data.get("explanation")
        or "No explanation generated",
        "raw_explanation": feature_data.get("raw_explanation", ""),
        "source": "circuit_tracer",
        "transcoder_id": feature_data.get("transcoder_id"),
        "circuit_tracer_index": feature_data.get("index", cantor_pair(layer, feature_idx)),
        "activation_frequency": feature_data.get("activation_frequency"),
        "top_logits": feature_data.get("top_logits", []),
        "bottom_logits": feature_data.get("bottom_logits", []),
        "act_min": feature_data.get("act_min"),
        "act_max": feature_data.get("act_max"),
        "quantile_values": feature_data.get("quantile_values", []),
        "histogram": feature_data.get("histogram", []),
    }


def write_clt_forge_feature_dict(
    feature_dict: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write one CLT-Forge feature JSON file in the frontend's expected layout."""
    layer = int(feature_dict["layer"])
    feature_idx = int(feature_dict["feature_index"])
    path = (
        Path(output_dir)
        / f"layer{layer}"
        / f"feature_{feature_idx}_complete.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_dict, f, indent=2)
    return path


def convert_circuit_tracer_feature_file(
    input_path: str | Path,
    output_dir: str | Path,
    layer: int | None = None,
    feature_idx: int | None = None,
    **kwargs: Any,
) -> Path:
    """Convert one circuit-tracer feature file to a CLT-Forge feature JSON file."""
    feature_data = load_circuit_tracer_feature_data(input_path)
    feature_dict = circuit_tracer_feature_to_clt_forge_feature_dict(
        feature_data=feature_data,
        layer=layer,
        feature_idx=feature_idx,
        **kwargs,
    )
    return write_clt_forge_feature_dict(feature_dict, output_dir)


def _parse_scan_ref(scan: str) -> tuple[str, str | None, str | None]:
    """Parse circuit-tracer scan strings into ``(repo_id, subfolder, revision)``."""
    if scan == "gemma":
        scan = "mwhanna/gemma-scope-transcoders"
    elif scan == "llama":
        scan = "mntss/transcoder-Llama-3.2-1B"

    revision = None
    subfolder = None

    if "//" in scan:
        repo_id, rest = scan.split("//", 1)
        if "@" in rest:
            subfolder, revision = rest.rsplit("@", 1)
        else:
            subfolder = rest
        return repo_id, subfolder, revision

    if "@" in scan:
        scan, revision = scan.rsplit("@", 1)

    parts = scan.split("/")
    if len(parts) > 2:
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
    else:
        repo_id = scan

    return repo_id, subfolder, revision


def download_circuit_tracer_feature_from_hub(
    scan: str,
    layer: int,
    feature_idx: int,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Download one circuit-tracer feature record from HuggingFace feature storage.

    circuit-tracer feature data may be stored either as ``features/{id}.json`` or
    in a binary file indexed by ``features/index.json.gz``. This helper supports
    both layouts.
    """
    from huggingface_hub import hf_hub_download

    repo_id, subfolder, revision = _parse_scan_ref(scan)
    prefix = f"{subfolder}/features" if subfolder else "features"
    feature_id = cantor_pair(layer, feature_idx)

    try:
        json_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{prefix}/{feature_id}.json",
            revision=revision,
            cache_dir=cache_dir,
        )
        return load_circuit_tracer_feature_data(json_path)
    except Exception as json_error:
        try:
            index_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix}/index.json.gz",
                revision=revision,
                cache_dir=cache_dir,
            )
            with gzip.open(index_path, "rt") as f:
                index_data = json.load(f)

            layer_data = index_data[str(layer)] if isinstance(index_data, dict) else index_data[layer]
            offsets = layer_data["offsets"]
            filename = layer_data["filename"]
            start_byte = offsets[feature_idx]
            end_byte = offsets[feature_idx + 1]

            bin_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix}/{filename}",
                revision=revision,
                cache_dir=cache_dir,
            )
            with open(bin_path, "rb") as f:
                f.seek(start_byte)
                record = f.read(end_byte - start_byte)

            return _decode_binary_feature_record(record)
        except Exception as index_error:
            raise FileNotFoundError(
                "Could not load circuit-tracer feature data for "
                f"layer {layer}, feature {feature_idx} from {scan!r}"
            ) from index_error or json_error


def download_clt_forge_feature_dicts_for_graph(
    graph_result: Mapping[str, Any],
    scan: str,
    output_dir: str | Path,
    max_features: int | None = None,
    cache_dir: str | None = None,
    skip_existing: bool = True,
    strict: bool = False,
    **kwargs: Any,
) -> list[Path]:
    """Download CT feature data for a CLT-Forge graph and write frontend JSONs."""
    seen: set[tuple[int, int]] = set()
    written: list[Path] = []

    for _, layer, feature_idx in graph_result["feature_indices"].tolist():
        key = (int(layer), int(feature_idx))
        if key in seen:
            continue
        seen.add(key)

        if max_features is not None and len(written) >= max_features:
            break

        output_path = (
            Path(output_dir)
            / f"layer{key[0]}"
            / f"feature_{key[1]}_complete.json"
        )
        if skip_existing and output_path.exists():
            written.append(output_path)
            continue

        try:
            feature_data = download_circuit_tracer_feature_from_hub(
                scan=scan,
                layer=key[0],
                feature_idx=key[1],
                cache_dir=cache_dir,
            )
            feature_dict = circuit_tracer_feature_to_clt_forge_feature_dict(
                feature_data=feature_data,
                layer=key[0],
                feature_idx=key[1],
                **kwargs,
            )
            written.append(write_clt_forge_feature_dict(feature_dict, output_dir))
        except Exception as exc:
            if strict:
                raise
            logger.warning(
                "Could not convert circuit-tracer feature %s from %s: %s",
                key,
                scan,
                exc,
            )

    return written
