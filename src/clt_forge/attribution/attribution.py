from clt_forge import logger

from clt_forge.attribution.conversion import build_clt_forge_attribution_result
from clt_forge.attribution.loading import (
    CircuitTracerCLTSource,
    _resolve_torch_dtype,
    load_attribution_clt,
    test_clt_performance_on_prompt,
    compare_reconstruction_with_local_clt_class,
)
from clt_forge.attribution.intervention import (
    run_intervention,
    run_intervention_per_feature,
)

from clt_forge.vendor.circuit_tracer.circuit_tracer import ReplacementModel, attribute
from clt_forge.vendor.circuit_tracer.circuit_tracer.graph import prune_graph, compute_graph_scores

import os
import torch
from typing import Any, Dict, List, Literal

class AttributionRunner:
    def __init__(
        self,
        clt_checkpoint: str | None = None,
        model_name: str = "gpt2",
        device: str = "cuda",
        dtype: str | torch.dtype = torch.float32,
        backend: Literal["nnsight", "transformerlens"] = "transformerlens",
        clt_source: CircuitTracerCLTSource = "clt_forge",
        circuit_tracer_clt: str | None = None,
        lazy_encoder: bool = False,
        lazy_decoder: bool = True,
        cache_dir: str | None = None,
        use_cache: bool = True,
        feature_input_hook: str = "hook_resid_mid",
        feature_output_hook: str = "hook_mlp_out",
        scan: str | list[str] | None = None,
        debug: bool = False,
        model_kwargs: Dict[str, Any] | None = None,
    ):
        self.debug = debug

        def log(msg):
            if self.debug:
                logger.info(msg)

        self.log = log
        torch_dtype = _resolve_torch_dtype(dtype)

        clt_ref = circuit_tracer_clt or clt_checkpoint
        if clt_ref is None:
            raise ValueError(
                "Pass clt_checkpoint for CLT-Forge checkpoints or "
                "circuit_tracer_clt for circuit-tracer CLTs."
            )

        self.log("Loading CLT...")
        self.clt = load_attribution_clt(
            clt_ref=clt_ref,
            source=clt_source,
            device=device,
            dtype=torch_dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
            cache_dir=cache_dir,
            use_cache=use_cache,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            debug=debug,
        )

        self.log("Loading model...")
        self.model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=model_name,
            transcoders=self.clt,
            backend=backend,
            device=torch.device(device),
            dtype=torch_dtype,
            **(model_kwargs or {}),
        )

        self.clt_checkpoint = clt_checkpoint
        self.clt_ref = clt_ref
        self.clt_source = clt_source
        self.model_name = model_name
        self.backend = backend

    @classmethod
    def from_circuit_tracer_hub(
        cls,
        hf_ref: str,
        model_name: str,
        **kwargs: Any,
    ) -> "AttributionRunner":
        """Create a runner from an open-source circuit-tracer CLT on HuggingFace."""
        return cls(
            model_name=model_name,
            clt_source="circuit_tracer_hub",
            circuit_tracer_clt=hf_ref,
            **kwargs,
        )

    @classmethod
    def from_circuit_tracer_local(
        cls,
        clt_path: str,
        model_name: str,
        **kwargs: Any,
    ) -> "AttributionRunner":
        """Create a runner from a local circuit-tracer safetensors CLT directory."""
        return cls(
            model_name=model_name,
            clt_source="circuit_tracer_local",
            circuit_tracer_clt=clt_path,
            **kwargs,
        )

    @classmethod
    def from_circuit_tracer_cache(
        cls,
        hf_ref: str,
        model_name: str,
        **kwargs: Any,
    ) -> "AttributionRunner":
        """Create a runner from a circuit-tracer CLT already saved in local cache."""
        return cls(
            model_name=model_name,
            clt_source="circuit_tracer_cache",
            circuit_tracer_clt=hf_ref,
            **kwargs,
        )

    def _build_result(self, graph, prune_result, input_string) -> Dict[str, Any]:
        return build_clt_forge_attribution_result(
            graph=graph,
            prune_result=prune_result,
            tokenizer=self.model.tokenizer,
            input_string=input_string,
            metadata={
                "clt_source": self.clt_source,
                "clt_ref": self.clt_ref,
                "model_name": self.model_name,
                "backend": self.backend,
            },
        )

    def run(
        self,
        input_string: str,
        folder_name: str,
        graph_name: str = "attribution_graph.pt",
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        max_feature_nodes: int = 8192,
        batch_size: int = 256,
        offload: str = "cpu",
        verbose: bool = True,
        feature_threshold: float = 0.8,
        edge_threshold: float = 0.95,
        run_interventions: bool = True,
        intervention_values: List[float] = [0, -5.0, -10.0],
    ):
        self.log(f"Running attribution for prompt: {input_string[:50]}...")

        if self.debug:
            self.log("Running CLT validation checks...")

            test_clt_performance_on_prompt(
                input_string, self.clt, self.model, debug=self.debug
            )

            if self.clt_source == "clt_forge" and self.clt_checkpoint is not None:
                compare_reconstruction_with_local_clt_class(
                    self.clt_checkpoint,
                    input_string,
                    self.clt,
                    self.model,
                    self.model_name,
                    debug=self.debug,
                )

        graph = attribute(
            prompt=input_string,
            model=self.model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )

        replacement_score, completeness_score = compute_graph_scores(graph)

        self.log(f"Replacement score: {replacement_score:.4f}")
        self.log(f"Completeness score: {completeness_score:.4f}")

        prune_result = prune_graph(
            graph=graph,
            node_threshold=feature_threshold,
            edge_threshold=edge_threshold,
        )

        if self.debug:
            self.log(f"Sparse adjacency shape: {prune_result.edge_mask.shape}")

        n_features = graph.selected_features.shape[0]
        self.log(f"Number of features before pruning: {n_features}")
        self.log(f"Number of feature after pruning (not counting error nodes): {prune_result.node_mask[:n_features].sum().item()}")

        result = self._build_result(graph, prune_result, input_string)

        if run_interventions:
            self.log("Running interventions...")

            intervention_results = self.run_intervention_per_feature(
                input_string=input_string,
                result=result,
                intervention_values=intervention_values,
            )

            result["intervention_top_tokens"] = intervention_results

        # save final results
        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, graph_name)
        torch.save(result, save_path)

        self.log(f"Saved attribution graph to {save_path}")

        return result

    def run_intervention_per_feature(
        self,
        input_string: str,
        result: Dict[str, Any],
        intervention_values: List[float],
    ):
        return run_intervention_per_feature(
            model=self.model,
            input_string=input_string,
            result=result,
            intervention_values=intervention_values,
            debug=self.debug,
        )

    def run_intervention(
        self,
        input_string: str,
        features,
        **kwargs,
    ):
        return run_intervention(
            model=self.model,
            input_string=input_string,
            features=features,
            debug=self.debug,
            **kwargs,
        )
