import os
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

from clt_forge import logger

from clt_forge.attribution.loading import (
    load_circuit_tracing_clt_from_local,
    test_clt_performance_on_prompt,
    compare_reconstruction_with_local_clt_class,
)
from clt_forge.attribution.intervention import (
    run_intervention,
    run_intervention_per_feature,
)

from clt_forge.vendor.circuit_tracer.circuit_tracer import ReplacementModel, attribute
from clt_forge.vendor.circuit_tracer.circuit_tracer.graph import prune_graph, compute_graph_scores

@dataclass
class DistributedConfig:
    """Configuration for distributed attribution computation."""
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    backend: str = "nccl"  # nccl, gloo, or feature_sharding
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

class AttributionRunner:
    def __init__(
        self,
        clt_checkpoint: str,
        model_name: str = "gpt2",
        device: str = "cuda",
        debug: bool = False,
        distributed_setup: str | None = None, # None, "ddp", "fsdp", or "feature_sharding"
        rank: int | None = None,
        world_size: int | None = None,
    ):
        self.debug = debug

        # Auto-detect distributed setup if process group is initialized
        if distributed_setup in ["ddp", "fsdp", "feature_sharding"] or (dist.is_available() and dist.is_initialized()):
            self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
            self.world_size = world_size if world_size is not None else (dist.get_world_size() if dist.is_initialized() else 1)
            self.device = f"cuda:{self.rank}"
            self.distributed = True
            self.distributed_setup = distributed_setup or "ddp"
        else:
            self.rank = 0
            self.world_size = 1
            self.device = device
            self.distributed = False
            self.distributed_setup = None

        # Create distributed config for easier access
        self.dist_config = DistributedConfig(
            enabled=self.distributed,
            rank=self.rank,
            world_size=self.world_size,
            backend=self.distributed_setup or "nccl"
        )

        def log(msg):
            if self.debug and self.rank == 0:
                logger.info(msg)

        self.log = log

        self.log("Loading CLT...")
        self.clt = load_circuit_tracing_clt_from_local(
            clt_checkpoint,
            device=self.device,
            debug=debug,
            is_sharded=(self.distributed_setup == "feature_sharding"),
            rank=self.rank,
            world_size=self.world_size,
        )

        self.log("Loading model...")
        self.model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=model_name,
            transcoders=self.clt,
            device=torch.device(self.device),
        )

        if self.distributed_setup == "ddp":
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
            )
        elif self.distributed_setup == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            self.model = FSDP(
                self.model,
                device_id=torch.device(self.device),
            )

        self.clt_checkpoint = clt_checkpoint
        self.model_name = model_name
        
        self.log(f"AttributionRunner initialized: rank={self.rank}, world_size={self.world_size}, "
                 f"distributed={self.distributed}, setup={self.distributed_setup}")

    def _build_result(self, graph, prune_result, input_string) -> Dict[str, Any]:
        sparse_adjacency = prune_result.edge_mask.float()

        active_feature = torch.stack(
            [
                graph.active_features[:, 1],
                graph.active_features[:, 0],
                graph.active_features[:, 2],
            ],
            dim=1,
        )

        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        token_string = [unwrapped_model.tokenizer.decode(t) for t in graph.input_tokens]
        logit_token_strings = [unwrapped_model.tokenizer.decode(t) for t in graph.logit_tokens]

        return {
            "adjacency_matrix": graph.adjacency_matrix.cpu(),
            "feature_indices": active_feature.cpu(),
            "sparse_pruned_adj": sparse_adjacency.cpu(),
            "feature_mask": prune_result.node_mask.cpu(),
            "edge_mask": prune_result.edge_mask.cpu(),
            "logit_tokens": graph.logit_tokens.cpu(),
            "logit_probabilities": graph.logit_probabilities.cpu(),
            "input_tokens": graph.input_tokens.cpu(),
            "input_string": input_string,
            "token_string": token_string,
            "logit_token_strings": logit_token_strings,
        }

    def _gather_graph(self, graph: Any) -> Any:
        """
        Gather graph from all ranks (for DDP/FSDP distributed runs).
        In feature_sharding mode, graphs are already local and don't need gathering.
        """
        if not self.distributed or self.distributed_setup == "feature_sharding":
            return graph
        
        self.log(f"Gathering graph from rank {self.rank}...")
        
        # Gather adjacency matrices from all ranks and average
        gathered_matrices = [torch.zeros_like(graph.adjacency_matrix) for _ in range(self.world_size)]
        dist.all_gather_object(gathered_matrices, graph.adjacency_matrix)
        
        # Average edge weights across ranks (they should be identical due to same data)
        # but averaging ensures numerical stability
        graph.adjacency_matrix = torch.stack(gathered_matrices).mean(dim=0)
        
        self.log(f"Graph gathered: shape={graph.adjacency_matrix.shape}")
        return graph

    def _synchronize_distributed(self) -> None:
        """Add synchronization barrier for distributed runs."""
        if self.distributed:
            dist.barrier()
            
    def _log_distributed_info(self, msg: str) -> None:
        """Log message with distributed context."""
        if self.debug and self.rank == 0:
            dist_info = f"[rank {self.rank}/{self.world_size}]"
            logger.info(f"{dist_info} {msg}")


    def run(
        self,
        input_string: Union[str, List[str]],
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
        if isinstance(input_string, list):
            self._log_distributed_info(f"Running batched attribution for {len(input_string)} prompts...")
            all_inputs = input_string
            if self.distributed:
                # Inter-prompt parallelization: distribute prompts across ranks
                # Each rank handles a strided subset of prompts
                local_inputs = [(idx, s) for idx, s in enumerate(all_inputs) if idx % self.world_size == self.rank]
                self._log_distributed_info(f"Rank {self.rank} handling {len(local_inputs)} prompts out of {len(all_inputs)}")
            else:
                local_inputs = list(enumerate(all_inputs))

            results = []
            for idx, test_string in local_inputs:
                p_graph_name = f"{os.path.splitext(graph_name)[0]}_{idx}.pt" if len(all_inputs) > 1 else graph_name
                res = self._run_single(
                    input_string=test_string,
                    folder_name=folder_name,
                    graph_name=p_graph_name,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    max_feature_nodes=max_feature_nodes,
                    batch_size=batch_size,
                    offload=offload,
                    verbose=verbose,
                    feature_threshold=feature_threshold,
                    edge_threshold=edge_threshold,
                    run_interventions=run_interventions,
                    intervention_values=intervention_values,
                )
                results.append(res)

            if self.distributed:
                self._log_distributed_info(f"Rank {self.rank} finished batch, synchronizing...")
                dist.barrier()
            return results
        else:
            return self._run_single(
                input_string=input_string,
                folder_name=folder_name,
                graph_name=graph_name,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                max_feature_nodes=max_feature_nodes,
                batch_size=batch_size,
                offload=offload,
                verbose=verbose,
                feature_threshold=feature_threshold,
                edge_threshold=edge_threshold,
                run_interventions=run_interventions,
                intervention_values=intervention_values,
            )

    def _run_single(
        self,
        input_string: str,
        folder_name: str,
        graph_name: str,
        max_n_logits: int,
        desired_logit_prob: float,
        max_feature_nodes: int,
        batch_size: int,
        offload: str,
        verbose: bool,
        feature_threshold: float,
        edge_threshold: float,
        run_interventions: bool,
        intervention_values: List[float],
    ):
        self._log_distributed_info(f"Running attribution for prompt: {input_string[:50]}...")

        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model

        if self.debug:
            self._log_distributed_info("Running CLT validation checks...")

            test_clt_performance_on_prompt(
                input_string, self.clt, unwrapped_model, debug=self.debug
            )

            compare_reconstruction_with_local_clt_class(
                self.clt_checkpoint,
                input_string,
                self.clt,
                unwrapped_model,
                self.model_name,
                debug=self.debug,
            )

        # ─────── Phase 1: Compute attribution graph ───────
        self._log_distributed_info(f"Computing attribution graph (rank {self.rank})...")
        
        graph = attribute(
            prompt=input_string,
            model=self.model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose and (self.rank == 0),  # Only main rank shows progress
        )
        
        # Synchronize after graph computation in distributed mode
        self._synchronize_distributed()
        
        # Gather graph from all ranks if needed
        graph = self._gather_graph(graph)
        
        # ─────── Phase 2: Compute graph metrics ───────
        self._log_distributed_info("Computing graph scores...")
        replacement_score, completeness_score = compute_graph_scores(graph)

        self._log_distributed_info(f"Replacement score: {replacement_score:.4f}")
        self._log_distributed_info(f"Completeness score: {completeness_score:.4f}")

        # ─────── Phase 3: Prune graph ───────
        prune_result = prune_graph(
            graph=graph,
            node_threshold=feature_threshold,
            edge_threshold=edge_threshold,
        )

        if self.debug:
            self._log_distributed_info(f"Sparse adjacency shape: {prune_result.edge_mask.shape}")

        n_features = graph.active_features.shape[0]
        self._log_distributed_info(f"Number of features before pruning: {n_features}")
        self._log_distributed_info(f"Number of features after pruning (not counting error nodes): {prune_result.node_mask[:n_features].sum().item()}")

        result = self._build_result(graph, prune_result, input_string)

        # ─────── Phase 4: Run interventions ───────
        if run_interventions:
            self._log_distributed_info(f"Running interventions (rank {self.rank})...")

            intervention_results = self.run_intervention_per_feature(
                input_string=input_string,
                result=result,
                intervention_values=intervention_values,
            )

            result["intervention_top_tokens"] = intervention_results
            
            # Synchronize after interventions
            self._synchronize_distributed()

        # ─────── Phase 5: Save results ───────
        # Only main rank saves to avoid race conditions
        if self.rank == 0:
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
