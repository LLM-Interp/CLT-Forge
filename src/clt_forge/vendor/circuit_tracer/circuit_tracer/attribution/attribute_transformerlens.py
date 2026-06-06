"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import logging
import time
from collections.abc import Sequence
from typing import Literal

import torch
from tqdm import tqdm

from circuit_tracer.attribution.targets import (
    AttributionTargets,
    TargetSpec,
    log_attribution_target_info,
)
from circuit_tracer.graph import Graph, compute_partial_influences
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.utils.disk_offload import offload_modules


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: TransformerLensReplacementModel,
    *,
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for *prompt* using TransformerLens backend.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``TransformerLensReplacementModel``
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or tuple)
                            with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


def _run_attribution(
    model,
    prompt,
    attribution_targets,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    logger,
    update_interval=4,
):
    import math
    import torch.distributed as dist
    is_dist = dist.is_available() and dist.is_initialized()
    if is_dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    unwrapped_model = model.module if hasattr(model, "module") else model

    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = unwrapped_model.ensure_tokenized(prompt)

    ctx = unwrapped_model.setup_attribution(input_ids, model_wrapper=model)
    activation_matrix = ctx.activation_matrix

    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {ctx.activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(unwrapped_model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(unwrapped_model):
        residual = model(input_ids.expand(batch_size, -1), stop_at_layer=unwrapped_model.cfg.n_layers)
        ctx._resid_activations[-1] = unwrapped_model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in unwrapped_model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=ctx.logits[0, -1],
        unembed_proj=unwrapped_model.unembed.W_U,
        tokenizer=unwrapped_model.tokenizer,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    log_attribution_target_info(targets, attribution_targets, logger)

    if offload:
        offload_handles += offload_modules([unwrapped_model.unembed, unwrapped_model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(targets)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    if is_dist:
        n_targets = len(targets)
        local_target_indices = list(range(rank, n_targets, world_size))
        local_rows_list = []
        for chunk_start in range(0, len(local_target_indices), batch_size):
            chunk_indices = local_target_indices[chunk_start : chunk_start + batch_size]
            batch = targets.logit_vectors[chunk_indices]
            rows = ctx.compute_batch(
                layers=torch.full((batch.shape[0],), n_layers, device=unwrapped_model.cfg.device),
                positions=torch.full((batch.shape[0],), n_pos - 1, device=unwrapped_model.cfg.device),
                inject_values=batch,
                retain_graph=True,
            )
            local_rows_list.append((chunk_indices, rows))
        
        edge_matrix_logits = torch.zeros(n_logits, logit_offset, device=unwrapped_model.cfg.device)
        for idxs, rows in local_rows_list:
            edge_matrix_logits[idxs] = rows.to(device=edge_matrix_logits.device)
            
        dist.all_reduce(edge_matrix_logits, op=dist.ReduceOp.SUM)
        edge_matrix[:n_logits, :logit_offset] = edge_matrix_logits.cpu()
        row_to_node_index[:n_logits] = torch.arange(n_logits) + logit_offset
    else:
        for i in range(0, len(targets), batch_size):
            batch = targets.logit_vectors[i : i + batch_size]
            rows = ctx.compute_batch(
                layers=torch.full((batch.shape[0],), n_layers),
                positions=torch.full((batch.shape[0],), n_pos - 1),
                inject_values=batch,
            )
            edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
            row_to_node_index[i : i + batch.shape[0]] = (
                torch.arange(i, i + batch.shape[0]) + logit_offset
            )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose or rank != 0)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], targets.logit_probabilities, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        if is_dist:
            local_size = (len(pending) + world_size - 1) // world_size
            local_pending = torch.zeros(local_size, dtype=torch.long, device=feat_layers.device)
            for r in range(world_size):
                r_indices = pending[r::world_size]
                if rank == r:
                    local_pending[:len(r_indices)] = r_indices
            
            for i in range(0, local_size, batch_size):
                local_batch = local_pending[i : i + batch_size]
                local_len = len(local_batch)
                
                rows = ctx.compute_batch(
                    layers=feat_layers[local_batch],
                    positions=feat_pos[local_batch],
                    inject_values=ctx.encoder_vecs[local_batch],
                    retain_graph=n_visited < max_feature_nodes,
                )
                
                gathered_rows = [torch.zeros_like(rows) for _ in range(world_size)]
                gathered_indices = [torch.zeros(local_len, dtype=torch.long, device=feat_layers.device) for _ in range(world_size)]
                
                dist.all_gather(gathered_rows, rows)
                dist.all_gather(gathered_indices, local_batch)
                
                for j in range(local_len):
                    for r in range(world_size):
                        total_idx = (i + j) * world_size + r
                        if total_idx < len(pending):
                            orig_idx = pending[total_idx].item()
                            row_val = gathered_rows[r][j]
                            
                            edge_matrix[st, :logit_offset] = row_val.cpu()
                            row_to_node_index[st] = orig_idx
                            visited[orig_idx] = True
                            st += 1
                            n_visited += 1
                            if rank == 0:
                                pbar.update(1)
        else:
            queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

            for idx_batch in queue:
                n_visited += len(idx_batch)

                rows = ctx.compute_batch(
                    layers=feat_layers[idx_batch],
                    positions=feat_pos[idx_batch],
                    inject_values=ctx.encoder_vecs[idx_batch],
                    retain_graph=n_visited < max_feature_nodes,
                )

                end = min(st + batch_size, st + rows.shape[0])
                edge_matrix[st:end, :logit_offset] = rows.cpu()
                row_to_node_index[st:end] = idx_batch
                visited[idx_batch] = True
                st = end
                pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=unwrapped_model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_targets=targets.logit_targets,
        logit_probabilities=targets.logit_probabilities,
        vocab_size=targets.vocab_size,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=unwrapped_model.cfg,
        scan=unwrapped_model.scan,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph
