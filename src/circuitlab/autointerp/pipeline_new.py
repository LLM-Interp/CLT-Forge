# import os
# import json
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import torch

# from circuitlab.config import AutoInterpConfig
# from circuitlab.utils import LatentCache_CFG_FILENAME, DICT_FOLDERNAME
# from circuitlab.clt import CLT
# from circuitlab.load_model import load_model
# from circuitlab.training.activations_store import ActivationsStore
# from circuitlab import logger
# from circuitlab.transformer_lens.multilingual_patching import (
#     patch_official_model_names,
#     patch_convert_hf_model_config,
# )
# from circuitlab.training.optim import JumpReLU

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TOP_K_DEFAULT = 100
# N_TOP_ACTIVATING_TOKENS_SHOWN = 4

# class AutoInterp:
#     """
#     Feature-parallel AutoInterp.

#     Key idea:
#       - parallelize over `index_list` (feature ids)
#       - keep CLT on CPU
#       - move only W_enc[:, :, index_list], b_enc[:, index_list], log_threshold[:, index_list] to GPU
#       - stream activations (x) from ActivationsStore
#       - maintain top-K per feature (values + tokens + ctx activations), per layer
#       - use the payload directly to create the feature dictionaries

#     This avoids:
#       - saving huge per-chunk latent caches
#       - reloading chunks per split
#       - heap loops over 50k features
#     """

#     def __init__(self, cfg: AutoInterpConfig):
#         self.cfg = cfg
#         self.total_autointerp_tokens = cfg.total_autointerp_tokens
#         self.device = torch.device(self.cfg.device)
#         self.ctx = cfg.context_size

#         patch_official_model_names()
#         patch_convert_hf_model_config()

#         self.model = load_model(
#             self.cfg.model_class_name,
#             self.cfg.model_name,
#             device=self.device,
#             model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
#         )

#         # CLT stays on CPU. We'll slice/move the necessary encoder params per split.
#         self.clt = CLT.load_from_pretrained(self.cfg.clt_path, "cpu")

#         self.activations_store = ActivationsStore(
#             self.model,
#             self.cfg,
#             estimated_norm_scaling_factor_in=self.clt.estimated_norm_scaling_factor_in,
#             estimated_norm_scaling_factor_out=self.clt.estimated_norm_scaling_factor_out,
#         )

#     def run(
#         self,
#         *,
#         worker_id: str,
#         index_list: List[int],
#         top_k: int = TOP_K_DEFAULT,
#         save_dir: Optional[Path] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> Path:
#         """
#         Stream over data once, compute top-K per feature (index_list),
#         and directly write feature dictionaries.
#         """

#         if not index_list:
#             raise ValueError("index_list must be non-empty.")

#         if save_dir is None:
#             if self.cfg.latent_cache_path is None:
#                 raise ValueError("cfg.latent_cache_path must be set or pass save_dir.")
#             save_dir = Path(self.cfg.latent_cache_path)

#         save_dir.mkdir(parents=True, exist_ok=True)

#         # Save config once
#         cfg_path = save_dir / LatentCache_CFG_FILENAME
#         cfg_path.write_text(json.dumps(self.cfg.to_dict()))
#         logger.info(f"Latent cache config saved to: {cfg_path}")

#         run_dtype = dtype if dtype is not None else self.activations_store.dtype
#         self._prepare_encoder_subset(index_list=index_list, dtype=run_dtype)

#         state: Optional[Dict[str, Any]] = None
#         iterator = iter(self.activations_store)
#         n_tokens = 0

#         while n_tokens < self.cfg.total_autointerp_tokens:
#             tokens_cpu, acts_in, _ = next(iterator)

#             # Move only inputs
#             x = acts_in.to(self.device, dtype=run_dtype)

#             with torch.no_grad():
#                 feat_act_sub = self._encode_subset(x)  # [B, L, F]

#             # Convert to sequences
#             tokens_seq, acts_seq = self._to_sequences(tokens_cpu, feat_act_sub)

#             n_tokens += int(feat_act_sub.shape[0])

#             if tokens_seq.shape[0] > 0:
#                 if state is None:
#                     n_layers = int(acts_seq.shape[2])
#                     state = self._init_topk_state(
#                         n_layers=n_layers,
#                         F=len(index_list),
#                         top_k=top_k,
#                         ctx=self.ctx,
#                         dtype=run_dtype,
#                         device=self.device,
#                     )

#                 self._update_state_with_batch(
#                     state=state,
#                     tokens_seq_cpu=tokens_seq,
#                     acts_seq_gpu=acts_seq,
#                     top_k=top_k,
#                 )

#             del x, feat_act_sub, acts_seq

#             if self.device.type == "cuda":
#                 torch.cuda.empty_cache()

#         if state is None:
#             raise RuntimeError("No sequences processed; state was never initialized.")

#         payload = self._state_to_cpu_payload(
#             state=state,
#             index_list=index_list,
#             worker_id=worker_id,
#             top_k=top_k,
#             dtype=run_dtype,
#         )

#         if self.cfg.latent_cache_path is not None:
#             dict_root = Path(self.cfg.latent_cache_path) / DICT_FOLDERNAME
#         else:
#             dict_root = save_dir / DICT_FOLDERNAME

#         n_layers = len(payload["top_vals_by_layer"])

#         for layer in range(n_layers):
#             self._write_feature_dicts_for_layer_from_payload(
#                 layer=layer,
#                 payload=payload,
#                 out_dir=dict_root / f"layer{layer}",
#             )

#         logger.info(f"Finished AutoInterp run for worker {worker_id}")

#         return save_dir

#     def _write_feature_dicts_for_layer_from_payload(
#         self,
#         *,
#         layer: int,
#         payload: Dict[str, Any],
#         out_dir: Path,
#         index_list: Optional[List[int]] = None,
#     ) -> None:
#         """
#         Write per-feature JSON dictionaries for a given layer directly from an in-memory payload.
#         """

#         out_dir.mkdir(parents=True, exist_ok=True)

#         all_feats: List[int] = payload["index_list"]
#         if index_list is None:
#             use_feats = all_feats
#         else:
#             use_set = set(index_list)
#             use_feats = [f for f in all_feats if f in use_set]

#         top_k = int(payload["top_k"])
#         ctx = int(payload["ctx"])
#         if ctx != self.ctx:
#             raise ValueError(f"Payload ctx={ctx} != self.ctx={self.ctx}")

#         top_vals: torch.Tensor = payload["top_vals_by_layer"][layer]      # [K, F]
#         top_tokens: torch.Tensor = payload["top_tokens_by_layer"][layer]  # [K, F, ctx]
#         top_acts: torch.Tensor = payload["top_acts_by_layer"][layer]      # [K, F, ctx]
#         sum_pos: torch.Tensor = payload["sum_pos_by_layer"][layer]        # [F]
#         count_pos: torch.Tensor = payload["count_pos_by_layer"][layer]    # [F]

#         if top_vals.dim() != 2:
#             raise ValueError(f"top_vals has wrong shape: {tuple(top_vals.shape)}")
#         if top_tokens.dim() != 3 or top_acts.dim() != 3:
#             raise ValueError(
#                 f"top_tokens/top_acts have wrong shapes: "
#                 f"{tuple(top_tokens.shape)} / {tuple(top_acts.shape)}"
#             )

#         K, F = top_vals.shape
#         if K != top_k or F != len(all_feats):
#             raise ValueError(
#                 f"Payload shapes inconsistent. "
#                 f"top_vals: {tuple(top_vals.shape)}, expected ({top_k}, {len(all_feats)})"
#             )

#         tokenizer = self.model.tokenizer

#         # Precompute mapping feature_id -> column index in [0..F-1]
#         feat_to_col: Dict[int, int] = {feat_id: j for j, feat_id in enumerate(all_feats)}

#         for feat_id in use_feats:
#             j = feat_to_col[feat_id]

#             # Build sequence list in descending order (topk is already sorted by value due to torch.topk)
#             sequences: List[Dict[str, Any]] = []
#             for k in range(top_k):
#                 v = float(top_vals[k, j].item())
#                 if v == float("-inf"):
#                     continue
#                 sequences.append(
#                     {
#                         "tokens": top_tokens[k, j],       # [ctx]
#                         "activations": top_acts[k, j],    # [ctx]
#                         "max_val": v,
#                     }
#                 )

#             top_examples: List[str] = []
#             sequences_serializable: List[Dict[str, Any]] = []

#             for s in sequences:
#                 # keep the same “remove BOS” behavior as your original code
#                 tks = s["tokens"][1:]
#                 acts = s["activations"][1:]
#                 top_examples.append(
#                     highlight_activations(tks, acts, tokenizer, threshold_ratio=0.6)
#                 )
#                 sequences_serializable.append(
#                     {
#                         "tokens": s["tokens"].tolist(),
#                         "activations": s["activations"].tolist(),
#                         "max_val": float(s["max_val"]),
#                     }
#                 )

#             top_activating_tokens = self._get_top_activating_tokens_from_sequences(
#                 sequences=sequences,
#                 tokenizer=tokenizer,
#                 top_k=N_TOP_ACTIVATING_TOKENS_SHOWN,
#                 threshold_ratio=0.6,
#             )

#             c = int(count_pos[j].item())
#             avg_activation = float(sum_pos[j].item()) / c if c > 0 else 0.0

#             feature_dict = {
#                 "layer": int(layer),
#                 "feature_index": int(feat_id),
#                 "description": "Unknown",
#                 "explanation": "No explanation generated",
#                 "top_examples": top_examples,
#                 "top_examples_tks": sequences_serializable,
#                 "average_activation": float(avg_activation),
#                 "top_activating_tokens": top_activating_tokens,
#                 "raw_explanation": "",
#             }

#             feature_file = out_dir / f"feature_{feat_id}_complete.json"
#             feature_file.write_text(json.dumps(feature_dict, indent=2))

#         logger.info(
#             f"Wrote {len(use_feats)}/{len(all_feats)} feature dictionaries "
#             f"for layer {layer} to: {out_dir}"
#         )

#     def _prepare_encoder_subset(self, *, index_list: List[int], dtype: torch.dtype) -> None:
#         """
#         Prepare (and cache on GPU) only the encoder parameters needed for `index_list`.
#         """
#         if not index_list:
#             raise ValueError("index_list must be non-empty.")

#         # Slice on CPU (CLT is on CPU), then move to GPU
#         idx = torch.as_tensor(index_list, dtype=torch.long, device="cpu")

#         W_sub = self.clt.W_enc[:, :, idx]          # [L, d_in, F]
#         b_sub = self.clt.b_enc[:, idx]             # [L, F]
#         lt_sub = self.clt.log_threshold[:, idx]    # [L, F]

#         self._enc_index_list = list(index_list)
#         self._W_enc_sub = W_sub.to(self.device, dtype=dtype, non_blocking=True)
#         self._b_enc_sub = b_sub.to(self.device, dtype=dtype, non_blocking=True)
#         self._threshold_sub = torch.exp(lt_sub).to(self.device, dtype=dtype, non_blocking=True)
#         self._bandwidth = self.clt.bandwidth

#     @torch.no_grad()
#     def _encode_subset(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x:   [B, L, d_in]
#         out: [B, L, F]
#         """
#         if not hasattr(self, "_W_enc_sub"):
#             raise RuntimeError("Call _prepare_encoder_subset(index_list=..., dtype=...) before _encode_subset().")

#         hidden_pre = torch.einsum("bld,ldf->blf", x, self._W_enc_sub) + self._b_enc_sub  # [B, L, F]
#         thresh = self._threshold_sub                                       # [L, F]
#         feat_act = JumpReLU.apply(hidden_pre, thresh, self._bandwidth)                    # [B, L, F]
#         return feat_act

#     def _to_sequences(
#         self, tokens_cpu: torch.Tensor, feat_act_gpu: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         tokens_cpu: [B] CPU
#         feat_act_gpu: [B, L, F] GPU
#         returns:
#           tokens_seq_cpu: [B_seq, ctx] CPU
#           acts_seq_gpu:   [B_seq, ctx, L, F] GPU
#         """
#         B = int(feat_act_gpu.shape[0])
#         excess = B % self.ctx
#         if excess != 0:
#             feat_act_gpu = feat_act_gpu[:-excess]
#             tokens_cpu = tokens_cpu[:-excess]
#             B = int(feat_act_gpu.shape[0])

#         if B == 0:
#             empty_tokens = tokens_cpu.new_zeros((0, self.ctx))
#             empty_acts = feat_act_gpu.new_zeros((0, self.ctx, feat_act_gpu.shape[1], feat_act_gpu.shape[2]))
#             return empty_tokens, empty_acts

#         B_seq = B // self.ctx
#         tokens_seq = tokens_cpu.view(B_seq, self.ctx)  # CPU
#         acts_seq = feat_act_gpu.view(B_seq, self.ctx, feat_act_gpu.shape[1], feat_act_gpu.shape[2])  # GPU
#         return tokens_seq, acts_seq

#     def _init_topk_state(
#         self,
#         *,
#         n_layers: int,
#         F: int,
#         top_k: int,
#         ctx: int,
#         dtype: torch.dtype,
#         device: torch.device,
#     ) -> Dict[str, Any]:

#         neg_inf = torch.full((top_k, F), float("-inf"), device=device, dtype=dtype)
#         tok_init = torch.zeros((top_k, F, ctx), device=device, dtype=torch.int32)
#         act_init = torch.zeros((top_k, F, ctx), device=device, dtype=dtype)

#         return {
#             "top_vals_by_layer": [neg_inf.clone() for _ in range(n_layers)],
#             "top_tokens_by_layer": [tok_init.clone() for _ in range(n_layers)],
#             "top_acts_by_layer": [act_init.clone() for _ in range(n_layers)],
#             "sum_pos_by_layer": [torch.zeros((F,), device=device, dtype=dtype) for _ in range(n_layers)],
#             "count_pos_by_layer": [torch.zeros((F,), device=device, dtype=torch.long) for _ in range(n_layers)],
#         }

#     @torch.no_grad()
#     def _update_state_with_batch(
#         self,
#         *,
#         state: Dict[str, Any],
#         tokens_seq_cpu: torch.Tensor,  # [B_seq, ctx] CPU
#         acts_seq_gpu: torch.Tensor,    # [B_seq, ctx, L, F] GPU
#         top_k: int,
#     ) -> None:
#         """
#         For each layer:
#           batch_acts: [B_seq, F, ctx]
#           batch_vals: [B_seq, F] (max over ctx)
#           merge with global [K, F] via topk(cat) and gather tokens/acts accordingly.
#         """
#         B_seq = int(tokens_seq_cpu.shape[0])
#         if B_seq == 0:
#             return

#         tokens_seq_gpu = tokens_seq_cpu.to(self.device, dtype=torch.int32)  # [B_seq, ctx]
#         L = int(acts_seq_gpu.shape[2])
#         F = int(acts_seq_gpu.shape[3])
#         ctx = int(acts_seq_gpu.shape[1])

#         # expand tokens across features
#         batch_tokens = tokens_seq_gpu[:, None, :].expand(B_seq, F, ctx)  # [B_seq, F, ctx]

#         for layer in range(L):
#             # [B_seq, ctx, F] -> [B_seq, F, ctx]
#             batch_acts = acts_seq_gpu[:, :, layer, :].permute(0, 2, 1).contiguous()

#             # [B_seq, F]
#             batch_vals = batch_acts.max(dim=2).values

#             # stats
#             pos = batch_vals > 0
#             state["sum_pos_by_layer"][layer] += (batch_vals * pos).sum(dim=0)
#             state["count_pos_by_layer"][layer] += pos.sum(dim=0)

#             # merge
#             top_vals = state["top_vals_by_layer"][layer]           # [K, F]
#             top_tokens = state["top_tokens_by_layer"][layer]       # [K, F, ctx]
#             top_acts = state["top_acts_by_layer"][layer]           # [K, F, ctx]

#             merged_vals = torch.cat([top_vals, batch_vals], dim=0)          # [K+B, F]
#             merged_tokens = torch.cat([top_tokens, batch_tokens], dim=0)    # [K+B, F, ctx]
#             merged_acts = torch.cat([top_acts, batch_acts], dim=0)          # [K+B, F, ctx]

#             new_vals, pos_idx = torch.topk(merged_vals, k=top_k, dim=0)     # [K, F], [K, F]
#             gather_idx = pos_idx[:, :, None].expand(top_k, F, ctx)          # [K, F, ctx]

#             new_tokens = torch.gather(merged_tokens, dim=0, index=gather_idx)
#             new_acts = torch.gather(merged_acts, dim=0, index=gather_idx)

#             state["top_vals_by_layer"][layer] = new_vals
#             state["top_tokens_by_layer"][layer] = new_tokens
#             state["top_acts_by_layer"][layer] = new_acts

#         del tokens_seq_gpu, batch_tokens

#     def _state_to_cpu_payload(
#         self,
#         *,
#         state: Dict[str, Any],
#         index_list: List[int],
#         worker_id: str,
#         top_k: int,
#         dtype: torch.dtype,
#     ) -> Dict[str, Any]:
#         return {
#             "worker_id": worker_id,
#             "index_list": index_list,
#             "top_k": int(top_k),
#             "ctx": int(self.ctx),
#             "dtype": str(dtype),
#             "top_vals_by_layer": [t.detach().cpu() for t in state["top_vals_by_layer"]],
#             "top_tokens_by_layer": [t.detach().cpu() for t in state["top_tokens_by_layer"]],
#             "top_acts_by_layer": [t.detach().cpu() for t in state["top_acts_by_layer"]],
#             "sum_pos_by_layer": [t.detach().cpu() for t in state["sum_pos_by_layer"]],
#             "count_pos_by_layer": [t.detach().cpu() for t in state["count_pos_by_layer"]],
#         }

#     def _get_top_activating_tokens_from_sequences(
#         self,
#         *,
#         sequences: List[Dict[str, Any]],
#         tokenizer,
#         top_k: int = 3,
#         threshold_ratio: float = 0.6,
#     ) -> List[Dict[str, Any]]:
#         if not sequences:
#             return []

#         all_tokens = []
#         all_activations = []
#         for s in sequences:
#             tks = s["tokens"][1:]
#             acts = s["activations"][1:]
#             all_tokens.append(tks)
#             all_activations.append(acts)

#         combined_tokens = torch.cat(all_tokens)
#         combined_activations = torch.cat(all_activations)
#         if combined_activations.numel() == 0:
#             return []

#         threshold = float(combined_activations.max().item()) * threshold_ratio
#         mask = combined_activations > threshold

#         activating_tokens = combined_tokens[mask].tolist()
#         activating_values = combined_activations[mask].tolist()

#         token_stats: Dict[int, Dict[str, float]] = {}
#         for token_id, activation in zip(activating_tokens, activating_values):
#             if token_id not in token_stats:
#                 token_stats[token_id] = {"count": 0.0, "total_activation": 0.0}
#             token_stats[token_id]["count"] += 1.0
#             token_stats[token_id]["total_activation"] += float(activation)

#         ranking: List[Dict[str, Any]] = []
#         for token_id, stats in token_stats.items():
#             avg_activation = stats["total_activation"] / stats["count"]
#             token_text = tokenizer.decode([int(token_id)])
#             ranking.append(
#                 {
#                     "token": token_text,
#                     "token_id": int(token_id),
#                     "frequency": int(stats["count"]),
#                     "average_activation": float(avg_activation),
#                 }
#             )

#         ranking.sort(key=lambda x: x["frequency"], reverse=True)
#         return ranking[:top_k]


# def highlight_activations(
#     tokens: torch.Tensor,
#     activations: torch.Tensor,
#     tokenizer,
#     threshold_ratio: float = 0.6,
# ) -> str:
#     assert len(tokens) == len(activations), "Token and activation lengths must match"

#     max_act = activations.max().item()
#     threshold = max_act * threshold_ratio
#     str_tokens = tokenizer.convert_ids_to_tokens(tokens)

#     highlight_mask = activations > threshold

#     def contains_chinese(text: str) -> bool:
#         return any("\u4e00" <= char <= "\u9fff" for char in text)

#     full_text_sample = tokenizer.convert_tokens_to_string(str_tokens[: min(10, len(str_tokens))])
#     is_chinese_text = contains_chinese(full_text_sample)

#     extended_mask = highlight_mask.clone()
#     if not is_chinese_text:
#         for i in range(len(str_tokens)):
#             if highlight_mask[i]:
#                 j = i - 1
#                 while (
#                     j >= 0
#                     and not str_tokens[j].startswith(("▁", " ", "Ġ"))
#                     and str_tokens[j] not in ["<|endoftext|>", "</s>", "<s>", "[CLS]", "[SEP]"]
#                 ):
#                     extended_mask[j] = True
#                     j -= 1

#                 j = i + 1
#                 while (
#                     j < len(str_tokens)
#                     and not str_tokens[j].startswith(("▁", " ", "Ġ"))
#                     and str_tokens[j] not in ["<|endoftext|>", "</s>", "<s>", "[CLS]", "[SEP]"]
#                 ):
#                     extended_mask[j] = True
#                     j += 1

#     marked_tokens: List[str] = []
#     in_highlight = False
#     for tok, is_high in zip(str_tokens, extended_mask):
#         if is_high and not in_highlight:
#             marked_tokens.append("<<")
#             in_highlight = True
#         elif (not is_high) and in_highlight:
#             marked_tokens.append(">>")
#             in_highlight = False
#         marked_tokens.append(tok)

#     if in_highlight:
#         marked_tokens.append(">>")

#     segments: List[str] = []
#     buffer: List[str] = []
#     for tok in marked_tokens:
#         if tok in {"<<", ">>"}:
#             if buffer:
#                 segments.append(tokenizer.convert_tokens_to_string(buffer))
#                 buffer = []
#             segments.append(tok)
#         else:
#             buffer.append(tok)

#     if buffer:
#         segments.append(tokenizer.convert_tokens_to_string(buffer))

#     result = ""
#     i = 0
#     while i < len(segments):
#         seg = segments[i]
#         if seg == "<<":
#             if result and not result.endswith(" "):
#                 result += " "
#             result += "<<"
#             i += 1
#             if i < len(segments):
#                 result += segments[i].lstrip()
#                 i += 1
#             if i < len(segments) and segments[i] == ">>":
#                 result += ">>"
#                 i += 1
#         else:
#             if result and not result.endswith(" "):
#                 result += " "
#             result += seg
#             i += 1

#     return result
