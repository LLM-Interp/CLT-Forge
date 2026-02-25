# import os
# import sys
# import torch
# from pathlib import Path

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from circuitlab.config.autointerp_config import AutoInterpConfig
# from circuitlab.autointerp.pipeline import AutoInterp

# def main():
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     print(f"CUDA device count: {torch.cuda.device_count()}")

#     job_id = int(sys.argv[1])
#     total_jobs = int(sys.argv[2])

#     d_in = 768
#     expansion_factor = 32
#     d_latent = d_in * expansion_factor

#     clt_path = "/fast/fdraye/data/featflow/cache/checkpoints/gpt2/d1s3fw30/middle_22137856"

#     autointerp_cfg = {
#         "device": "cuda",  # 👈 NOW GPU matters
#         "model_name": "gpt2",
#         "clt_path": clt_path,
#         "latent_cache_path": "/fast/fdraye/data/featflow/cache/gpt2",
#         "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
#         "context_size": 16,
#         "total_autointerp_tokens": 12 * (16 * 4096),
#         "train_batch_size_tokens": 4096,
#         "n_batches_in_buffer": 32,
#         "store_batch_size_prompts": 32,
#         "d_in": d_in,
#     }

#     print("Creating AutoInterpConfig...", flush=True)
#     cfg = AutoInterpConfig(**autointerp_cfg)

#     print("Initializing AutoInterp...", flush=True)
#     autointerp = AutoInterp(cfg)

#     features_per_job = d_latent // total_jobs
#     start_idx = job_id * features_per_job
#     end_idx = start_idx + features_per_job if job_id < total_jobs - 1 else d_latent
#     index_list = list(range(start_idx, end_idx))

#     print(
#         f"[Job {job_id}/{total_jobs}] "
#         f"Processing features {start_idx} → {end_idx - 1} "
#         f"({len(index_list)} features)",
#         flush=True,
#     )

#     autointerp.run(
#         worker_id=f"{job_id}",
#         index_list=index_list,
#         top_k=100,
#         save_dir=Path(cfg.latent_cache_path),
#     )

#     print(f"[Job {job_id}] DONE", flush=True)

# if __name__ == "__main__":
#     main()
