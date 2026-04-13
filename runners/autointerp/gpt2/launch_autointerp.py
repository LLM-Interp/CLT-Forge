import argparse
from pathlib import Path
import torch
import os

from clt_forge.config.autointerp_config import AutoInterpConfig
from clt_forge.autointerp.pipeline import AutoInterp

import clt_forge
STORAGE_ROOT = Path(clt_forge.__file__).resolve().parents[2] / "storage"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_config() -> AutoInterpConfig:
    MODEL = "gpt2"
    d_in = 768

    return AutoInterpConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        model_name=MODEL,
        clt_path=str(STORAGE_ROOT / "checkpoints" / MODEL / "d1s3fw30/middle_22137856"),
        latent_cache_path=str(STORAGE_ROOT / "autointerp" / MODEL),
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        context_size=16,
        total_autointerp_tokens=32 * 16 * 4096,
        train_batch_size_tokens=4 * 4096,
        n_batches_in_buffer=32,
        store_batch_size_prompts=32,
        d_in=d_in,
        storage_backend="parquet",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    args = parser.parse_args()
    job_id = args.job_id
    total_jobs = args.total_jobs

    print(f"[Job {job_id}/{total_jobs}] Starting...")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    cfg = build_config()
    autointerp = AutoInterp(cfg)

    autointerp.run(
        job_id=job_id,
        total_jobs=total_jobs,
        save_dir=Path(cfg.latent_cache_path),
        generate_explanations=False,
    )

    print(f"[Job {job_id}] DONE")


if __name__ == "__main__":
    main()
