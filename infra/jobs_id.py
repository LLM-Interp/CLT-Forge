from typing import Tuple

def compute_job_split_range(
    *,
    job_id: int,
    total_jobs: int,
    total_splits: int,
) -> Tuple[int, int]:
    if not (0 <= job_id < total_jobs):
        raise ValueError(
            f"job_id must be in [0, {total_jobs - 1}]"
        )

    splits_per_job = total_splits // total_jobs

    start = job_id * splits_per_job
    end = (job_id + 1) * splits_per_job

    # last job takes remainder
    if job_id == total_jobs - 1:
        end = total_splits

    return start, end
