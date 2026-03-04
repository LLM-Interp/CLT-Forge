from __future__ import annotations

from typing import Optional

from wandb.util import generate_id
import torch.distributed as dist


def broadcast_object(obj, src: int = 0):
    """
    Broadcast a python object from src to all ranks, returns the object on every rank.
    Requires dist initialized. Use only in distributed mode.
    """
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def get_synced_wandb_id(rank: int) -> str:
    """
    Generates a wandb id on rank 0 and broadcasts it to all ranks.
    If not in distributed mode, just returns a local id.
    """
    if dist.is_available() and dist.is_initialized():
        wandb_id: Optional[str] = generate_id() if rank == 0 else None
        return broadcast_object(wandb_id, src=0)
    return generate_id()
