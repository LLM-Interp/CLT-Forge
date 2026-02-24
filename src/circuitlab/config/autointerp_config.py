from pydantic import BaseModel
from typing import TypeVar
from typing import Any, Dict, Optional

T = TypeVar("T", bound=BaseModel)

class AutoInterpConfig(BaseModel): 
    device : str = "cuda"
    dtype: str = "float32"
    # model 
    model_class_name: str = "HookedTransformer"
    model_name: str = "gpt2"
    model_kwargs: Optional[Dict[str, Any]] = None
    model_from_pretrained_kwargs: Optional[Dict[str, Any]] = None
    d_in: int = 768
    # data
    dataset_path: str = "" # Hugging face path
    is_dataset_tokenized: bool = True
    context_size: int = 128
    # activations store
    n_batches_in_buffer: int = 20
    store_batch_size_prompts: int = 32
    n_train_batch_per_buffer: Optional[int] = None
    cached_activations_path: Optional[str] = None
    train_batch_size_tokens: int = 1024
    # clt 
    clt_path: str = "checkpoints/gpt2"
    # autointerp
    total_autointerp_tokens: int = 10_000_000
    latent_cache_path: Optional[str] = None
    n_splits: int = 5
    n_chunks: int
    vllm_model: str = "meta-llama/llama-3.1-8b-instruct"
    vllm_max_tokens: int = 3000
    chunk_list: Optional[list[int]] = None
    # ddp 
    ddp: bool = False
    fsdp: bool = False
    
    # one‑liner to get a json‑safe dict
    def to_dict(self, *, exclude_none: bool = True,**kw) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "AutoInterpConfig":
        """
        counterpart to `to_dict` – parses dtype string back to torch.dtype
        """
        return cls.model_validate(cfg_dict)
