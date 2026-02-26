from pydantic import BaseModel
from typing import TypeVar, Any, Dict, Optional

T = TypeVar("T", bound=BaseModel)


class AutoInterpConfig(BaseModel):
    # ---- Device ----
    device: str = "cuda"
    dtype: str = "float32"

    # ---- Model ----
    model_class_name: str = "HookedTransformer"
    model_name: str = "gpt2"
    model_kwargs: Optional[Dict[str, Any]] = None
    model_from_pretrained_kwargs: Optional[Dict[str, Any]] = None
    d_in: int = 768

    # ---- Dataset ----
    dataset_path: str = ""          # HuggingFace path or local path
    is_dataset_tokenized: bool = True
    split: str = "train"            # dataset split passed to load_dataset
    disk: bool = False              # use load_from_disk instead of load_dataset
    is_multilingual_split_dataset: bool = False  # only for multilingual setups

    # ---- ActivationsStore ----
    context_size: int = 128
    n_batches_in_buffer: int = 20
    store_batch_size_prompts: int = 32
    n_train_batch_per_buffer: Optional[int] = None
    cached_activations_path: Optional[str] = None
    train_batch_size_tokens: int = 1024

    # ---- CLT ----
    clt_path: str = "checkpoints/gpt2"

    # ---- AutoInterp ----
    total_autointerp_tokens: int = 10_000_000
    latent_cache_path: Optional[str] = None
    generate_explanations: bool = False   # set True to run vLLM explanations

    # ---- vLLM (only used when generate_explanations=True) ----
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_max_tokens: int = 3000

    # ---- ActivationsStore norm estimate ----
    n_batches_for_norm_estimate: int = 10

    # ---- Distributed — all False for autointerp, but ActivationsStore reads them ----
    ddp: bool = False
    fsdp: bool = False
    feature_sharding: bool = False

    # Properties mirroring CLTTrainingRunnerConfig so ActivationsStore works unchanged
    @property
    def is_distributed(self) -> bool:
        return self.ddp or self.fsdp

    @property
    def is_sharded(self) -> bool:
        return self.feature_sharding

    @property
    def uses_process_group(self) -> bool:
        return self.is_distributed or self.is_sharded

    def to_dict(self, *, exclude_none: bool = True, **kw) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "AutoInterpConfig":
        return cls.model_validate(cfg_dict)
