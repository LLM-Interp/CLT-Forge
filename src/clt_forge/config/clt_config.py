from pydantic import BaseModel
from typing import TypeVar
from typing import Dict, Any, Literal, Optional

T = TypeVar("T", bound=BaseModel)


class CLTConfig(BaseModel): 
    # -----MISC------------------------------
    device : str 
    dtype: str
    seed: int 
    model_name: str
    debug: bool = False

    # -----CLT parameters---------------------
    d_in: int 
    d_latent: int
    n_layers: int
    jumprelu_bandwidth: float
    jumprelu_init_threshold: float
    normalize_decoder: bool
    dead_feature_window: int
    cross_layer_decoders: bool
    context_size: int
    functional_loss: Optional[str] = None

    # -----Sparsity---------------------------
    l0_coefficient: float

    # -----Sparse decode----------------------
    # auto mode switches dense->gather once sparsity >= sparse_threshold. The
    # default is the empirically measured crossover: on a 3090 (bf16, B=1024,
    # d_latent=24576, fwd+bwd) gather only beats dense above ~99.95% sparsity
    # (see scripts/benchmark_sparse_decode.py). Below that, gather is slower and
    # can OOM (it materializes an [nnz, d_in] intermediate), so switch late.
    sparse_decode: Literal["dense", "gather", "csr", "auto"] = "dense"
    sparse_threshold: float = 0.9995

    # -----DDP--------------------------------
    ddp: bool = False
    fsdp: bool = False
    feature_sharding: bool = False

    # one‑liner to get a json‑safe dict
    def to_dict(self, *, exclude_none: bool = True,**kw) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "CLTConfig":
        """
        counterpart to `to_dict` – parses dtype string back to torch.dtype
        """
        return cls.model_validate(cfg_dict)

    @property
    def is_distributed(self) -> bool:
        return self.ddp or self.fsdp

    @property
    def is_sharded(self) -> bool:
        return self.feature_sharding

    @property
    def uses_process_group(self) -> bool:
        return self.is_distributed or self.is_sharded
