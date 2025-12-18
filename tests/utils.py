from featflow.config import CLTTrainingRunnerConfig, AutoInterpConfig
from typing import Any
import torch 
from featflow.training.activations_store import ActivationsStore

TINYSTORIES_MODEL = "tiny-stories-1M"
NEEL_NANDA_C4_10K_DATASET = "data/NeelNanda_c4_10k_tokenized"

def build_clt_training_runner_cfg(**kwargs: Any) -> CLTTrainingRunnerConfig:
    """
    Helper to create a mock instance of CLTTrainingRunnerConfig.
    """
    mock_config_dict = { 
        "device": "cpu", 
        "model_name": TINYSTORIES_MODEL, 
        "dataset_path": NEEL_NANDA_C4_10K_DATASET, 
        "d_in": 12,
        "l0_coefficient": 2e-3,
        "lr": 1e-2,
        "d_latent": 4,
        "train_batch_size_tokens": 4,
        "context_size": 4,
        "n_batches_in_buffer": 4,
        "total_training_tokens": 100,
        "l0_warm_up_steps": 1,
        "lr_decay_steps": 1,
        "lr_warm_up_steps": 1,
        "store_batch_size_prompts": 4,
        "log_to_wandb": False,
        "wandb_project": "test_project",
        "wandb_entity": "test_entity",
        "wandb_log_frequency": 5,
        "checkpoint_path": "test/checkpoints",
        "n_checkpoints": 1,
        "dead_feature_window": 1
    }

    for key, value in kwargs.items():
        mock_config_dict[key] = value

    print(mock_config_dict)

    mock_config = CLTTrainingRunnerConfig(**mock_config_dict)

    # reset checkpoint path (as we add an id to each each time)
    mock_config.checkpoint_path = kwargs.get("checkpoint_path", "test/checkpoints")

    return mock_config

class FakeActivationsStore(ActivationsStore):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x, self.y

def build_autointerp_cfg(**kwargs: Any) -> AutoInterpConfig:
    """
    Helper to create a mock instance of AutoInterp.
    """
    mock_config_dict = { 
        "device" : "cuda",
        "model_name" : "roneneldan/TinyStories-33M", 
        "context_size" : 32,
        "total_autointerp_tokens" : 10_000,
        "vllm_model" : "meta-llama/Llama-3.1-8B-Instruct",
        "train_batch_size_tokens": 100,
        "n_batches_in_buffer": 10,
        "store_batch_size_prompts": 10, 
        "d_in": 768
    }

    for key, value in kwargs.items():
        mock_config_dict[key] = value

    print(mock_config_dict)

    mock_config = AutoInterpConfig(**mock_config_dict)

    return mock_config
