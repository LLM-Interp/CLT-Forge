import json
import pytest
from featflow.config.clt_config import CLTConfig
from featflow.config.clt_training_runner_config import CLTTrainingRunnerConfig
import torch 

def make_cfg(**kwargs):
    return CLTTrainingRunnerConfig(**kwargs)

def test_device_fallback_cpu():
    not_device = "mps" if torch.cuda.is_available() else "cuda"
    cfg = make_cfg(device=not_device)
    assert cfg.device == "cpu"

def test_device_cuda_ok():
    with pytest.raises(ValueError):
        make_cfg(device="cuda:0")

def test_latent_expansion_mutual_exclusive():
    with pytest.raises(ValueError):
        make_cfg(d_latent=256, expansion_factor=4)

def test_latent_computed_from_expansion():
    cfg = make_cfg(d_in=128, expansion_factor=8)
    assert cfg.d_latent == 1024

def test_latent_default_expansion():
    # default expansion_factor should be 16 when neither given
    cfg = make_cfg(d_in=512)
    assert cfg.d_latent == 512 * 16

def test_to_dict_json_serialisable():
    cfg = make_cfg()
    dumped = cfg.to_dict()
    # should raise no TypeError
    json.dumps(dumped)

def test_create_sub_config_success():
    cfg = make_cfg(d_in=128, d_latent=256)
    sub = cfg.create_sub_config(
        CLTConfig,
        n_layers=12
    )
    assert sub.n_layers == 12
    assert sub.d_in == cfg.d_in
    assert sub.d_latent == cfg.d_latent

def test_create_sub_config_requires_n_layers():
    cfg = make_cfg(d_in=128, d_latent=256)
    with pytest.raises(ValueError):
        cfg.create_sub_config(CLTConfig)
