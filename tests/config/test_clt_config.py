import json
from circuitlab.config.clt_config import CLTConfig

# @pytest.fixture(scope="session", autouse=True)
# def stub_torch(monkeypatch):
#     # minimalist torch substitute
#     torch_stub = types.ModuleType("torch")
#     torch_stub.float32 = "float32"
#     sys.modules["torch"] = torch_stub
#     yield

def make_cfg():
    return CLTConfig(
        model_name="gpt2",
        device="cpu",
        dtype="float32",
        seed=42,
        debug=False, 
        d_in=128,
        d_latent=256,
        n_layers=12,
        jumprelu_bandwidth=0.5,
        jumprelu_init_threshold=1.0,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=True,
        context_size=64,
        l0_coefficient=0.1
    )

def test_round_trip():
    cfg = make_cfg()
    clone = cfg.from_dict(cfg.to_dict())
    assert clone == cfg

def test_to_dict_json_safe():
    cfg = make_cfg()
    json.dumps(cfg.to_dict())  # should not raise
