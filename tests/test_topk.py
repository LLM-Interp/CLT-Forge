import pytest
import torch
from clt_forge.config import CLTConfig
from clt_forge.clt import CLT

def get_base_config(activation_fn="topk", k=4, cross_layer_decoders=True):
    return CLTConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="dummy",
        d_in=16,
        d_latent=32,
        n_layers=2,
        jumprelu_bandwidth=1.0,
        jumprelu_init_threshold=0.03,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=cross_layer_decoders,
        context_size=32,
        l0_coefficient=1e-3,
        activation_fn=activation_fn,
        k=k
    )

@pytest.mark.parametrize("activation_fn", ["topk", "groupmax"])
@pytest.mark.parametrize("cross_layer_decoders", [True, False])
def test_topk_groupmax_sparsity(activation_fn, cross_layer_decoders):
    k = 4
    cfg = get_base_config(activation_fn=activation_fn, k=k, cross_layer_decoders=cross_layer_decoders)
    clt = CLT(cfg)
    
    # Check requires_grad is False for log_threshold
    assert clt.log_threshold.requires_grad is False
    
    # Input tensor shape: [B, N_layers, d_in]
    B = 5
    acts_in = torch.randn(B, cfg.n_layers, cfg.d_in)
    
    feat_act, hidden_pre = clt.encode(acts_in)
    
    assert feat_act.shape == (B, cfg.n_layers, cfg.d_latent)
    
    # Assert exactly k active features per token/layer
    for b in range(B):
        for l in range(cfg.n_layers):
            active_count = (feat_act[b, l] > 0).sum().item()
            assert active_count == k, f"Expected exactly {k} active features, got {active_count}"
            
    # For groupmax, assert exactly 1 active feature per group
    if activation_fn == "groupmax":
        group_size = cfg.d_latent // k
        for b in range(B):
            for l in range(cfg.n_layers):
                for g in range(k):
                    group_acts = feat_act[b, l, g*group_size:(g+1)*group_size]
                    group_active_count = (group_acts > 0).sum().item()
                    assert group_active_count == 1, f"Expected exactly 1 active feature in group {g}, got {group_active_count}"

@pytest.mark.parametrize("activation_fn", ["topk", "groupmax"])
def test_topk_groupmax_loss_and_gradients(activation_fn):
    cfg = get_base_config(activation_fn=activation_fn, k=4)
    clt = CLT(cfg)
    
    B = 5
    acts_in = torch.randn(B, cfg.n_layers, cfg.d_in)
    acts_out = torch.randn(B, cfg.n_layers, cfg.d_in)
    
    # Compute loss
    loss_metrics = clt.loss(acts_in, acts_out, l0_coef=1e-3, df_coef=1e-5)
    
    # Assert L0 loss and dead feature loss are exactly 0
    assert loss_metrics.l0_loss.item() == 0.0
    assert loss_metrics.dead_feature_loss.item() == 0.0
    assert loss_metrics.mse_loss.item() > 0.0
    
    total_loss = loss_metrics.mse_loss + loss_metrics.l0_loss + loss_metrics.dead_feature_loss
    total_loss.backward()
    
    # Verify gradients flow to weights
    assert clt.W_enc.grad is not None
    assert clt.b_enc.grad is not None
    assert clt.W_dec.grad is not None
    
    # Ensure no gradients flow to log_threshold
    assert clt.log_threshold.grad is None or (clt.log_threshold.grad == 0).all()

@pytest.mark.parametrize("activation_fn", ["topk", "groupmax"])
def test_topk_groupmax_b_enc_initialization(activation_fn):
    # Use larger d_latent to test in-bounds target_idx
    cfg = CLTConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="dummy",
        d_in=16,
        d_latent=16384,
        n_layers=2,
        jumprelu_bandwidth=1.0,
        jumprelu_init_threshold=0.03,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=True,
        context_size=32,
        l0_coefficient=1e-3,
        activation_fn=activation_fn,
        k=4
    )
    clt = CLT(cfg)
    
    # Assert initial b_enc is all zeros
    assert (clt.b_enc == 0).all()
    
    B = 300
    # hidden_pre needs to be shape [B, N_layers, d_latent]
    hidden_pre = torch.randn(B, cfg.n_layers, cfg.d_latent)
    
    clt._initialize_b_enc(hidden_pre)
    
    # Assert b_enc is no longer all zeros
    assert not (clt.b_enc == 0).all()
