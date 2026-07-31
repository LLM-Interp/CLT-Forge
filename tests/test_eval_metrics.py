import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from clt_forge.training.eval_metrics import CLTEvaluator

class MockCLTConfig:
    def __init__(self):
        self.cross_layer_decoders = False
        self.model_name = "mock_model"
        self.model_from_pretrained_kwargs = {}

class MockCLT:
    def __init__(self, n_layers, d_in, d_latent):
        self.N_layers = n_layers
        self.d_in = d_in
        self.d_latent = d_latent
        self.local_d_latent = d_latent
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.cfg = MockCLTConfig()
        
    def encode(self, act_in, layer=None):
        # mock encode: just return act_in expanded to d_latent, and act_in
        shape = list(act_in.shape)
        shape[-1] = self.d_latent
        feat_act = torch.ones(*shape, device=self.device, dtype=self.dtype)
        return feat_act, act_in
        
    def decode(self, z, layer=None):
        # mock decode: return z reduced to d_in
        shape = list(z.shape)
        shape[-1] = self.d_in
        return torch.ones(*shape, device=self.device, dtype=self.dtype) * 0.5

class MockHookedTransformer(nn.Module):
    def __init__(self, hook_names):
        super().__init__()
        self.hook_names = hook_names
        
    def forward(self, tokens, return_type="logits"):
        # Returns fake logits
        batch_size, seq_len = tokens.shape
        # d_vocab = 100
        return torch.ones((batch_size, seq_len, 100))
        
    def run_with_hooks(self, tokens, return_type="logits", fwd_hooks=[]):
        # Execute hooks manually on dummy activations
        batch_size, seq_len = tokens.shape
        dummy_act = torch.ones((batch_size, seq_len, 12)) # d_in = 12
        for name, hook in fwd_hooks:
            # Just call the hook
            dummy_act = hook(dummy_act, Mock())
        return self.forward(tokens, return_type=return_type)
        
    def run_with_cache(self, tokens, names_filter=None):
        batch_size, seq_len = tokens.shape
        logits = self.forward(tokens)
        cache = {name: torch.ones((batch_size, seq_len, 12)) for name in self.hook_names}
        return logits, cache
        
    def eval(self):
        pass

@patch("clt_forge.training.eval_metrics.HookedTransformer.from_pretrained")
def test_evaluate_replacement_and_kl(mock_from_pretrained):
    clt = MockCLT(n_layers=2, d_in=12, d_latent=24)
    hook_names = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    
    # Setup mock
    mock_model = MockHookedTransformer(hook_names)
    mock_from_pretrained.return_value = mock_model
    
    evaluator = CLTEvaluator(clt, clt.cfg)
    
    tokens = torch.randint(0, 100, (2, 5)) # B=2, seq_len=5
    
    metrics = evaluator.evaluate_replacement_and_kl(tokens)
    
    assert "replacement_score" in metrics
    assert "kl_divergence" in metrics
    assert "M_clean" in metrics
    assert "M_zero" in metrics
    assert "M_transcoder" in metrics

@patch("clt_forge.training.eval_metrics.HookedTransformer.from_pretrained")
def test_evaluate_sparsity_and_l0(mock_from_pretrained):
    clt = MockCLT(n_layers=2, d_in=12, d_latent=24)
    hook_names = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    
    mock_model = MockHookedTransformer(hook_names)
    mock_from_pretrained.return_value = mock_model
    
    evaluator = CLTEvaluator(clt, clt.cfg)
    
    tokens = torch.randint(0, 100, (2, 5))
    metrics = evaluator.evaluate_sparsity_and_l0(tokens, tau=0.5)
    
    assert "l0_norm" in metrics
    assert "pruned_l0_norm" in metrics
    assert "training_sparsity" in metrics
    assert "pruning_sparsity" in metrics

@patch("clt_forge.training.eval_metrics.HookedTransformer.from_pretrained")
def test_evaluate_activation_density(mock_from_pretrained):
    clt = MockCLT(n_layers=2, d_in=12, d_latent=24)
    hook_names = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    
    mock_model = MockHookedTransformer(hook_names)
    mock_from_pretrained.return_value = mock_model
    
    evaluator = CLTEvaluator(clt, clt.cfg)
    
    tokens = torch.randint(0, 100, (2, 5))
    metrics = evaluator.evaluate_activation_density(tokens)
    
    assert "activation_density_per_feature" in metrics
    assert "mean_activation_density" in metrics
    assert "dead_features_count" in metrics
    assert "dead_features_ratio" in metrics
