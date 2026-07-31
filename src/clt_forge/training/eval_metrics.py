import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from transformer_lens import HookedTransformer

class CLTEvaluator:
    def __init__(self, clt, cfg):
        """
        clt: The CLT model.
        cfg: The CLTTrainingRunnerConfig instance.
        """
        self.clt = clt
        self.cfg = cfg
        
        from clt_forge import logger
        logger.info(f"Instantiating base model {cfg.model_name} in CLTEvaluator")
        self.base_model = HookedTransformer.from_pretrained(
            cfg.model_name,
            device=clt.device,
            **(cfg.model_from_pretrained_kwargs if cfg.model_from_pretrained_kwargs else {})
        )
        self.base_model.eval()
        
        # Assuming residual stream hook names match layers directly
        self.hook_names = [f"blocks.{i}.hook_resid_post" for i in range(clt.N_layers)]

    @torch.no_grad()
    def evaluate_replacement_and_kl(
        self, 
        tokens: torch.Tensor, 
        layers: Optional[List[int]] = None, 
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Calculates the Replacement Score and KL Divergence for the specified layers.
        If layers is None, evaluates all layers.
        """
        if layers is None:
            layers = list(range(self.clt.N_layers))
            
        if metric_fn is None:
            def default_metric(logits, tokens):
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tokens[:, 1:].contiguous()
                return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
            metric_fn = default_metric

        # 1. Clean Run
        clean_logits = self.base_model(tokens, return_type="logits")
        M_clean = metric_fn(clean_logits, tokens)
        P_clean = F.log_softmax(clean_logits, dim=-1)

        # 2. Zero (Ablated) Run
        def zero_hook_fn(act, hook):
            return torch.zeros_like(act)
            
        zero_hooks = [(self.hook_names[i], zero_hook_fn) for i in layers]
        zero_logits = self.base_model.run_with_hooks(tokens, return_type="logits", fwd_hooks=zero_hooks)
        M_zero = metric_fn(zero_logits, tokens)

        # 3. Transcoder (Replaced) Run
        replace_hooks = []
        
        if self.clt.cfg.cross_layer_decoders:
            # We need to hook all layers up to max(layers) to accumulate correctly
            max_layer = max(layers)
            layers_to_hook = list(range(max_layer + 1))
        else:
            layers_to_hook = layers

        B = tokens.size(0)
        seq_len = tokens.size(1)
        recon_acc = torch.zeros(B * seq_len, self.clt.N_layers, self.clt.d_in, device=self.clt.device, dtype=self.clt.dtype)
        
        for layer_idx in layers_to_hook:
            def get_replace_hook(l_idx, replace):
                def replace_hook_fn(act, hook):
                    act_flat = act.view(-1, self.clt.d_in).to(self.clt.dtype)
                    
                    z_i, _ = self.clt.encode(act_flat, layer=l_idx)
                    out_i = self.clt.decode(z_i, layer=l_idx)
                    
                    if self.clt.cfg.cross_layer_decoders:
                        indices = (self.clt.l_idx == l_idx).nonzero(as_tuple=True)[0]
                        target_layers = self.clt.k_idx[indices]
                        
                        for idx, target_layer in enumerate(target_layers):
                            recon_acc[:, target_layer, :] += out_i[:, idx, :]
                        
                        recon = recon_acc[:, l_idx, :]
                    else:
                        recon = out_i
                        
                    if replace:
                        return recon.view(*act.shape).to(act.dtype)
                    else:
                        return act
                return replace_hook_fn
            
            should_replace = layer_idx in layers
            replace_hooks.append((self.hook_names[layer_idx], get_replace_hook(layer_idx, should_replace)))

        transcoder_logits = self.base_model.run_with_hooks(tokens, return_type="logits", fwd_hooks=replace_hooks)
        M_transcoder = metric_fn(transcoder_logits, tokens)
        P_transcoder = F.log_softmax(transcoder_logits, dim=-1)

        denom = (M_clean - M_zero)
        if denom == 0:
            replacement_score = float('nan')
        else:
            replacement_score = (M_transcoder - M_zero) / denom

        prob_clean = torch.exp(P_clean)
        kl_div = F.kl_div(P_transcoder, prob_clean, reduction='batchmean', log_target=False).item()

        return {
            "M_clean": M_clean,
            "M_zero": M_zero,
            "M_transcoder": M_transcoder,
            "replacement_score": replacement_score,
            "kl_divergence": kl_div
        }

    @torch.no_grad()
    def evaluate_sparsity_and_l0(self, tokens: torch.Tensor, layers: Optional[List[int]] = None, tau: float = 0.0) -> Dict[str, float]:
        """
        Calculates the L0 Norm and Sparsity Comparisons.
        """
        if layers is None:
            layers = list(range(self.clt.N_layers))
            
        _, cache = self.base_model.run_with_cache(tokens, names_filter=self.hook_names)
        
        B, seq_len = tokens.shape
        d_in = self.clt.d_in
        
        act_in_list = []
        for name in self.hook_names:
            if name in cache:
                act_in_list.append(cache[name].view(-1, d_in))
            else:
                raise ValueError(f"Hook name {name} not found in model cache.")
                
        act_in = torch.stack(act_in_list, dim=1).to(self.clt.dtype).to(self.clt.device)
        
        feat_act, _ = self.clt.encode(act_in)
        
        feat_act_selected = feat_act[:, layers, :]
        
        active_counts = (feat_act_selected > 0).float().sum(dim=-1)
        l0_norm = active_counts.mean().item()
        
        active_counts_pruned = (feat_act_selected > tau).float().sum(dim=-1)
        pruned_l0_norm = active_counts_pruned.mean().item()
        
        total_features = self.clt.local_d_latent
        training_sparsity = 1.0 - (l0_norm / total_features)
        pruning_sparsity = 1.0 - (pruned_l0_norm / total_features)
        
        return {
            "l0_norm": l0_norm,
            "pruned_l0_norm": pruned_l0_norm,
            "training_sparsity": training_sparsity,
            "pruning_sparsity": pruning_sparsity
        }

    @torch.no_grad()
    def evaluate_activation_density(self, tokens: torch.Tensor, layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculates the activation density (proportion of tokens each feature is active for).
        """
        if layers is None:
            layers = list(range(self.clt.N_layers))
            
        _, cache = self.base_model.run_with_cache(tokens, names_filter=self.hook_names)
        
        B, seq_len = tokens.shape
        d_in = self.clt.d_in
        
        act_in_list = []
        for name in self.hook_names:
            if name in cache:
                act_in_list.append(cache[name].view(-1, d_in))
            else:
                raise ValueError(f"Hook name {name} not found in model cache.")
                
        act_in = torch.stack(act_in_list, dim=1).to(self.clt.dtype).to(self.clt.device)
        
        feat_act, _ = self.clt.encode(act_in)
        
        feat_act_selected = feat_act[:, layers, :]
        
        density = (feat_act_selected > 0).float().mean(dim=0)
        
        mean_density = density.mean().item()
        
        dead_features = (density == 0).sum().item()
        total_features = len(layers) * self.clt.local_d_latent
        
        return {
            "activation_density_per_feature": density, 
            "mean_activation_density": mean_density,
            "dead_features_count": dead_features,
            "dead_features_ratio": dead_features / total_features
        }
