import math
import torch 
import torch.nn.functional as F
from typing import Any, Literal

class LearningRateScheduler:
    def __init__(
        self,
        warmup_type: str,
        base_lr: float,
        total_training_steps: int,
        warmup_steps: int,
        lr_decay_steps: int = 1,
        final_lr_scale: float = 0.0,
        lr_waiting_steps: int = 0, 
        decay_stable: int = 0 # plateau at final lr (useful for replacement score finetuning)
    ):
        assert 0 <= final_lr_scale <= 1.0, "final_lr_scale must be between 0 and 1"
        assert warmup_steps >= 0 and lr_decay_steps > 0, "warmup_steps must be ≥ 0, lr_decay_steps > 0"
        assert lr_waiting_steps + warmup_steps <= total_training_steps - lr_decay_steps - decay_stable, "warm up and waiting too long"
        assert warmup_type in ["cosine", "linear"], "warmup_type must be either 'cosine' or 'linear'"
        
        self.warmup_type = warmup_type
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.final_lr_scale = final_lr_scale
        self.total_training_steps = total_training_steps
        self.lr_waiting_steps = lr_waiting_steps
        self.decay_stable = decay_stable

        self.current_step = 1
        self.lr = 0.0

    def _compute_lr(self, step: int) -> float:
        if step < self.lr_waiting_steps:
            # Stay at zero during waiting phase
            return 0.0
        elif step < self.lr_waiting_steps + self.warmup_steps:
            # Cosine warmup from 0 to base_lr
            warmup_step = step - self.lr_waiting_steps
            if self.warmup_type == "cosine":
                return self.base_lr * 0.5 * (1 - math.cos(math.pi * warmup_step / self.warmup_steps))
            elif self.warmup_type == "linear":
                return self.base_lr * warmup_step / self.warmup_steps
            else:
                raise ValueError(f"Unknown warmup_type: {self.warmup_type}")
        elif step < self.total_training_steps - (self.lr_decay_steps + self.decay_stable):
            return self.base_lr
        elif step < self.total_training_steps - self.decay_stable:
            # Linear decay from base_lr to final_lr
            decay_progress = (step - (self.total_training_steps - self.lr_decay_steps - self.decay_stable)) / self.lr_decay_steps
            scale = 1 - (1 - self.final_lr_scale) * decay_progress
            return self.base_lr * scale
        else:
            return self.base_lr * self.final_lr_scale

    def step(self) -> float:
        self.lr = self._compute_lr(self.current_step)
        self.current_step += 1
        return self.lr

    def get_lr(self) -> float:
        return self.lr

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)

class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None

class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth


        # # our own interpretation of allow the gradient to pass through STE to all model parameters
        # threshold_band = threshold + bandwidth / 2
        # x_grad = ((x > threshold_band) + rectangle((x - threshold) / bandwidth) * (threshold_band / bandwidth)) * grad_output
        # I should compute the gradient using STE also for W_enc and b_enc ? 
        
        x_grad = (x > threshold) * grad_output

        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class FusedEncoder(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, k: int, activation: Literal["groupmax", "topk"]
    ):
        """
        input:  (B, L, D)
        weight: (L, D, M)
        bias:   (L, M)
        k:      int (number of top elements to select along dim=-1)
        """
        preacts = torch.einsum("bld,ldm->blm", input, weight)
        if bias is not None:
            preacts = preacts + bias.unsqueeze(0)
        preacts = F.relu(preacts)

        # Get top-k values and indices for each row
        if activation == "topk":
            values, indices = torch.topk(preacts, k, dim=-1, sorted=False)
        elif activation == "groupmax":
            values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)

            # torch.max gives us indices into each group, but we want indices into the
            # flattened tensor. Add the offsets to get the correct indices.
            num_latents = preacts.shape[-1]
            offsets = torch.arange(
                0, num_latents, num_latents // k, device=preacts.device
            )
            indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

        ctx.save_for_backward(input, weight, bias, indices)
        ctx.k = k
        ctx.activation = activation
        return values, indices, preacts

    @staticmethod
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, indices = ctx.saved_tensors
        
        B, L, D = input.shape
        _, _, M = weight.shape
        
        grad_input = grad_weight = grad_bias = None

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(input)
            for l in range(L):
                embedding_weight = weight[l].T  # Shape: (M, D)
                grad_input[:, l, :] = F.embedding_bag(
                    indices[:, l, :],
                    embedding_weight,
                    mode="sum",
                    per_sample_weights=grad_values[:, l, :].type_as(embedding_weight),
                )

        # --- Grad w.r.t. weight ---
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            for l in range(L):
                # Compute contributions from each top-k element for layer l
                contributions_l = grad_values[:, l, :].unsqueeze(2) * input[:, l, :].unsqueeze(1) # Shape: (B, k, D)
                contributions_l = contributions_l.reshape(-1, D) # Shape: (B*k, D)
                
                grad_weight_l_T = torch.zeros(M, D, device=weight.device, dtype=weight.dtype)
                grad_weight_l_T.index_add_(0, indices[:, l, :].flatten(), contributions_l.type_as(weight))
                grad_weight[l] = grad_weight_l_T.T

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)
            for l in range(L):
                grad_bias[l].index_add_(
                    0, indices[:, l, :].flatten(), grad_values[:, l, :].flatten().type_as(bias)
                )

        return grad_input, grad_weight, grad_bias, None, None


def fused_encoder(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    activation: Literal["groupmax", "topk"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper that performs a multi-layer projection followed by `activation`
    with a backward pass optimized using index_add and embedding_bag.
    """
    is_2d = (input.dim() == 2)
    if is_2d:
        input = input.unsqueeze(1)    # (B, 1, D)
        weight = weight.unsqueeze(0)  # (1, D, M)
        if bias is not None:
            bias = bias.unsqueeze(0)  # (1, M)
            
    values, indices, preacts = FusedEncoder.apply(input, weight, bias, k, activation)
    
    if is_2d:
        values = values.squeeze(1)
        indices = indices.squeeze(1)
        preacts = preacts.squeeze(1)
        
    return values, indices, preacts
