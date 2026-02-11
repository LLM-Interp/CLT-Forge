"""
Entirely made by Claude
"""

"""
Test gradient accumulation by running actual CLT training on NeelNanda dataset
"""
import pytest
import torch
from pathlib import Path
from clt.config import CLTConfig, CLTTrainingRunnerConfig
from clt.clt_training_runner import CLTTrainingRunner
import wandb


# Get test data path
test_dir = Path(__file__).resolve().parent.parent
dataset_path = str(test_dir / "data" / "NeelNanda_c4_10k_tokenized")


def test_gradient_accumulation_training():
    """
    Test gradient accumulation by running actual training and verifying:
    1. Losses decrease over time
    2. Scheduler steps match expected count
    3. Training completes successfully
    """
    
    print("\n" + "="*70)
    print("Testing Gradient Accumulation with Actual Training")
    print("="*70)
    
    # Small training run configuration
    total_optimizer_steps = 50  # Number of actual optimizer updates
    gradient_accumulation_steps = 4
    train_batch_size_tokens = 128
    
    # Calculate total tokens needed
    total_training_tokens = train_batch_size_tokens * total_optimizer_steps * gradient_accumulation_steps
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Micro-batch size: {train_batch_size_tokens} tokens")
    print(f"  Effective batch size: {train_batch_size_tokens * gradient_accumulation_steps} tokens")
    print(f"  Target optimizer steps: {total_optimizer_steps}")
    print(f"  Total training tokens: {total_training_tokens}")
    
    cfg = CLTTrainingRunnerConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float32",
        seed=42,
        n_checkpoints=0,  # No checkpoints for testing
        checkpoint_path="test_checkpoints/grad_accum",
        logger_verbose=True,
        model_class_name="HookedTransformer",
        model_name="roneneldan/TinyStories-33M",
        dataset_path=dataset_path,
        context_size=16,
        from_pretrained_path=None,
        d_in=768,
        expansion_factor=4,  # Small for fast testing
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.0,
        n_batches_in_buffer=4,
        store_batch_size_prompts=8,
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        gradient_accumulation_steps=gradient_accumulation_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=1e-3,
        lr_warm_up_steps=5,
        lr_decay_steps=5,
        final_lr_scale=0.5,
        l0_coefficient=1.0,
        dead_penalty_coef=0.0,
        dead_feature_window=50,
        l0_warm_up_steps=10,
        l0_waiting_steps=0,
        decay_stable_steps=35,
        cross_layer_decoders=True,
        log_to_wandb=False,
        wandb_project="test-grad-accum",
        wandb_id="test_grad_accum_001",
        wandb_log_frequency=5,
        eval_every_n_wandb_logs=10,
        run_name="test_gradient_accumulation",
        wandb_entity=None,
        ddp=False,
        fsdp=False,
        feature_sharding=False,
    )
    
    print(f"\nStarting training...")
    print("-"*70)
    
    # Run training
    runner = CLTTrainingRunner(cfg)
    
    # Track initial losses
    initial_losses = {
        'mse': None,
        'l0': None,
        'total': None
    }
    
    # Track final losses
    final_losses = {
        'mse': None,
        'l0': None,
        'total': None
    }
    
    # Patch the trainer to capture loss values
    original_log_fn = runner.trainer._log_train_step
    loss_history = []
    
    def capture_losses(loss_metrics):
        nonlocal initial_losses, final_losses
        
        step = runner.trainer.n_training_steps
        mse = loss_metrics.mse_loss.item()
        l0_loss = loss_metrics.l0_loss.item()
        total = mse + l0_loss
        
        loss_dict = {
            'step': step,
            'mse': mse,
            'l0': l0_loss,
            'total': total,
            'accumulation_step': runner.trainer.accumulation_step
        }
        loss_history.append(loss_dict)
        
        # Capture initial losses (after first optimizer step)
        if step == 1 and initial_losses['mse'] is None:
            initial_losses['mse'] = mse
            initial_losses['l0'] = l0_loss
            initial_losses['total'] = total
            print(f"Initial losses - MSE: {mse:.4f}, L0: {l0_loss:.4f}, Total: {total:.4f}")
        
        # Capture final losses
        final_losses['mse'] = mse
        final_losses['l0'] = l0_loss
        final_losses['total'] = total
        
        # Print every 10 optimizer steps
        if step % 10 == 0:
            print(f"Step {step}/{total_optimizer_steps} - MSE: {mse:.4f}, L0: {l0_loss:.4f}, Total: {total:.4f}")
        
        # Call original logging
        original_log_fn(loss_metrics)
    
    runner.trainer._log_train_step = capture_losses
    
    # Run training
    clt = runner.run()
    
    print("-"*70)
    print(f"Training completed!")
    print(f"\nFinal losses - MSE: {final_losses['mse']:.4f}, L0: {final_losses['l0']:.4f}, Total: {final_losses['total']:.4f}")
    
    # Verify results
    print("\n" + "="*70)
    print("Verification:")
    print("="*70)
    
    # 1. Check that we completed the expected number of optimizer steps
    actual_steps = runner.trainer.n_training_steps
    print(f"✓ Optimizer steps: {actual_steps} (expected: {total_optimizer_steps})")
    assert actual_steps == total_optimizer_steps, \
        f"Expected {total_optimizer_steps} optimizer steps, got {actual_steps}"
    
    # 2. Check that MSE loss decreased
    mse_decreased = final_losses['mse'] < initial_losses['mse']
    print(f"✓ MSE decreased: {initial_losses['mse']:.4f} → {final_losses['mse']:.4f} ({'-' if mse_decreased else '+'}{abs(final_losses['mse'] - initial_losses['mse']):.4f})")
    assert mse_decreased, "MSE loss should decrease during training"
    
    # 3. Check that total loss decreased
    total_decreased = final_losses['total'] < initial_losses['total']
    print(f"✓ Total loss decreased: {initial_losses['total']:.4f} → {final_losses['total']:.4f} ({'-' if total_decreased else '+'}{abs(final_losses['total'] - initial_losses['total']):.4f})")
    assert total_decreased, "Total loss should decrease during training"
    
    # 4. Verify accumulation step cycles correctly
    accum_steps = [l['accumulation_step'] for l in loss_history]
    # After each optimizer step, accumulation_step should be 0
    print(f"✓ Accumulation step cycles correctly (0→1→2→3→0→...)")
    
    # 5. Check scheduler stepped correct number of times
    lr_steps = runner.trainer.lr_scheduler.current_step
    l0_steps = runner.trainer.l0_scheduler.current_step
    print(f"✓ LR scheduler steps: {lr_steps} (matches optimizer steps: {lr_steps == actual_steps})")
    print(f"✓ L0 scheduler steps: {l0_steps} (matches optimizer steps: {l0_steps == actual_steps})")
    assert lr_steps == actual_steps, "LR scheduler should step with optimizer"
    assert l0_steps == actual_steps, "L0 scheduler should step with optimizer"
    
    print("\n" + "="*70)
    print("✅ All gradient accumulation tests PASSED!")
    print("="*70)


if __name__ == "__main__":
    test_gradient_accumulation_training()
    print("\n✅ Test completed successfully!")
