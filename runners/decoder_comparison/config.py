import wandb
from circuitlab.config.clt_training_runner_config import CLTTrainingRunnerConfig


def clt_training_runner_config(decoder_type: str = "full", decoder_rank: int = 64, checkpoint_path: str = "checkpoints"):
    MODEL = "gpt2"

    total_training_steps = 500
    train_batch_size_tokens = 2048
    total_training_tokens = train_batch_size_tokens * total_training_steps

    lr_decay_steps = total_training_steps // 10
    lr_warm_up_steps = 50
    l0_waiting_steps = 0
    l0_warm_up_steps = int(0.7 * total_training_steps) - l0_waiting_steps - 1
    decay_stable_steps = total_training_steps - l0_warm_up_steps - lr_decay_steps

    run_name = f"{decoder_type}" if decoder_type == "full" else f"{decoder_type}_r{decoder_rank}"

    cfg = CLTTrainingRunnerConfig(
        device="cuda",
        dtype="bfloat16",
        seed=42,
        n_checkpoints=0,
        checkpoint_path=checkpoint_path,
        logger_verbose=True,
        model_class_name="HookedTransformer",
        model_name=MODEL,
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        context_size=64,
        from_pretrained_path=None,
        d_in=768,
        expansion_factor=16,
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.0,
        n_batches_in_buffer=20,
        store_batch_size_prompts=32,
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        gradient_accumulation_steps=1,
        adam_beta1=0.0,
        adam_beta2=0.999,
        lr=4e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        final_lr_scale=0.0,
        l0_coefficient=2.0,
        dead_penalty_coef=1e-3,
        dead_feature_window=100,
        l0_warm_up_steps=l0_warm_up_steps,
        l0_waiting_steps=l0_waiting_steps,
        decay_stable_steps=decay_stable_steps,
        cross_layer_decoders=True,
        decoder_type=decoder_type,
        decoder_rank=decoder_rank,
        log_to_wandb=True,
        wandb_project="gpt2-clt-lora-comparison",
        wandb_id=wandb.util.generate_id(),
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100,
        run_name=run_name,
        wandb_entity=None,
        distributed_setup="None",
    )

    return cfg
