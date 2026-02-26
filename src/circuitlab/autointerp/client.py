import os

import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mp.set_start_method("spawn", force=True)


def run_client(
    prompts: list[str],
    vllm_model: str,
    vllm_max_tokens: int,
) -> list[str]:
    """
    Run vLLM inference on a list of prompt strings.

    Args:
        prompts:        Raw prompt texts (one per feature).
        vllm_model:     HuggingFace model ID to load with vLLM.
        vllm_max_tokens: Maximum tokens to generate per prompt.

    Returns:
        List of generated explanation strings, in the same order as prompts.
    """
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(vllm_model)

    messages = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"You are an expert at summarizing neuron behaviors.\n\n{prompt}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    gpu_count = torch.cuda.device_count()

    llm_config: dict = {
        "model": vllm_model,
        "hf_token": os.environ.get("HF_TOKEN"),
        "gpu_memory_utilization": 0.90,
        "max_model_len": 3500,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 8,
        "use_v2_block_manager": False,
    }

    if gpu_count > 1:
        llm_config["tensor_parallel_size"] = gpu_count
    else:
        llm_config["tensor_parallel_size"] = 1
        llm_config["enforce_eager"] = False
        llm_config["enable_prefix_caching"] = True

    llm = LLM(**llm_config)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=vllm_max_tokens)

    results = llm.generate(messages, sampling_params)

    outputs = []
    for result in results:
        text = result.outputs[0].text.strip()
        if text.startswith("<|assistant|>"):
            text = text[len("<|assistant|>"):].lstrip()
        if text.startswith("LLM ANSWER:"):
            text = text[len("LLM ANSWER:"):].lstrip()
        outputs.append(text)

    return outputs
