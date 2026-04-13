from datasets import load_dataset
from transformers import AutoTokenizer

# Prepare path
save_path = "NeelNanda_c4_10k_tokenized"
# os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Load and tokenize
raw = load_dataset("NeelNanda/c4-10k", split="train[:100]")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize(example):
    return {"tokens": tokenizer(example["text"], truncation=True)["input_ids"]}

tokenized = raw.map(tokenize)
tokenized.save_to_disk(save_path)
print(f"Tokenized dataset saved to {save_path}")
