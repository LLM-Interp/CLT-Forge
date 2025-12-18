from datasets import load_dataset
import argparse

def inspect_dataset(dataset_name: str, split: str = "train"):
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"\n Successfully loaded split '{split}' from dataset '{dataset_name}'.")

        print("\n Column names:")
        print(dataset.column_names)

        print("\n🔍 Checking for 'tokens' column:")
        if "tokens" in dataset.column_names:
            print(" 'tokens' column found.")
        else:
            print(" 'tokens' column NOT found.")

        print("\n🔎 First example:")
        print(dataset[0])

    except Exception as e:
        print(f" Error loading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a HuggingFace dataset for tokenization.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (e.g. 'wikitext', 'apollo-research/Skylion007-openwebtext-tokenizer-gpt2')")
    parser.add_argument("--split", type=str, default="train", help="Split to load (default: 'train')")

    args = parser.parse_args()
    inspect_dataset(args.dataset_name, args.split)
