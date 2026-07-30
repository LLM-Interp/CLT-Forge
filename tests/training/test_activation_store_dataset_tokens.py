from types import SimpleNamespace

import pytest
import torch

from clt_forge.config import CLTTrainingRunnerConfig
from clt_forge.training.activations_store import ActivationsStore


class DummyTokenizer:
    bos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


class DummyDataset:
    def __init__(self, rows, column_names):
        self.rows = rows
        self.column_names = column_names
        self.renamed = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def rename_column(self, old_name: str, new_name: str):
        self.renamed = (old_name, new_name)
        self.column_names = [
            new_name if column_name == old_name else column_name
            for column_name in self.column_names
        ]
        for row in self.rows:
            row[new_name] = row.pop(old_name)
        return self


def make_store(*, is_dataset_tokenized: bool = True, dataset_text_column: str = "text"):
    store = object.__new__(ActivationsStore)
    store.cfg = SimpleNamespace(
        dataset_path="dummy-dataset",
        dataset_text_column=dataset_text_column,
        is_dataset_tokenized=is_dataset_tokenized,
        is_distributed=False,
        is_multilingual_split_dataset=False,
    )
    store.model = SimpleNamespace(tokenizer=DummyTokenizer())
    store.context_size = 4
    return store


class DummyModel:
    def __init__(self, d_in: int = 3, n_layers: int = 1):
        self.cfg = SimpleNamespace(n_layers=n_layers)
        self.tokenizer = DummyTokenizer()
        self.d_in = d_in
        self.seen_batches = []

    def run_with_cache(self, batch_tokens, names_filter, prepend_bos=False):
        assert prepend_bos is False
        self.seen_batches.append(batch_tokens.detach().cpu())
        base = batch_tokens.float().unsqueeze(-1).repeat(1, 1, self.d_in)
        cache = {
            name: base + float(i)
            for i, name in enumerate(names_filter)
        }
        return None, cache


def make_runner_cfg(*, is_dataset_tokenized: bool, dataset_text_column: str = "text"):
    return CLTTrainingRunnerConfig(
        device="cpu",
        dtype="float32",
        model_name="dummy-model",
        dataset_path="dummy-dataset",
        is_dataset_tokenized=is_dataset_tokenized,
        dataset_text_column=dataset_text_column,
        d_in=3,
        d_latent=4,
        train_batch_size_tokens=4,
        context_size=4,
        n_batches_in_buffer=2,
        store_batch_size_prompts=1,
        total_training_tokens=8,
        n_batches_for_norm_estimate=1,
        log_to_wandb=False,
        distributed_setup="None",
        logger_verbose=False,
    )


def test_tokens_from_dataset_row_uses_pre_tokenized_tokens_and_strips_bos():
    store = make_store(is_dataset_tokenized=True)

    toks = store._tokens_from_dataset_row({"tokens": [0, 11, 12, 13]})

    assert torch.equal(toks, torch.tensor([11, 12, 13]))


def test_tokens_from_dataset_row_tokenizes_raw_text():
    store = make_store(is_dataset_tokenized=False)

    toks = store._tokens_from_dataset_row({"text": "abc"})

    assert torch.equal(toks, torch.tensor([97, 98, 99]))


def test_tokens_from_dataset_row_uses_custom_raw_text_column():
    store = make_store(is_dataset_tokenized=False, dataset_text_column="content")

    toks = store._tokens_from_dataset_row({"content": "abc"})

    assert torch.equal(toks, torch.tensor([97, 98, 99]))


def test_validate_dataset_columns_renames_input_ids_for_tokenized_dataset():
    store = make_store(is_dataset_tokenized=True)
    store.raw_ds = DummyDataset([{"input_ids": [1, 2, 3, 4]}], ["input_ids"])

    store._validate_dataset_columns()

    assert store.raw_ds.renamed == ("input_ids", "tokens")
    assert store.raw_ds.column_names == ["tokens"]


def test_validate_dataset_columns_requires_text_column_for_raw_dataset():
    store = make_store(is_dataset_tokenized=False)
    store.raw_ds = DummyDataset([{"content": "abc"}], ["content"])

    with pytest.raises(ValueError, match="dataset_text_column='text'"):
        store._validate_dataset_columns()


def test_iterate_raw_dataset_tokens_tokenizes_and_truncates_raw_text():
    store = make_store(is_dataset_tokenized=False)
    store.raw_ds = DummyDataset([{"text": "abcdef"}], ["text"])

    toks = list(store._iterate_raw_dataset_tokens())

    assert len(toks) == 1
    assert torch.equal(toks[0], torch.tensor([97, 98, 99, 100]))


def test_iterate_raw_dataset_tokens_keeps_short_raw_text_sequences():
    store = make_store(is_dataset_tokenized=False)
    store.raw_ds = DummyDataset([{"text": "abc"}], ["text"])

    toks = list(store._iterate_raw_dataset_tokens())

    assert len(toks) == 1
    assert torch.equal(toks[0], torch.tensor([97, 98, 99]))


def test_raw_text_dataset_runs_through_activation_store(monkeypatch):
    dataset = DummyDataset([{"text": "abcdefgh"}], ["text"])
    monkeypatch.setattr(
        "clt_forge.training.activations_store.load_dataset_auto",
        lambda *args, **kwargs: dataset,
    )
    cfg = make_runner_cfg(is_dataset_tokenized=False)
    model = DummyModel(d_in=cfg.d_in)

    store = ActivationsStore(
        model,
        cfg,
        estimated_norm_scaling_factor_in=torch.ones(model.cfg.n_layers),
        estimated_norm_scaling_factor_out=torch.ones(model.cfg.n_layers),
    )
    act_in, act_out = next(iter(store))

    assert act_in.shape == (cfg.train_batch_size_tokens, model.cfg.n_layers, cfg.d_in)
    assert act_out.shape == act_in.shape
    assert len(model.seen_batches) == cfg.n_batches_in_buffer
    assert all(batch.shape == (cfg.store_batch_size_prompts, cfg.context_size + 1) for batch in model.seen_batches)
    assert all(torch.all(batch[:, 0] == model.tokenizer.bos_token_id) for batch in model.seen_batches)


def test_pre_tokenized_dataset_runs_through_activation_store(monkeypatch):
    dataset = DummyDataset([{"tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8]}], ["tokens"])
    monkeypatch.setattr(
        "clt_forge.training.activations_store.load_dataset_auto",
        lambda *args, **kwargs: dataset,
    )
    cfg = make_runner_cfg(is_dataset_tokenized=True)
    model = DummyModel(d_in=cfg.d_in)

    store = ActivationsStore(
        model,
        cfg,
        estimated_norm_scaling_factor_in=torch.ones(model.cfg.n_layers),
        estimated_norm_scaling_factor_out=torch.ones(model.cfg.n_layers),
    )
    act_in, act_out = next(iter(store))

    assert act_in.shape == (cfg.train_batch_size_tokens, model.cfg.n_layers, cfg.d_in)
    assert act_out.shape == act_in.shape
    assert len(model.seen_batches) == cfg.n_batches_in_buffer


def test_raw_text_dataset_can_generate_cached_activations(monkeypatch, tmp_path):
    dataset = DummyDataset([{"text": "abcdefgh"}], ["text"])
    monkeypatch.setattr(
        "clt_forge.training.activations_store.load_dataset_auto",
        lambda *args, **kwargs: dataset,
    )
    cfg = make_runner_cfg(is_dataset_tokenized=False)
    model = DummyModel(d_in=cfg.d_in)
    store = ActivationsStore(
        model,
        cfg,
        estimated_norm_scaling_factor_in=torch.ones(model.cfg.n_layers),
        estimated_norm_scaling_factor_out=torch.ones(model.cfg.n_layers),
    )

    store.generate_and_save_activations(path=str(tmp_path), split_count=1)

    assert (tmp_path / f"ctx_{cfg.context_size}" / "activations_split_0.safetensors").exists()
