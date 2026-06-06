"""
Smoke-test for the GPU normalization fix (Task 1).

Checks:
1. After ActivationsStore.__init__, estimated_norm_scaling_factor_in/out
   - are on the correct device (cpu in CI, cuda if available)
   - have the correct dtype (matching cfg.dtype)
   - are not ones (i.e. actually estimated, not left at default)

2. apply_norm_scaling_factor_in/out and remove_norm_scaling_factor_in/out
   - work without device-mismatch errors
   - are exact inverses of each other

3. Calling estimate_norm_scaling_factor explicitly keeps everything on-device.

Run from the project root:
    python tests/test_norm_fix.py
Or via pytest:
    pytest tests/test_norm_fix.py -v
"""
import sys
from pathlib import Path

# Make sure project root is on the path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest

from tests.utils import build_clt_training_runner_cfg, NEEL_NANDA_C4_10K_DATASET
from clt_forge.training.activations_store import ActivationsStore
from sae_lens.load_model import load_model

# ── helpers ───────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _make_store(device: str = DEVICE, dtype: str = "float32") -> ActivationsStore:
    cfg = build_clt_training_runner_cfg(
        device=device,
        dtype=dtype,
        disk=True,   # local test dataset uses save_to_disk format
        # tiny settings so the test is fast
        n_batches_for_norm_estimate=2,
        n_batches_in_buffer=4,
        store_batch_size_prompts=4,
        context_size=4,
        train_batch_size_tokens=4,
    )
    model = load_model(
        cfg.model_class_name,
        cfg.model_name,
        device=torch.device(cfg.device),
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )
    return ActivationsStore(model, cfg)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestNormScalingFactorDevice:

    def test_factors_live_on_correct_device(self):
        store = _make_store()
        expected_device = torch.device(DEVICE)
        assert store.estimated_norm_scaling_factor_in.device.type  == expected_device.type, \
            f"factor_in on {store.estimated_norm_scaling_factor_in.device}, expected {expected_device}"
        assert store.estimated_norm_scaling_factor_out.device.type == expected_device.type, \
            f"factor_out on {store.estimated_norm_scaling_factor_out.device}, expected {expected_device}"
        print(f"\n  [OK] Scaling factors are on: {store.estimated_norm_scaling_factor_in.device}")

    def test_factors_have_correct_dtype_float32(self):
        store = _make_store(dtype="float32")
        assert store.estimated_norm_scaling_factor_in.dtype  == torch.float32
        assert store.estimated_norm_scaling_factor_out.dtype == torch.float32
        print(f"\n  [OK] Dtype (float32): {store.estimated_norm_scaling_factor_in.dtype}")

    def test_factors_have_correct_dtype_bfloat16(self):
        if DEVICE == "cpu":
            print("\n  [SKIP] bfloat16 autocast only relevant on CUDA")
            return
        store = _make_store(dtype="bfloat16")
        assert store.estimated_norm_scaling_factor_in.dtype  == torch.bfloat16, \
            f"Expected bfloat16, got {store.estimated_norm_scaling_factor_in.dtype}"
        assert store.estimated_norm_scaling_factor_out.dtype == torch.bfloat16
        print(f"\n  [OK] Dtype (bfloat16): {store.estimated_norm_scaling_factor_in.dtype}")

    def test_factors_are_not_all_ones(self):
        """Estimation must have actually run — factors shouldn't be the default ones."""
        store = _make_store()
        ones_in  = torch.ones_like(store.estimated_norm_scaling_factor_in)
        ones_out = torch.ones_like(store.estimated_norm_scaling_factor_out)
        assert not torch.allclose(store.estimated_norm_scaling_factor_in,  ones_in), \
            "factor_in is all ones — estimation did not run"
        assert not torch.allclose(store.estimated_norm_scaling_factor_out, ones_out), \
            "factor_out is all ones — estimation did not run"
        print(f"\n  [OK] factor_in  (not ones): {store.estimated_norm_scaling_factor_in}")
        print(f"  [OK] factor_out (not ones): {store.estimated_norm_scaling_factor_out}")

    def test_factor_shape(self):
        store = _make_store()
        n_layers = store.N_layers
        assert store.estimated_norm_scaling_factor_in.shape  == (n_layers,)
        assert store.estimated_norm_scaling_factor_out.shape == (n_layers,)
        print(f"\n  [OK] Shape: ({n_layers},) - one scalar per layer")


class TestApplyRemoveInverses:

    def test_apply_remove_in_are_inverses(self):
        store = _make_store()
        original = torch.randn(8, store.N_layers, store.cfg.d_in,
                               device=torch.device(DEVICE))
        clone = original.clone()

        store.apply_norm_scaling_factor_in(clone)
        store.remove_norm_scaling_factor_in(clone)

        assert torch.allclose(clone, original, rtol=1e-4, atol=1e-4), \
            f"apply then remove did not return to original: max diff={( clone - original).abs().max():.6f}"
        print("\n  [OK] apply_in * remove_in = identity")

    def test_apply_remove_out_are_inverses(self):
        store = _make_store()
        original = torch.randn(8, store.N_layers, store.cfg.d_in,
                               device=torch.device(DEVICE))
        clone = original.clone()

        store.apply_norm_scaling_factor_out(clone)
        store.remove_norm_scaling_factor_out(clone)

        assert torch.allclose(clone, original, rtol=1e-4, atol=1e-4), \
            f"apply then remove did not return to original: max diff={(clone - original).abs().max():.6f}"
        print("\n  [OK] apply_out * remove_out = identity")

    def test_no_device_mismatch_error(self):
        """
        Explicitly pass a CPU tensor to a store that may have GPU factors
        to confirm the defensive .to() prevents a RuntimeError.
        """
        store = _make_store()
        cpu_acts = torch.randn(4, store.N_layers, store.cfg.d_in, device="cpu")

        # Should not raise even if store's factors are on CUDA
        try:
            result = store.apply_norm_scaling_factor_in(cpu_acts.clone())
            print(f"\n  [OK] No device-mismatch error (result on: {result.device})")
        except RuntimeError as e:
            pytest.fail(f"Device mismatch raised: {e}")


class TestIteratorYieldsNormalizedActivations:

    def test_iterator_output_on_correct_device(self):
        store = _make_store()
        acts_in, acts_out = next(iter(store))
        assert acts_in.device.type  == DEVICE
        assert acts_out.device.type == DEVICE
        print(f"\n  [OK] Iterator yields tensors on: {acts_in.device}")

    def test_iterator_norm_close_to_sqrt_d_in(self):
        """After normalization, per-layer norms should be ≈ sqrt(d_in)."""
        store = _make_store()
        acts_in, _ = next(iter(store))
        # acts_in: [train_batch_size_tokens, N_layers, d_in]
        per_layer_norm = acts_in.float().norm(dim=-1).mean(dim=0)  # [N_layers]
        target = (store.cfg.d_in ** 0.5) * torch.ones_like(per_layer_norm)
        assert torch.allclose(per_layer_norm, target, rtol=0.3, atol=0.3), \
            f"Norms not close to sqrt(d_in)={store.cfg.d_in**0.5:.2f}: got {per_layer_norm}"
        print(f"\n  [OK] Per-layer norms close to sqrt(d_in)={store.cfg.d_in**0.5:.2f}: {per_layer_norm.tolist()}")


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"  GPU Normalization Fix - Smoke Test")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    suites = [
        TestNormScalingFactorDevice(),
        TestApplyRemoveInverses(),
        TestIteratorYieldsNormalizedActivations(),
    ]

    passed = failed = 0
    for suite in suites:
        suite_name = type(suite).__name__
        print(f"\n>> {suite_name}")
        for name in [m for m in dir(suite) if m.startswith("test_")]:
            method = getattr(suite, name)
            try:
                method()
                print(f"   PASS  {name}")
                passed += 1
            except (AssertionError, pytest.skip.Exception, Exception) as e:
                print(f"   FAIL  {name}")
                print(f"         {e}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
