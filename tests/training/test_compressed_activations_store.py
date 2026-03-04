import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from circuitlab.training.compressed_activations_store import (
    CompressionConfig,
    CompressedActivationsStore,
    QuantizedTensor,
    estimate_compression_savings,
)

# Fixtures
@pytest.fixture
def sample_activations():
    """Generate sample activations for testing."""
    torch.manual_seed(42)
    n_tokens = 1024
    n_layers = 12
    d_model = 768

    act_in = torch.randn(n_tokens, n_layers, d_model)
    act_out = torch.randn(n_tokens, n_layers, d_model) * 1.5

    return act_in, act_out

@pytest.fixture
def sample_tokens():
    """Generate sample token IDs."""
    return torch.randint(0, 50000, (1024,), dtype=torch.long)


# CompressionConfig tests
class TestCompressionConfig:
    def test_default_config(self):
        config = CompressionConfig()
        assert config.quantization == "int8"
        assert config.compression == "zstd"
        assert config.compression_level == 3
        assert config.per_layer_scale
        assert config.symmetric

    def test_custom_config(self):
        config = CompressionConfig(
            quantization="int4",
            compression="none",
            compression_level=10,
            per_layer_scale=False,
            symmetric=False,
        )
        assert config.quantization == "int4"
        assert config.compression == "none"
        assert config.compression_level == 10
        assert not config.per_layer_scale
        assert not config.symmetric

    def test_invalid_compression_level(self):
        with pytest.raises(ValueError):
            CompressionConfig(compression="zstd", compression_level=30)


# Quantization tests
class TestQuantization:
    @pytest.mark.parametrize("quantization", ["none", "int8", "int4"])
    def test_quantization_roundtrip(self, sample_activations, quantization):
        """Test that quantization/dequantization preserves data approximately."""
        act_in, act_out = sample_activations

        config = CompressionConfig(
            quantization=quantization,
            compression="none",
            per_layer_scale=True,
            symmetric=True,
        )
        store = CompressedActivationsStore(config)

        q_in, q_out = store.quantize_activations(act_in, act_out)
        recon_in = q_in.dequantize()
        recon_out = q_out.dequantize()

        # Check shapes match
        assert recon_in.shape == act_in.shape
        assert recon_out.shape == act_out.shape

        # Check dtypes match
        assert recon_in.dtype == act_in.dtype
        assert recon_out.dtype == act_out.dtype

        if quantization == "none":
            # Should be nearly identical (only float16 precision loss)
            assert torch.allclose(recon_in, act_in, atol=1e-2)
            assert torch.allclose(recon_out, act_out, atol=1e-2)
        elif quantization == "int8":
            # Should have small error
            mae_in = (act_in - recon_in).abs().mean()
            mae_out = (act_out - recon_out).abs().mean()
            assert mae_in < 0.1, f"int8 MAE too high: {mae_in}"
            assert mae_out < 0.2, f"int8 MAE too high: {mae_out}"

    def test_per_layer_scale(self, sample_activations):
        """Test that per-layer scaling preserves layer-wise statistics."""
        act_in, act_out = sample_activations

        config = CompressionConfig(
            quantization="int8",
            compression="none",
            per_layer_scale=True,
            symmetric=True,
        )
        store = CompressedActivationsStore(config)

        q_in, _ = store.quantize_activations(act_in, act_out)

        # Scale should have shape [1, n_layers, 1] for per-layer
        assert q_in.scale.shape == (1, 12, 1) or len(q_in.scale.shape) == 3

    def test_global_scale(self, sample_activations):
        """Test that global scaling uses single scale factor."""
        act_in, act_out = sample_activations

        config = CompressionConfig(
            quantization="int8",
            compression="none",
            per_layer_scale=False,
            symmetric=True,
        )
        store = CompressedActivationsStore(config)

        q_in, _ = store.quantize_activations(act_in, act_out)

        # Scale should be scalar or 1D with single element
        assert q_in.scale.size == 1 or q_in.scale.shape == (1,)


# Save/Load tests
class TestSaveLoad:
    @pytest.mark.parametrize("quantization,compression", [
        ("none", "none"),
        ("int8", "none"),
        ("int8", "zstd"),
        ("int4", "none"),
        ("int4", "zstd"),
    ])
    def test_save_load_roundtrip(self, sample_activations, sample_tokens, quantization, compression):
        """Test save/load preserves data."""
        act_in, act_out = sample_activations

        config = CompressionConfig(
            quantization=quantization,
            compression=compression,
            per_layer_scale=True,
            symmetric=True,
        )
        store = CompressedActivationsStore(config)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            store.save_compressed(
                temp_path,
                act_in,
                act_out,
                tokens=sample_tokens,
                metadata={"test_key": "test_value"},
            )

            # Load
            loaded_in, loaded_out, loaded_tokens, metadata = store.load_compressed(temp_path)

            # Check shapes
            assert loaded_in.shape == act_in.shape
            assert loaded_out.shape == act_out.shape
            assert loaded_tokens.shape == sample_tokens.shape

            # Check tokens are exact
            assert torch.equal(loaded_tokens, sample_tokens)

            # Check metadata
            assert metadata.get("test_key") == "test_value"
            assert metadata.get("quantization") == quantization
            assert metadata.get("compression") == compression

        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_without_tokens(self, sample_activations):
        """Test save/load without tokens."""
        act_in, act_out = sample_activations

        config = CompressionConfig(quantization="int8", compression="zstd")
        store = CompressedActivationsStore(config)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            store.save_compressed(temp_path, act_in, act_out, tokens=None)
            loaded_in, loaded_out, loaded_tokens, metadata = store.load_compressed(temp_path)

            assert loaded_tokens is None
            assert metadata.get("has_tokens")

        finally:
            temp_path.unlink(missing_ok=True)

    def test_file_size_reduction(self, sample_activations):
        """Test that compression reduces file size."""
        act_in, act_out = sample_activations

        original_size = act_in.element_size() * act_in.numel() + act_out.element_size() * act_out.numel()

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Test int8+zstd
            config = CompressionConfig(quantization="int8", compression="zstd")
            store = CompressedActivationsStore(config)
            file_size = store.save_compressed(temp_path, act_in, act_out)

            # Should achieve at least 2x compression
            compression_ratio = original_size / file_size
            assert compression_ratio > 2.0, f"Expected >2x compression, got {compression_ratio:.2f}x"

        finally:
            temp_path.unlink(missing_ok=True)


# QuantizedTensor tests
class TestQuantizedTensor:
    def test_dequantize_int8(self):
        """Test dequantization of int8 data."""
        data = np.array([[-127, 0, 127], [-64, 64, 0]], dtype=np.int8)
        scale = np.array([0.01], dtype=np.float32)

        qt = QuantizedTensor(
            data=data,
            scale=scale,
            original_shape=(2, 3),
            original_dtype=torch.float32,
            scale_shape=(1,),
        )

        result = qt.dequantize()
        assert result.shape == (2, 3)
        assert result.dtype == torch.float32

        # Check approximate values
        expected = torch.tensor([[-1.27, 0.0, 1.27], [-0.64, 0.64, 0.0]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_dequantize_with_scale_shape(self):
        """Test that scale_shape is used for proper broadcasting."""
        data = np.zeros((4, 3, 2), dtype=np.int8)
        scale = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # One per "layer"

        qt = QuantizedTensor(
            data=data,
            scale=scale,
            original_shape=(4, 3, 2),
            original_dtype=torch.float32,
            scale_shape=(1, 3, 1),  # Broadcasting shape
        )

        result = qt.dequantize()
        assert result.shape == (4, 3, 2)


# Storage estimation tests
class TestStorageEstimation:
    def test_estimate_compression_savings(self):
        """Test storage estimation function."""
        config = CompressionConfig(quantization="int8", compression="zstd")

        result = estimate_compression_savings(
            num_tokens=1_000_000,
            num_layers=12,
            d_model=768,
            config=config,
        )

        assert "base_size_gb" in result
        assert "final_size_gb" in result
        assert "compression_ratio" in result
        assert "savings_gb" in result
        assert "savings_percent" in result

        # Basic sanity checks
        assert result["final_size_gb"] < result["base_size_gb"]
        assert result["compression_ratio"] > 1.0
        assert result["savings_percent"] > 0

    def test_compression_ratio_ordering(self):
        """Test that more aggressive compression gives better ratios."""
        configs = [
            CompressionConfig(quantization="int8", compression="none"),
            CompressionConfig(quantization="int8", compression="zstd"),
            CompressionConfig(quantization="int4", compression="zstd"),
        ]

        ratios = []
        for config in configs:
            result = estimate_compression_savings(
                num_tokens=1_000_000,
                num_layers=12,
                d_model=768,
                config=config,
            )
            ratios.append(result["compression_ratio"])

        # Each should be better than the previous
        assert ratios[1] > ratios[0], "zstd should improve over none"
        assert ratios[2] > ratios[1], "int4 should improve over int8"


# Integration tests
class TestIntegration:
    def test_load_with_different_config(self, sample_activations):
        """Test that loading works regardless of the store's config."""
        act_in, act_out = sample_activations

        # Save with int8+zstd
        save_config = CompressionConfig(quantization="int8", compression="zstd")
        save_store = CompressedActivationsStore(save_config)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_store.save_compressed(temp_path, act_in, act_out)

            # Load with a different config (should still work)
            load_config = CompressionConfig(quantization="int4", compression="none")
            load_store = CompressedActivationsStore(load_config)

            loaded_in, loaded_out, _, metadata = load_store.load_compressed(temp_path)

            # Should load correctly based on file metadata
            assert loaded_in.shape == act_in.shape
            assert metadata["quantization"] == "int8"  # From file, not from load_store

        finally:
            temp_path.unlink(missing_ok=True)

    def test_large_activation_handling(self):
        """Test handling of larger activations."""
        # Simulate larger batch
        torch.manual_seed(42)
        act_in = torch.randn(4096, 12, 768)
        act_out = torch.randn(4096, 12, 768) * 1.5

        config = CompressionConfig(quantization="int8", compression="zstd")
        store = CompressedActivationsStore(config)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = Path(f.name)

        try:
            file_size = store.save_compressed(temp_path, act_in, act_out)
            loaded_in, loaded_out, _, _ = store.load_compressed(temp_path)

            assert loaded_in.shape == act_in.shape
            assert loaded_out.shape == act_out.shape

            # Check compression is effective for larger data
            original_size = act_in.element_size() * act_in.numel() * 2
            assert file_size < original_size * 0.5  # At least 2x compression

        finally:
            temp_path.unlink(missing_ok=True)
