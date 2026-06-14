import pytest
import torch
from clt_forge.config import CLTConfig
from clt_forge.clt import CLT


def _make_clt(cross_layer: bool, sparse_decode: str = "dense", **overrides) -> CLT:
    cfg_dict = dict(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="test",
        d_in=32,
        d_latent=64,
        n_layers=4,
        jumprelu_bandwidth=0.001,
        jumprelu_init_threshold=0.01,
        normalize_decoder=False,
        dead_feature_window=1000,
        cross_layer_decoders=cross_layer,
        context_size=16,
        l0_coefficient=1e-3,
        sparse_decode=sparse_decode,
        functional_loss=None,
    )
    cfg_dict.update(overrides)
    cfg = CLTConfig(**cfg_dict)
    return CLT(cfg)


def _make_sparse_z(B, N, K, density=0.01, dtype=torch.float32):
    z = torch.zeros(B, N, K, dtype=dtype)
    mask = torch.rand(B, N, K) < density
    z[mask] = torch.randn(mask.sum().item(), dtype=dtype)
    z = torch.relu(z)  # JumpReLU output is non-negative
    return z


@pytest.mark.parametrize("cross_layer", [False, True])
@pytest.mark.parametrize("density", [0.01, 0.05, 0.10])
def test_gather_matches_dense(cross_layer, density):
    clt_dense = _make_clt(cross_layer, sparse_decode="dense")
    clt_sparse = _make_clt(cross_layer, sparse_decode="gather")
    clt_sparse.load_state_dict(clt_dense.state_dict(), strict=False)

    z = _make_sparse_z(8, clt_dense.N_layers, clt_dense.local_d_latent, density=density)

    with torch.no_grad():
        out_dense = clt_dense.decode(z)
        out_sparse = clt_sparse.decode(z)

    torch.testing.assert_close(out_sparse, out_dense, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("density", [0.01, 0.05, 0.10])
def test_csr_matches_dense(density):
    clt_dense = _make_clt(False, sparse_decode="dense")
    clt_csr = _make_clt(False, sparse_decode="csr")
    clt_csr.load_state_dict(clt_dense.state_dict(), strict=False)

    z = _make_sparse_z(8, clt_dense.N_layers, clt_dense.local_d_latent, density=density)

    with torch.no_grad():
        out_dense = clt_dense.decode(z)
        out_csr = clt_csr.decode(z)

    torch.testing.assert_close(out_csr, out_dense, atol=1e-5, rtol=1e-5)


def test_csr_rejects_cross_layer():
    clt = _make_clt(True, sparse_decode="csr")
    z = _make_sparse_z(4, clt.N_layers, clt.local_d_latent)
    with pytest.raises(ValueError, match="CSR sparse decode is not supported"):
        clt.decode(z)


def test_auto_selects_dense_when_not_sparse():
    clt = _make_clt(False, sparse_decode="auto", sparse_threshold=0.90)
    z = torch.randn(4, clt.N_layers, clt.local_d_latent)  # ~50% nonzero
    method = clt._resolve_decode_method(z)
    assert method == "dense"


def test_auto_selects_gather_when_sparse():
    clt = _make_clt(False, sparse_decode="auto", sparse_threshold=0.90)
    z = _make_sparse_z(4, clt.N_layers, clt.local_d_latent, density=0.02)
    method = clt._resolve_decode_method(z)
    assert method == "gather"


def test_all_zeros():
    for cross_layer in [False, True]:
        clt_dense = _make_clt(cross_layer, sparse_decode="dense")
        clt_sparse = _make_clt(cross_layer, sparse_decode="gather")
        clt_sparse.load_state_dict(clt_dense.state_dict(), strict=False)

        z = torch.zeros(4, clt_dense.N_layers, clt_dense.local_d_latent)

        with torch.no_grad():
            out_dense = clt_dense.decode(z)
            out_sparse = clt_sparse.decode(z)

        torch.testing.assert_close(out_sparse, out_dense, atol=1e-6, rtol=1e-6)


def test_gather_gradient_flows():
    clt = _make_clt(False, sparse_decode="gather")
    z = _make_sparse_z(4, clt.N_layers, clt.local_d_latent, density=0.05)
    z.requires_grad_(True)

    out = clt.decode(z)
    loss = out.sum()
    loss.backward()

    assert z.grad is not None
    assert clt.W_dec.grad is not None
    assert z.grad.shape == z.shape
    # Gradient should be zero where z was zero (no contribution)
    zero_mask = z.detach() == 0
    torch.testing.assert_close(
        z.grad[zero_mask],
        torch.zeros_like(z.grad[zero_mask]),
        atol=1e-7, rtol=0,
    )


def test_gather_cross_layer_gradient_flows():
    clt = _make_clt(True, sparse_decode="gather")
    z = _make_sparse_z(4, clt.N_layers, clt.local_d_latent, density=0.05)
    z.requires_grad_(True)

    out = clt.decode(z)
    loss = out.sum()
    loss.backward()

    assert z.grad is not None
    assert clt.W_dec.grad is not None
