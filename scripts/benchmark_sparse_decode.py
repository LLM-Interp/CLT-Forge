"""Benchmark dense vs sparse decode at various sparsity levels.

Usage (on a GPU node via hrun):
    python scripts/benchmark_sparse_decode.py [--device cuda] [--cross-layer]
"""

import argparse
import time
import torch
import sys
sys.path.insert(0, "src")

from clt_forge.config import CLTConfig
from clt_forge.clt import CLT


def make_clt(cross_layer: bool, sparse_decode: str, device: str) -> CLT:
    cfg = CLTConfig(
        device=device,
        dtype="bfloat16",
        seed=42,
        model_name="benchmark",
        d_in=768,
        d_latent=16384,
        n_layers=12,
        jumprelu_bandwidth=0.001,
        jumprelu_init_threshold=0.01,
        normalize_decoder=False,
        dead_feature_window=1000,
        cross_layer_decoders=cross_layer,
        context_size=128,
        l0_coefficient=1e-3,
        sparse_decode=sparse_decode,
    )
    return CLT(cfg)


def make_sparse_z(B, N, K, density, device, dtype=torch.bfloat16):
    z = torch.zeros(B, N, K, dtype=dtype, device=device)
    mask = torch.rand(B, N, K, device=device) < density
    z[mask] = torch.randn(mask.sum().item(), dtype=dtype, device=device).abs()
    return z


def benchmark_method(clt, z, method_name, warmup=5, repeats=20):
    clt.cfg.sparse_decode = method_name

    for _ in range(warmup):
        _ = clt.decode(z)
    if z.device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if z.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = clt.decode(z)
        if z.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cross-layer", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = args.device
    cross_layer = args.cross_layer
    B = args.batch_size

    methods = ["dense", "gather"]
    if not cross_layer:
        methods.append("csr")

    densities = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]

    clt = make_clt(cross_layer, "dense", device)
    N = clt.N_layers
    K = clt.local_d_latent

    print(f"Config: B={B}, N={N}, K={K}, d_in={clt.d_in}, cross_layer={cross_layer}, device={device}")
    print(f"{'density':>8}  {'sparsity':>8}", end="")
    for m in methods:
        print(f"  {m+' (ms)':>14}", end="")
    print(f"  {'speedup':>10}")
    print("-" * (20 + 16 * len(methods) + 12))

    with torch.no_grad():
        for density in densities:
            z = make_sparse_z(B, N, K, density, device, clt.dtype)
            actual_sparsity = 1.0 - (z.count_nonzero().item() / z.numel())

            results = {}
            for method in methods:
                results[method] = benchmark_method(clt, z, method)

            dense_ms = results["dense"]["mean_ms"]
            best_sparse_ms = min(
                results[m]["mean_ms"] for m in methods if m != "dense"
            )
            speedup = dense_ms / best_sparse_ms if best_sparse_ms > 0 else float("inf")

            print(f"{density:>8.3f}  {actual_sparsity:>8.3f}", end="")
            for m in methods:
                print(f"  {results[m]['mean_ms']:>14.2f}", end="")
            print(f"  {speedup:>9.2f}x")

    clt.cfg.sparse_decode = "dense"


if __name__ == "__main__":
    main()
