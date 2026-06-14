"""Benchmark dense vs sparse decode: latency (forward, forward+backward) and peak
memory across sparsity levels, to find the dense->sparse crossover point.

Target is the TRAINING decode: JumpReLU activations start dense early in training
and become ~99% sparse as it converges, so we time forward AND backward (training
backprops through the decoder). The baseline we compare against is the existing
dense decode (torch.einsum) already in clt.py.

Outputs, per (mode, batch, density), for each method (dense / gather / csr):
  - fwd_ms      : median forward time
  - fb_ms       : median forward+backward time   (the training-relevant number)
  - mem_fb_gb   : peak CUDA memory during fwd+bwd
and the speedup of the best sparse method vs dense, plus the crossover sparsity.

Run on a GPU node via hrun (login node has no GPU):
    hrun -c 8 -m 64 -g RTXA6000 -d -x -n clt_bench bash -c '
      source ~/miniforge3/etc/profile.d/conda.sh && conda activate clt-forge
      cd <repo> && python scripts/benchmark_sparse_decode.py --device cuda'

Quick CPU smoke test (tiny, just checks it runs without crashing):
    python scripts/benchmark_sparse_decode.py --device cpu --quick
"""

import argparse
import time
import torch
import sys

sys.path.insert(0, "src")

from clt_forge.config import CLTConfig
from clt_forge.clt import CLT


def make_clt(cross_layer, sparse_decode, d_in, d_latent, n_layers, dtype, device):
    cfg = CLTConfig(
        device=device,
        dtype=dtype,
        seed=42,
        model_name="bench",
        d_in=d_in,
        d_latent=d_latent,
        n_layers=n_layers,
        jumprelu_bandwidth=0.001,
        jumprelu_init_threshold=0.01,
        normalize_decoder=False,
        dead_feature_window=1000,
        cross_layer_decoders=cross_layer,
        context_size=128,
        l0_coefficient=1e-3,
        sparse_decode=sparse_decode,
        functional_loss=None,
    )
    return CLT(cfg)


def make_sparse_z(B, N, K, density, device, dtype, requires_grad=False):
    # Cheap construction: scatter nnz random values into a zero tensor. Avoids
    # allocating a full float32 rand/mask over [B,N,K] (which alone is ~7GB at
    # B=4096). Duplicate indices just overwrite -> density is approximate, fine
    # for timing.
    z = torch.zeros(B, N, K, device=device, dtype=dtype)
    total = B * N * K
    nnz = int(total * density)
    if nnz:
        idx = torch.randint(0, total, (nnz,), device=device)
        vals = torch.randn(nnz, device=device, dtype=dtype).abs()
        z.view(-1).index_copy_(0, idx, vals)
    if requires_grad:
        z.requires_grad_(True)  # set AFTER fill (can't fill a grad leaf in place)
    return z


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_call(fn, device, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000.0  # median, ms


def peak_gb(device):
    return torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0


def bench_one(clt, z, method, device, do_backward):
    clt.cfg.sparse_decode = method

    def fwd():
        with torch.no_grad():
            clt.decode(z)

    def fwdbwd():
        clt.zero_grad(set_to_none=True)
        z.grad = None
        out = clt.decode(z)
        out.float().pow(2).sum().backward()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fwd_ms = time_call(fwd, device)
    mem_fwd = peak_gb(device)

    fb_ms = None
    mem_fb = None
    if do_backward:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        fb_ms = time_call(fwdbwd, device)
        mem_fb = peak_gb(device)

    return dict(fwd_ms=fwd_ms, mem_fwd_gb=mem_fwd, fb_ms=fb_ms, mem_fb_gb=mem_fb)


def run_mode(cross_layer, batch_sizes, densities, args):
    methods = ["dense", "gather"] + ([] if cross_layer else ["csr"])
    do_bwd = not args.no_backward
    requires_grad = do_bwd

    label = "CROSS-LAYER" if cross_layer else "NON-CROSS-LAYER"
    print(f"\n{'='*100}\n{label}  decoders   "
          f"(d_in={args.d_in}, d_latent={args.d_latent}, n_layers={args.n_layers}, "
          f"dtype={args.dtype})\n{'='*100}")

    # ONE model; decode() dispatches on cfg.sparse_decode, so we just switch that
    # per method. (Building one CLT per method would triple the resident weights.)
    clt = make_clt(cross_layer, "dense", args.d_in, args.d_latent,
                   args.n_layers, args.dtype, args.device)
    N = clt.N_layers
    K = clt.local_d_latent

    for B in batch_sizes:
        print(f"\n--- batch (tokens) = {B} ---")
        hdr = f"{'density':>8} {'sparsity':>9}"
        for m in methods:
            hdr += f" | {m+' fb_ms':>13} {m+' mem_gb':>11}"
        hdr += f" | {'best speedup':>12}"
        print(hdr)
        print("-" * len(hdr))

        for density in densities:
            try:
                z = make_sparse_z(B, N, K, density, args.device, clt.dtype,
                                  requires_grad=requires_grad)
            except RuntimeError as e:
                print(f"{density:>8.3f}  (z alloc OOM/err: {str(e)[:40]})")
                continue
            sparsity = 1.0 - (z.count_nonzero().item() / max(z.numel(), 1))

            res = {}
            for m in methods:
                try:
                    res[m] = bench_one(clt, z, m, args.device, do_bwd)
                except RuntimeError as e:
                    res[m] = None
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    print(f"   [{m}] OOM/err at density={density}: {str(e)[:60]}")

            key = "fb_ms" if do_bwd else "fwd_ms"
            memkey = "mem_fb_gb" if do_bwd else "mem_fwd_gb"
            dense_t = res.get("dense") and res["dense"][key]
            sparse_ts = [res[m][key] for m in methods if m != "dense" and res.get(m)]
            speedup = (dense_t / min(sparse_ts)) if (dense_t and sparse_ts) else float("nan")

            row = f"{density:>8.3f} {sparsity:>9.4f}"
            for m in methods:
                if res.get(m):
                    row += f" | {res[m][key]:>13.3f} {res[m][memkey]:>11.3f}"
                else:
                    row += f" | {'OOM':>13} {'-':>11}"
            row += f" | {speedup:>11.2f}x"
            print(row)

            del z
            if args.device == "cuda":
                torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--d-in", type=int, default=768)
    p.add_argument("--d-latent", type=int, default=24576)   # GPT-2 d_model 768 * expansion 32 (runners/training/gpt2)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--no-backward", action="store_true", help="time forward only")
    p.add_argument("--quick", action="store_true", help="tiny CPU smoke config")
    args = p.parse_args()

    if args.quick:
        args.d_in, args.d_latent, args.n_layers, args.dtype = 32, 512, 4, "float32"
        noncross_batches, cross_batches = [64], [64]
        densities = [0.20, 0.05, 0.01]
    else:
        # gather materializes a [nnz, d_in] intermediate, so it only fits at high
        # sparsity; use modest batches and focus the sweep on the realistic training
        # region (~99% sparse = density ~0.01 and below). Low-density rows will OOM
        # the sparse paths and are recorded as such.
        noncross_batches = [1024]
        cross_batches = [256]
        # sweep down past the real CLT operating point: optimal_l0=10 per layer /
        # d_latent=24576 = density ~0.0004 (sparsity ~99.96%). Bracket it directly.
        densities = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0004, 0.0002, 0.0001]

    print(f"device={args.device} dtype={args.dtype} "
          f"target={'forward-only' if args.no_backward else 'forward+backward (training)'}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_mode(False, noncross_batches, densities, args)
    run_mode(True, cross_batches, densities, args)

    print("\nNotes:")
    print("- fb_ms = forward+backward median (training-relevant); lower is better.")
    print("- Theoretical FLOP ratio dense/sparse = 1/density; the measured speedup is")
    print("  lower because gather/csr add nonzero-finding + scatter overhead.")
    print("- Crossover = the sparsity where best-sparse fb_ms drops below dense fb_ms")
    print("  (speedup crosses 1.0x). That sparsity is the value to set sparse_threshold to.")


if __name__ == "__main__":
    main()
