"""End-to-end benchmark: does sparse decode make a full TRAINING STEP faster?

The per-op benchmark (benchmark_sparse_decode.py) shows gather beats dense on the
decode op at the realistic CLT operating point. But decode is only PART of a
training step (encode is a comparable matmul, plus loss/penalties). This script
measures the honest end-to-end number via Amdahl: it times a full training step
(encode -> JumpReLU -> decode -> loss -> backward) with dense vs gather decode,
and separately the decode-only cost, so we can report:

  - achieved sparsity (we force the threshold so feat_act ~ realistic L0)
  - full training-step time, dense vs gather   -> the real end-to-end speedup
  - decode's share of the step                 -> why the end-to-end speedup is
                                                  smaller than the decode speedup

Run on a GPU node via hrun (capture output yourself; hrun -d does not):
    python scripts/benchmark_end_to_end.py --device cuda
"""

import argparse
import time
import torch
import sys

sys.path.insert(0, "src")

from clt_forge.config import CLTConfig
from clt_forge.clt import CLT


def make_clt(sparse_decode, d_in, d_latent, n_layers, dtype, device):
    cfg = CLTConfig(
        device=device, dtype=dtype, seed=42, model_name="bench",
        d_in=d_in, d_latent=d_latent, n_layers=n_layers,
        jumprelu_bandwidth=0.001, jumprelu_init_threshold=0.01,
        normalize_decoder=False, dead_feature_window=1000,
        cross_layer_decoders=False, context_size=128,
        l0_coefficient=1e-3, sparse_decode=sparse_decode, functional_loss=None,
    )
    return CLT(cfg)


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


def force_sparsity(model, act_in, target_density):
    """Set per-layer thresholds so encode->JumpReLU yields ~target_density active."""
    with torch.no_grad():
        _, hidden_pre = model.encode(act_in)  # [B, N, d_latent]
        N = hidden_pre.shape[1]
        for n in range(N):
            hp = hidden_pre[:, n, :].flatten().float()
            if hp.numel() > 1_000_000:  # torch.quantile caps ~16M elements
                idx = torch.randint(0, hp.numel(), (1_000_000,), device=hp.device)
                hp = hp[idx]
            t = torch.quantile(hp, 1.0 - target_density).clamp_min(1e-6)
            model.log_threshold.data[n, :] = torch.log(t).to(model.log_threshold.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--d-in", type=int, default=768)
    p.add_argument("--d-latent", type=int, default=24576)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--target-density", type=float, default=0.0004)  # L0~=10 / 24576
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.d_in, args.d_latent, args.n_layers, args.batch, args.dtype = 32, 512, 4, 64, "float32"
        args.target_density = 0.02

    print(f"device={args.device} dtype={args.dtype} B={args.batch} "
          f"d_in={args.d_in} d_latent={args.d_latent} n_layers={args.n_layers}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dense = make_clt("dense", args.d_in, args.d_latent, args.n_layers, args.dtype, args.device)
    gather = make_clt("gather", args.d_in, args.d_latent, args.n_layers, args.dtype, args.device)
    gather.load_state_dict(dense.state_dict(), strict=False)

    torch.manual_seed(0)
    act_in = (torch.randn(args.batch, args.n_layers, args.d_in,
                          device=args.device, dtype=dense.dtype) * 0.5)
    act_out = (torch.randn(args.batch, args.n_layers, args.d_in,
                           device=args.device, dtype=dense.dtype) * 0.5)

    # force realistic sparsity on BOTH models identically
    force_sparsity(dense, act_in, args.target_density)
    gather.load_state_dict(dense.state_dict(), strict=False)
    with torch.no_grad():
        feat, _ = dense.encode(act_in)
        ach = feat.count_nonzero().item() / feat.numel()
    print(f"forced sparsity: density={ach:.5f} (sparsity={1-ach:.5f}), "
          f"~L0/layer={ach*args.d_latent:.1f}")

    # precompute a sparse feat_act to feed decode-only timings
    with torch.no_grad():
        feat0, _ = dense.encode(act_in)

    # ---- TRAINING (fwd+bwd) timers ----
    def full_step(model):
        def f():
            model.zero_grad(set_to_none=True)
            m = model.loss(act_in, act_out, l0_coef=1e-3, df_coef=0.0)
            (m.mse_loss + m.l0_loss + m.dead_feature_loss).backward()
        return f

    def encode_train(model):
        def f():
            model.zero_grad(set_to_none=True)
            model.encode(act_in)[0].float().pow(2).sum().backward()
        return f

    def decode_train(model):
        z = feat0.detach().clone().requires_grad_(True)
        def f():
            model.zero_grad(set_to_none=True)
            z.grad = None
            model.decode(z).float().pow(2).sum().backward()
        return f

    # ---- INFERENCE (fwd only) timers ----
    def encode_infer(model):
        def f():
            with torch.no_grad():
                model.encode(act_in)
        return f

    def decode_infer(model):
        z = feat0.detach()
        def f():
            with torch.no_grad():
                model.decode(z)
        return f

    def full_infer(model):
        def f():
            with torch.no_grad():
                feat, _ = model.encode(act_in)
                model.decode(feat)
        return f

    T = {}
    for name, m in [("dense", dense), ("gather", gather)]:
        T[name] = {
            "step":       time_call(full_step(m),    args.device),
            "enc_tr":     time_call(encode_train(m), args.device),
            "dec_tr":     time_call(decode_train(m), args.device),
            "inf":        time_call(full_infer(m),   args.device),
            "enc_inf":    time_call(encode_infer(m), args.device),
            "dec_inf":    time_call(decode_infer(m), args.device),
        }

    d = T["dense"]
    print("\n================ TRAINING step (fwd+bwd) ================")
    print(f"  full step      : {d['step']:.2f} ms")
    print(f"  encode-only    : {d['enc_tr']:.2f} ms  ({d['enc_tr']/d['step']:.1%})")
    print(f"  decode-only    : {d['dec_tr']:.2f} ms  ({d['dec_tr']/d['step']:.1%})  <- only this is sparsifiable")
    other = d['step'] - d['enc_tr'] - d['dec_tr']
    print(f"  other(JumpReLU+loss+penalty) ~ {other:.2f} ms  ({other/d['step']:.1%})")
    print(f"  => END-TO-END training-step speedup (gather): {d['step']/T['gather']['step']:.2f}x")

    print("\n================ INFERENCE (forward only) ================")
    print(f"  full inference : {d['inf']:.2f} ms")
    print(f"  encode-only    : {d['enc_inf']:.2f} ms  ({d['enc_inf']/d['inf']:.1%})")
    print(f"  decode-only    : {d['dec_inf']:.2f} ms (dense) / {T['gather']['dec_inf']:.2f} ms (gather)"
          f"  ({d['dec_inf']/d['inf']:.1%} of inference)")
    print(f"  => END-TO-END inference speedup (gather):     {d['inf']/T['gather']['inf']:.2f}x")
    print(f"  => decode-op inference speedup (gather):       {d['dec_inf']/T['gather']['dec_inf']:.2f}x")


if __name__ == "__main__":
    main()
