"""engines/qwen35moe/tools/ref_forward.py - a plain numpy forward pass of qwen35moe, the float reference for parity.

Weights are the GGUF's quantized weights, dequantized to float32; every activation stays in float32/64, so this
is the model the quantized weights DEFINE, without any engine's activation quantization.  Slow (seconds per token)
and only meant for short prompts.

    python engines/qwen35moe/tools/ref_forward.py MODEL.gguf 760,6511,314,9338,369 [out_logits.npy]
"""
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


class W:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            t = self.t[name]
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
            self.cache[name] = a.reshape([int(x) for x in reversed(t.shape.tolist())]) if a.ndim == 1 and len(t.shape) > 1 else a
        return self.cache[name]

    def expert(self, name, e):
        t = self.t[name]
        raw = t.data[e]   # [rows, bytes] for quantized, one expert
        return dequantize(raw, t.tensor_type).astype(np.float32)

    def field(self, key):
        return self.r.fields[key].contents()


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    path, ids = sys.argv[1], [int(t) for t in sys.argv[2].split(",")]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    w = W(path)
    nl = int(w.field("qwen35moe.block_count"))
    eps = float(w.field("qwen35moe.attention.layer_norm_rms_epsilon"))
    base = float(w.field("qwen35moe.rope.freq_base"))
    E, HD, NH, NKV, KH, VH, DS = 2048, 256, 16, 2, 16, 32, 128
    emb = w.get("token_embd.weight")
    T = len(ids)
    x = emb[ids].astype(np.float64)                    # [T, E]
    conv_state = {}
    kcache, vcache = {}, {}
    for il in range(nl):
        p = f"blk.{il}."
        h = rms(x, w.get(p + "attn_norm.weight"), eps)
        if (il + 1) % 4 != 0:
            qkv = h @ w.get(p + "attn_qkv.weight").T.astype(np.float64)       # [T, 8192]
            z = h @ w.get(p + "attn_gate.weight").T                            # [T, 4096]
            beta = sigmoid(h @ w.get(p + "ssm_beta.weight").T)                 # [T, 32]
            alpha = h @ w.get(p + "ssm_alpha.weight").T
            a = alpha + w.get(p + "ssm_dt.bias")
            sp = np.where(a > 20, a, np.log1p(np.exp(np.minimum(a, 20))))
            g = sp * w.get(p + "ssm_a")                                        # [T, 32]
            cw = w.get(p + "ssm_conv1d.weight").reshape(8192, 4)
            xin = np.concatenate([np.zeros((3, 8192)), qkv], 0)
            conv = np.stack([(xin[t:t + 4] * cw.T).sum(0) for t in range(T)])  # [T, 8192]
            conv = silu(conv)
            q = conv[:, :2048].reshape(T, KH, DS)
            k = conv[:, 2048:4096].reshape(T, KH, DS)
            v = conv[:, 4096:].reshape(T, VH, DS)
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + eps)
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + eps)
            S = np.zeros((VH, DS, DS))   # S[h][i][j]: key dim i, value dim j
            o = np.zeros((T, VH, DS))
            for t in range(T):
                for hh in range(VH):
                    kk = k[t, hh % KH]
                    qq = q[t, hh % KH]
                    dec = np.exp(g[t, hh])
                    kv = S[hh].T @ kk
                    delta = (v[t, hh] - dec * kv) * beta[t, hh]
                    S[hh] = dec * S[hh] + np.outer(kk, delta)
                    o[t, hh] = S[hh].T @ qq / np.sqrt(DS)
            on = rms(o, w.get(p + "ssm_norm.weight"), eps) * silu(z.reshape(T, VH, DS))
            att = on.reshape(T, 4096) @ w.get(p + "ssm_out.weight").T
        else:
            qg = (h @ w.get(p + "attn_q.weight").T).reshape(T, NH, 2 * HD)
            q, gate = qg[:, :, :HD], qg[:, :, HD:]
            k = (h @ w.get(p + "attn_k.weight").T).reshape(T, NKV, HD)
            v = (h @ w.get(p + "attn_v.weight").T).reshape(T, NKV, HD)
            q = rms(q, w.get(p + "attn_q_norm.weight"), eps)
            k = rms(k, w.get(p + "attn_k_norm.weight"), eps)
            inv = base ** (-np.arange(32) * 2.0 / 64)
            ang = np.arange(T)[:, None] * inv[None, :]                           # [T, 32]
            cs, sn = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
            for arr in (q, k):
                x0, x1 = arr[..., :32].copy(), arr[..., 32:64].copy()
                arr[..., :32] = x0 * cs - x1 * sn
                arr[..., 32:64] = x0 * sn + x1 * cs
            o = np.zeros((T, NH, HD))
            for hh in range(NH):
                kv = hh // (NH // NKV)
                s = q[:, hh] @ k[:, kv].T / np.sqrt(HD)
                s = s + np.triu(np.full((T, T), -np.inf), 1)
                s = np.exp(s - s.max(-1, keepdims=True))
                s /= s.sum(-1, keepdims=True)
                o[:, hh] = s @ v[:, kv]
            o = o * sigmoid(gate)
            att = o.reshape(T, NH * HD) @ w.get(p + "attn_output.weight").T
        x = x + att
        hn = rms(x, w.get(p + "post_attention_norm.weight"), eps)
        logits = hn @ w.get(p + "ffn_gate_inp.weight").T                        # [T, 256]
        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        moe = np.zeros_like(x)
        for t in range(T):
            top = np.argsort(-pr[t])[:8]
            wt = pr[t, top] / pr[t, top].sum()
            for e, ww in zip(top, wt):
                gw = w.expert(p + "ffn_gate_exps.weight", e)
                uw = w.expert(p + "ffn_up_exps.weight", e)
                dw = w.expert(p + "ffn_down_exps.weight", e)
                moe[t] += ww * (dw @ (silu(gw @ hn[t]) * (uw @ hn[t])))
        sh = (silu(hn @ w.get(p + "ffn_gate_shexp.weight").T) * (hn @ w.get(p + "ffn_up_shexp.weight").T)) @ w.get(p + "ffn_down_shexp.weight").T
        sg = sigmoid(hn @ w.get(p + "ffn_gate_inp_shexp.weight"))
        x = x + moe + sh * sg[:, None]
        print(f"layer {il:2d} |x| = {np.linalg.norm(x[-1]):.3f}", file=sys.stderr)
    hf = rms(x[-1], w.get("output_norm.weight"), eps)
    lg = hf @ w.get("output.weight").T
    top = np.argsort(-lg)[:10]
    m = lg.max()
    lp = lg - (m + np.log(np.exp(lg - m).sum()))
    print("top10:", [(int(i), round(float(lp[i]), 4)) for i in top])
    if out:
        np.save(out, lg.astype(np.float32))


if __name__ == "__main__":
    main()
