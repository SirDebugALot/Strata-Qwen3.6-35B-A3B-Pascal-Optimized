// Isolated exact-layout experiment; include after accepted constants/helpers.
#pragma once
// PAD_Q changes shared Q pitch only. PERMUTE_K changes score ownership, not softmax/key order.
template <int AQ, bool SHARE_KV = false, bool PAD_Q = false, bool PERMUTE_K = false>
constexpr size_t attn_layout_smem() {
    constexpr int AR = AQ * 8;
    return (size_t)AR * (kHD + (PAD_Q ? 8 : 0)) * 2 + (size_t)kAK * kKPad * 2 + (SHARE_KV ? 0 : (size_t)kAK * kHD * 2) +
           (size_t)AR * kAK * 4 + 3 * AR * 4;
}

template <int AQ, bool SHARE_KV = false, bool PAD_Q = false, bool PERMUTE_K = false>
__global__ void __launch_bounds__(256) attn_layout_kernel(const __half* __restrict__ Qh, const __half* __restrict__ kc,
                                                           const __half* __restrict__ vc, int T, int p0, int max_ctx,
                                                           const float* __restrict__ qkv, int ld, __half* __restrict__ out) {
    constexpr int kAQ = AQ, kAR = AQ * 8;
    constexpr int kQStride = kHD + (PAD_Q ? 8 : 0);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* sQ = (__half*)smem_raw;                          // [64][256]
    __half* sK = sQ + kAR * kQStride;                              // [32][264]
    __half* sV = SHARE_KV ? sK : sK + kAK * kKPad;             // [32][256], optionally aliases K
    float* sS = (float*)(SHARE_KV ? sK + kAK * kKPad : sV + kAK * kHD); // after max(K,V) when shared
    float* sM = sS + kAR * kAK;                               // [64] running max
    float* sL = sM + kAR;                                     // [64] running sum
    float* sA = sL + kAR;                                     // [64] rescale factor
    const int kh = blockIdx.y, tq0 = blockIdx.x * kAQ, tid = threadIdx.x;
    const int nq = min(kAQ, T - tq0);
    // load Q rows: r = tok*8 + h  ->  Qh[tq0+tok][kh*8+h]
    for (int i = tid; i < kAR * kHD / 8; i += AQ * 32) {
        const int r = i / (kHD / 8), c8 = i % (kHD / 8);
        const int tok = r >> 3, h = r & 7;
        int4 v = make_int4(0, 0, 0, 0);
        if (tok < nq) v = *(const int4*)(Qh + ((size_t)(tq0 + tok) * kNH + kh * 8 + h) * kHD + c8 * 8);
        *(int4*)(sQ + r * kQStride + c8 * 8) = v;
    }
    if (tid < kAR) { sM[tid] = -INFINITY; sL[tid] = 0.0f; }
    // output accumulators: thread -> rows rg*8 .. rg*8+7 (rg = warp), dims lane*8 .. lane*8+7
    const int warp = tid >> 5, lane = tid & 31;
    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b) acc[a][b] = 0.0f;
    const int last_pos = p0 + tq0 + nq - 1;
    const int n_keys = last_pos + 1;
    // score micro-tile: rows (rp*2, rp*2+1), keys kq*4 .. kq*4+3
    const int rp = tid >> 3, kq = tid & 7;
    for (int k0 = 0; k0 < n_keys; k0 += kAK) {
        __syncthreads();
        for (int i = tid; i < kAK * kHD / 8; i += AQ * 32) {
            const int j = i / (kHD / 8), c8 = i % (kHD / 8);
            int4 kv4 = make_int4(0, 0, 0, 0), vv4 = make_int4(0, 0, 0, 0);
            if (k0 + j < n_keys) {
                kv4 = *(const int4*)(kc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
                if constexpr (!SHARE_KV)
                    vv4 = *(const int4*)(vc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
            }
            *(int4*)(sK + j * kKPad + c8 * 8) = kv4;
            if constexpr (!SHARE_KV) *(int4*)(sV + j * kHD + c8 * 8) = vv4;
        }
        __syncthreads();
        {
            float s[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            const __half2* q0 = (const __half2*)(sQ + (rp * 2) * kQStride);
            const __half2* q1 = (const __half2*)(sQ + (rp * 2 + 1) * kQStride);
#pragma unroll 8
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 a0 = __half22float2(q0[d2]), a1 = __half22float2(q1[d2]);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 kf = __half22float2(((const __half2*)(sK + (PERMUTE_K ? kq + u * 8 : kq * 4 + u) * kKPad))[d2]);
                    s[0][u] += a0.x * kf.x + a0.y * kf.y;
                    s[1][u] += a1.x * kf.x + a1.y * kf.y;
                }
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int r = rp * 2 + a;
                const int qpos = p0 + tq0 + (r >> 3);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int kp = k0 + (PERMUTE_K ? kq + u * 8 : kq * 4 + u);
                    sS[r * kAK + (PERMUTE_K ? kq + u * 8 : kq * 4 + u)] = (kp <= qpos && (r >> 3) < nq) ? s[a][u] : -INFINITY;
                }
            }
        }
        __syncthreads();
        {   // online softmax: 4 threads per row, 8 scores each
            const int r = tid >> 2, part = tid & 3;
            float* row = sS + r * kAK + part * 8;
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) mx = fmaxf(mx, row[u]);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float m_old = sM[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float e = m_new == -INFINITY ? 0.0f : __expf(row[u] - m_new);
                row[u] = e;
                sum += e;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            __syncwarp();
            if (part == 0) {
                const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                sA[r] = alpha;
                sL[r] = sL[r] * alpha + sum;
                sM[r] = m_new;
            }
        }
        __syncthreads();
        if constexpr (SHARE_KV) {
            // The preceding barrier covers every K reader and softmax writer. K is now dead;
            // overwrite its larger padded allocation with V, then publish V to every accumulator.
            for (int i = tid; i < kAK * kHD / 8; i += AQ * 32) {
                const int j = i / (kHD / 8), c8 = i % (kHD / 8);
                int4 vv4 = make_int4(0, 0, 0, 0);
                if (k0 + j < n_keys)
                    vv4 = *(const int4*)(vc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
                *(int4*)(sV + j * kHD + c8 * 8) = vv4;
            }
            __syncthreads();
        }
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            const float al = sA[warp * 8 + a];
#pragma unroll
            for (int b = 0; b < 8; ++b) acc[a][b] *= al;
        }
        const int kn = min(kAK, n_keys - k0);
        for (int j = 0; j < kn; ++j) {
            const int4 raw = *(const int4*)(sV + j * kHD + lane * 8);
            const __half2* hv = (const __half2*)&raw;
            float vf[8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 f = __half22float2(hv[u]);
                vf[2 * u] = f.x;
                vf[2 * u + 1] = f.y;
            }
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const float p = sS[(warp * 8 + a) * kAK + j];
#pragma unroll
                for (int b = 0; b < 8; ++b) acc[a][b] += p * vf[b];
            }
        }
    }
    __syncthreads();
    // write: out[t][h*256 + d] = acc / l * sigmoid(gate)
#pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int r = warp * 8 + a;
        const int tok = r >> 3, h = kh * 8 + (r & 7);
        if (tok >= nq) continue;
        const float inv_l = 1.0f / sL[r];
        const float* gate = qkv + (size_t)(tq0 + tok) * ld + h * 2 * kHD + kHD;
        __align__(16) __half o[8];
#pragma unroll
        for (int b = 0; b < 8; ++b) o[b] = to_h(acc[a][b] * inv_l * sigm(gate[lane * 8 + b]));
        *(int4*)(out + (size_t)(tq0 + tok) * (kNH * kHD) + h * kHD + lane * 8) = *(int4*)o;
    }
}
