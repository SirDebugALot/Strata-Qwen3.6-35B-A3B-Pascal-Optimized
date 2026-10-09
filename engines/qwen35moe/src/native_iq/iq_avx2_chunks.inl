// Exact opt-in chunk form of the upstream MIT iq_avx2_rows.inl row_dot loop.
// Included once per ISA namespace after iq_avx2_rows.inl. MSVC otherwise outlines row_dot and repeats its
// XMM/GPR save/restore plus prefetch-setting TLS guard for every row (particularly costly for 512-column down).
// Keep the old row functions intact. Only this variant forces inlining; no extra row accumulators, weight
// expansion, float reassociation, or change to per-token integer/FMA/hsum order is introduced.
#if defined(_MSC_VER)
#define STRATA_CHUNK_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STRATA_CHUNK_INLINE inline __attribute__((always_inline))
#else
#define STRATA_CHUNK_INLINE inline
#endif

template <int TY, int NT> STRATA_CHUNK_FN STRATA_CHUNK_INLINE
void row_dot_inline(const uint8_t* row, int nblocks, const sq::Q8K* const* y, float* res, int pf) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t)i * Fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*)(y[t][i].nat + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int TY, int NT, bool GU> STRATA_CHUNK_FN
void inline_chunk(const uint8_t* weights, size_t row_bytes, size_t up_off, int n, const void* const* act,
                  float* const* out, int r0, int r1) {
    const sq::Q8K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const sq::Q8K*)act[t];
    const int nb = n / QK_K, pf = prefetch_distance();
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_inline<TY, NT>(weights + (size_t)r * row_bytes, nb, y, g, pf);
        if constexpr (GU) {
            row_dot_inline<TY, NT>(weights + up_off + (size_t)r * row_bytes, nb, y, u, pf);
            for (int t = 0; t < NT; ++t) out[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        } else {
            for (int t = 0; t < NT; ++t) out[t][r] = g[t];
        }
    }
}

template <int TY, bool GU> STRATA_CHUNK_FN
void inline_chunk_nt(int nt, const uint8_t* weights, size_t row_bytes, size_t up_off, int n,
                     const void* const* act, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: inline_chunk<TY, 1, GU>(weights, row_bytes, up_off, n, act, out, r0, r1); break;
        case 2: inline_chunk<TY, 2, GU>(weights, row_bytes, up_off, n, act, out, r0, r1); break;
        case 3: inline_chunk<TY, 3, GU>(weights, row_bytes, up_off, n, act, out, r0, r1); break;
        case 4: inline_chunk<TY, 4, GU>(weights, row_bytes, up_off, n, act, out, r0, r1); break;
    }
}

template <bool GU> STRATA_CHUNK_FN
bool inline_chunk_type(bool gather, int type, int nt, const uint8_t* weights, size_t row_bytes, size_t up_off,
                       int n, const void* const* act, float* const* out, int r0, int r1, bool compact = false) {
    if (nt < 1 || nt > 4) return false;
    switch (type) {
        case 21:
            if (compact) {
                if (gather) inline_chunk_nt<321, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
                else inline_chunk_nt<221, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
                return true;
            }
            if (gather) inline_chunk_nt<121, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
            else inline_chunk_nt<21, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
            return true;
        case 22:
            if (compact) {
                if (gather) inline_chunk_nt<322, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
                else inline_chunk_nt<222, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
                return true;
            }
            if (gather) inline_chunk_nt<122, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
            else inline_chunk_nt<22, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1);
            return true;
        case 23: inline_chunk_nt<23, GU>(nt, weights, row_bytes, up_off, n, act, out, r0, r1); return true;
        default: return false;
    }
}

// Compact scales without the force-inline feature retain the existing row function and its call order.
// Limit instantiation to NT1..4 instead of expanding the upstream NT5..8 gate/up variants.
template <int TY, int NT, bool GU> STRATA_CHUNK_FN
void compact_chunk(const uint8_t* w, size_t rb, size_t up_off, int n, const void* const* act,
                   float* const* out, int r0, int r1) {
    if constexpr (GU) gu_rows<TY, NT>(w, rb, up_off, n, act, out, r0, r1);
    else dot_rows<TY, NT>(w, rb, n, act, out, r0, r1);
}
template <int TY, bool GU> STRATA_CHUNK_FN
void compact_chunk_nt(int nt, const uint8_t* w, size_t rb, size_t up_off, int n, const void* const* act,
                      float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: compact_chunk<TY, 1, GU>(w, rb, up_off, n, act, out, r0, r1); break;
        case 2: compact_chunk<TY, 2, GU>(w, rb, up_off, n, act, out, r0, r1); break;
        case 3: compact_chunk<TY, 3, GU>(w, rb, up_off, n, act, out, r0, r1); break;
        case 4: compact_chunk<TY, 4, GU>(w, rb, up_off, n, act, out, r0, r1); break;
    }
}
template <bool GU> STRATA_CHUNK_FN
bool compact_chunk_type(bool gather, int type, int nt, const uint8_t* w, size_t rb, size_t up_off, int n,
                        const void* const* act, float* const* out, int r0, int r1) {
    if (nt < 1 || nt > 4) return false;
    switch (type) {
        case 21:
            if (gather) compact_chunk_nt<321, GU>(nt, w, rb, up_off, n, act, out, r0, r1);
            else compact_chunk_nt<221, GU>(nt, w, rb, up_off, n, act, out, r0, r1);
            return true;
        case 22:
            if (gather) compact_chunk_nt<322, GU>(nt, w, rb, up_off, n, act, out, r0, r1);
            else compact_chunk_nt<222, GU>(nt, w, rb, up_off, n, act, out, r0, r1);
            return true;
        default: return false;
    }
}

// Pair one gate row with its up row only for a single activation. Both rows retain the original per-half
// integer accumulation and per-block float FMA order. Their independent chains now share a single natural-
// order activation load, with two FP and two integer accumulators instead of the 16 accumulators NT4 pairing
// would require. No weight expansion, activation repack, sum reassociation or SiLU approximation is used.
template <int TY> STRATA_CHUNK_FN
void paired_gu_chunk(const uint8_t* weights, size_t rb, size_t up_off, int n, const void* const* act,
                      float* const* out, int r0, int r1) {
    const sq::Q8K* y = (const sq::Q8K*)act[0];
    const int nb = n / QK_K, pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gate_row = weights + (size_t)r * rb;
        const uint8_t* up_row = gate_row + up_off;
        __m256 gate_f = _mm256_setzero_ps(), up_f = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gate_block = gate_row + (size_t)i * Fmt32<TY>::bytes;
            const uint8_t* up_block = up_row + (size_t)i * Fmt32<TY>::bytes;
            rows_ahead(gate_block, pf); rows_ahead(up_block, pf);
            __m256i gate_i = _mm256_setzero_si256(), up_i = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j) for (int half = 0; half < 2; ++half) {
                const __m256i yv = _mm256_loadu_si256((const __m256i*)(y[i].nat + 64 * j + 32 * half));
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(gate_block, j, half, g, sgn, sc);
                gate_i = madd_add(gate_i, _mm256_maddubs_epi16(g, _mm256_sign_epi8(yv, sgn)), sc);
                Fmt32<TY>::decode(up_block, j, half, g, sgn, sc);
                up_i = madd_add(up_i, _mm256_maddubs_epi16(g, _mm256_sign_epi8(yv, sgn)), sc);
            }
            const float gate_d = h2f(u16(gate_block)) * Fmt32<TY>::K;
            const float up_d = h2f(u16(up_block)) * Fmt32<TY>::K;
            gate_f = _mm256_fmadd_ps(_mm256_set1_ps(gate_d * y[i].d), _mm256_cvtepi32_ps(gate_i), gate_f);
            up_f = _mm256_fmadd_ps(_mm256_set1_ps(up_d * y[i].d), _mm256_cvtepi32_ps(up_i), up_f);
        }
        const float gate = hsum8(gate_f), up = hsum8(up_f);
        out[0][r] = (gate / (1.f + std::exp(-gate))) * up;
    }
}

STRATA_CHUNK_FN
bool paired_gu_type(bool gather, bool compact, int type, const uint8_t* w, size_t rb, size_t up_off, int n,
                    const void* const* act, float* const* out, int r0, int r1) {
    switch (type) {
        case 21:
            if (compact) {
                if (gather) paired_gu_chunk<321>(w, rb, up_off, n, act, out, r0, r1);
                else paired_gu_chunk<221>(w, rb, up_off, n, act, out, r0, r1);
            } else {
                if (gather) paired_gu_chunk<121>(w, rb, up_off, n, act, out, r0, r1);
                else paired_gu_chunk<21>(w, rb, up_off, n, act, out, r0, r1);
            }
            return true;
        case 22:
            if (compact) {
                if (gather) paired_gu_chunk<322>(w, rb, up_off, n, act, out, r0, r1);
                else paired_gu_chunk<222>(w, rb, up_off, n, act, out, r0, r1);
            } else {
                if (gather) paired_gu_chunk<122>(w, rb, up_off, n, act, out, r0, r1);
                else paired_gu_chunk<22>(w, rb, up_off, n, act, out, r0, r1);
            }
            return true;
        case 23: paired_gu_chunk<23>(w, rb, up_off, n, act, out, r0, r1); return true;
        default: return false;
    }
}

#undef STRATA_CHUNK_INLINE
#undef STRATA_CHUNK_FN
