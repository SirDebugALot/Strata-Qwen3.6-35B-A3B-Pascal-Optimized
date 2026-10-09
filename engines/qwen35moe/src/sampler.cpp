// sampler.cpp - see sampler.hpp.
#include "sampler.hpp"

#include <algorithm>
#include <cmath>

namespace sq {

namespace {

/// The filtered distribution: p[0..n) over c.id[0..n) (unnormalised after the filters; returns the total).
double filtered(const Candidates& c, const SamplingParams& sp, float* p, int& n) {
    n = kCand;
    while (n > 1 && !(c.logit[n - 1] > -INFINITY)) --n;
    if (sp.top_k > 0) n = std::min(n, sp.top_k);
    const float mx = c.logit[0];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        p[i] = std::exp((c.logit[i] - mx) / sp.temperature);
        sum += p[i];
    }
    for (int i = 0; i < n; ++i) p[i] = (float)(p[i] / sum);
    // min-p: relative to the most likely token
    if (sp.min_p > 0.0f) {
        const float thr = sp.min_p * p[0];
        int m = 1;
        while (m < n && p[m] >= thr) ++m;
        n = m;
    }
    // top-p: smallest prefix whose mass reaches top_p
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        double cum = 0.0;
        int m = 0;
        while (m < n) {
            cum += p[m++];
            if (cum >= sp.top_p) break;
        }
        n = m;
    }
    double tot = 0.0;
    for (int i = 0; i < n; ++i) tot += p[i];
    return tot;
}

int draw(const Candidates& c, const float* p, int n, double tot, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    int last = 0;
    for (int i = 0; i < n; ++i) {
        if (p[i] <= 0.0f) continue;
        last = i;
        r -= p[i];
        if (r <= 0.0) return c.id[i];
    }
    return c.id[last];
}

}  // namespace

int sample_candidates(const Candidates& c, const SamplingParams& sp, std::mt19937_64& rng) {
    if (sp.temperature <= 0.0f) return c.id[0];
    float p[kCand];
    int n = 0;
    const double tot = filtered(c, sp, p, n);
    return draw(c, p, n, tot, rng);
}

int sample_speculative(const Candidates& c, const SamplingParams& sp, std::mt19937_64& rng, int draft, bool& accepted) {
    if (sp.temperature <= 0.0f) {
        accepted = c.id[0] == draft;
        return c.id[0];
    }
    float p[kCand];
    int n = 0;
    double tot = filtered(c, sp, p, n);
    int di = -1;
    for (int i = 0; i < n; ++i)
        if (c.id[i] == draft) { di = i; break; }
    if (di >= 0) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (u(rng) * tot < p[di]) {
            accepted = true;
            return draft;
        }
        tot -= p[di];
        p[di] = 0.0f;
    }
    accepted = false;
    if (!(tot > 0.0)) return c.id[0];   // numerically the draft had all the mass
    return draw(c, p, n, tot, rng);
}

}  // namespace sq
