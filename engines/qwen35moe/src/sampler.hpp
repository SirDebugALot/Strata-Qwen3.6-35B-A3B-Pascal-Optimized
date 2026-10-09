// sampler.hpp - temperature / top-k / top-p / min-p sampling over the GPU's top candidates.
#pragma once

#include "engine.hpp"

#include <random>

namespace sq {

/// `c` holds the kCand highest (penalised) logits, sorted descending.
int sample_candidates(const Candidates& c, const SamplingParams& sp, std::mt19937_64& rng);

/// Speculative sampling against a greedy (deterministic) draft: with p the distribution sample_candidates() draws
/// from, accepts `draft` with probability p(draft) (accepted = true, returns draft), otherwise returns a token drawn
/// from p with the draft removed.  The result is distributed exactly as sample_candidates() would be.
int sample_speculative(const Candidates& c, const SamplingParams& sp, std::mt19937_64& rng, int draft, bool& accepted);

}  // namespace sq
