// prefill.hpp - batched prompt processing on the GPU (tensor-core GEMMs, experts streamed over PCIe).
#pragma once

#include <string>

namespace sq {

class Engine;

class Prefill {
public:
    explicit Prefill(Engine& e) : e_(e) {}
    ~Prefill();
    bool init(std::string& err);
    bool ready() const { return ready_; }
    /// Processes `n` tokens at positions n_past .. n_past+n-1 of the engine; leaves the last token's logits.
    /// With MTP on, also runs the MTP layer over every position whose next token is known (the pending pairs
    /// first), and leaves the last position pending.
    void run(const int* tokens, int n);

private:
    /// One decoder layer (attention or delta net, then MoE) over the T rows of X at positions p0 .. p0+T-1.
    void layer(int il, int T, int p0);
    /// Long-prompt layer-major path: one layer traverses a bounded window of prompt tiles before the next layer.
    void run_layer_major(const int* tokens, int n);
    /// A window of the original prompt; explicit offsets preserve positions and MTP pairs across windows.
    void run_layer_major_window(const int* tokens, int total_n, int window_start, int n, bool ordered_combine);
    /// Mixer half using both expanded dense matrices prepared once for the whole layer.
    void mixer_prepared(int il, int T, int p0);
    /// The MTP pass after a chunk of T tokens (chunk offset `done` in tokens[0..n), first position p0).
    void mtp(const int* tokens, int n, int done, int T, int p0);

    Engine& e_;
    bool ready_ = false;
    struct Impl;
    Impl* p_ = nullptr;
};

}  // namespace sq
