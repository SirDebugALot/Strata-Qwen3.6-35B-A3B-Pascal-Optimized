// model.cpp - see model.hpp.
#include "model.hpp"

#include "common.hpp"
#include "dense_q6_layout.hpp"
#include "quant.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

namespace sq {

namespace {

template <typename F>
void parallel_for(int64_t n, F&& f, int n_threads = 0) {
    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = (int)std::min<int64_t>(n_threads, std::max<int64_t>(1, n));
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        th.emplace_back([&, t] {
            const int64_t a = n * t / n_threads, b = n * (t + 1) / n_threads;
            for (int64_t i = a; i < b; ++i) f(i);
        });
    }
    for (auto& x : th) x.join();
}

constexpr int64_t kAlign = 256;
int64_t align_up(int64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

int64_t q8_dev_bytes(int64_t rows, int64_t cols) { return align_up(rows * cols) + align_up(rows * cols / 32 * 2); }

std::string blk(int il, const char* name) { return "blk." + std::to_string(il) + "." + name; }

constexpr uint64_t kCacheHeaderBytes = 4096;
struct ExpertCacheHeader {
    char magic[8];
    uint64_t version, source_size, source_time, source_hash, expert_bytes;
    uint64_t n_layers, n_embd, n_ff, n_experts, completed_layers;
};

uint64_t hash_bytes(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

#ifdef _WIN32
std::wstring wide_path(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}
#endif

}  // namespace

Model::~Model() {
    if (head_dev_) cudaFree(head_dev_);
    if (dev_arena_) cudaFree(dev_arena_);
    for (auto* p : pinned_) cudaFreeHost(p);
#ifdef _WIN32
    if (expert_cache_base_) UnmapViewOfFile(expert_cache_base_);
    if (expert_cache_mapping_) CloseHandle((HANDLE)expert_cache_mapping_);
    if (expert_cache_file_) CloseHandle((HANDLE)expert_cache_file_);
#endif
}

int64_t Model::release_lm_head() {
    if (!phase_lm_head || !head_dev_) return 0;
    CUDA_CHECK(cudaFree(head_dev_));
    head_dev_ = nullptr;
    lm_head.qs = nullptr;
    lm_head.d = nullptr;
    return head_dev_bytes_;
}

void Model::restore_lm_head() {
    if (!phase_lm_head || head_dev_) return;
    SQ_CHECK(!head_qs_.empty() && !head_d_.empty(), "phase LM head host copy is missing");
    const size_t qb = head_qs_.size();
    const size_t db = head_d_.size() * sizeof(uint16_t);
    head_dev_bytes_ = (int64_t)(qb + db);
    CUDA_CHECK(cudaMalloc(&head_dev_, (size_t)head_dev_bytes_));
    CUDA_CHECK(cudaMemcpy(head_dev_, head_qs_.data(), qb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(head_dev_ + qb, head_d_.data(), db, cudaMemcpyHostToDevice));
    lm_head.qs = (const int8_t*)head_dev_;
    lm_head.d = (const uint16_t*)(head_dev_ + qb);
}

bool Model::open_expert_cache(const std::string& cache_path, const std::string& source_path,
                              int& completed_layers, std::string& err) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA attr{};
    if (!GetFileAttributesExW(wide_path(source_path).c_str(), GetFileExInfoStandard, &attr)) {
        err = "cannot stat source GGUF for expert cache";
        return false;
    }
    ExpertCacheHeader expected{};
    std::memcpy(expected.magic, "SQEXQ6K", 8);
    expected.version = 1;
    expected.source_size = gguf.file_size();
    expected.source_time = ((uint64_t)attr.ftLastWriteTime.dwHighDateTime << 32) | attr.ftLastWriteTime.dwLowDateTime;
    expected.source_hash = hash_bytes(1469598103934665603ull, source_path.data(), source_path.size());
    expected.source_hash = hash_bytes(expected.source_hash, gguf.base(), (size_t)std::min<uint64_t>(gguf.file_size(), 1 << 20));
    const uint32_t conversion_mode = fast_expert_conversion_ ? 1 : 0;
    expected.source_hash = hash_bytes(expected.source_hash, &conversion_mode, sizeof(conversion_mode));
    expected.expert_bytes = expert_bytes;
    expected.n_layers = cfg.n_moe_layers();
    expected.n_embd = cfg.n_embd;
    expected.n_ff = cfg.n_ff_exp;
    expected.n_experts = cfg.n_expert;
    // Include every output layer's type/layout in the identity, even for mixed K-quant GGUFs.
    for (const auto& l : layers) {
        expected.source_hash = hash_bytes(expected.source_hash, &l.t_gu, sizeof(l.t_gu));
        expected.source_hash = hash_bytes(expected.source_hash, &l.t_down, sizeof(l.t_down));
        expected.source_hash = hash_bytes(expected.source_hash, &l.blob_bytes, sizeof(l.blob_bytes));
    }
    HANDLE f = CreateFileW(wide_path(cache_path).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { err = "cannot open expert cache (Windows error " + std::to_string(GetLastError()) + ")"; return false; }
    expert_cache_file_ = f;
    LARGE_INTEGER size{};
    GetFileSizeEx(f, &size);
    ExpertCacheHeader old{};
    DWORD read = 0;
    if (size.QuadPart) {
        if (!ReadFile(f, &old, sizeof(old), &read, nullptr) || read != sizeof(old) ||
            std::memcmp(old.magic, expected.magic, sizeof(old.magic))) {
            err = "refusing to overwrite an unrecognized expert cache file";
            return false;
        }
    }
    const uint64_t bytes = kCacheHeaderBytes + (uint64_t)expert_bytes;
    completed_layers = 0;
    ExpertCacheHeader compare = old;
    compare.completed_layers = 0;
    const bool valid = size.QuadPart == (LONGLONG)bytes && !std::memcmp(&compare, &expected, sizeof(expected)) &&
                       old.completed_layers <= expected.n_layers;
    // A different quantization/layout or MTP layer count must use a different
    // cache path. Never silently truncate a working cache to rebuild another.
    if (size.QuadPart && !valid) {
        err = "expert cache layout/source differs; choose a new STRATA_EXPERT_CACHE_FILE path";
        return false;
    }
    if (valid) completed_layers = (int)old.completed_layers;
    LARGE_INTEGER end{};
    end.QuadPart = (LONGLONG)bytes;
    if (!SetFilePointerEx(f, end, nullptr, FILE_BEGIN) || !SetEndOfFile(f)) {
        err = "cannot size expert cache; check disk space";
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READWRITE, (DWORD)(bytes >> 32), (DWORD)bytes, nullptr);
    if (!m) { err = "cannot map expert cache file"; return false; }
    expert_cache_mapping_ = m;
    expert_cache_base_ = (uint8_t*)MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!expert_cache_base_) { err = "cannot map expert cache view"; return false; }
    if (!valid) {
        std::memset(expert_cache_base_, 0, kCacheHeaderBytes);
        std::memcpy(expert_cache_base_, &expected, sizeof(expected));
        if (!FlushViewOfFile(expert_cache_base_, kCacheHeaderBytes) || !FlushFileBuffers(f)) {
            err = "cannot initialize expert cache header";
            return false;
        }
    }
    log("expert cache: %s; %d/%d layers reusable (pageable file mapping)", cache_path.c_str(),
        completed_layers, cfg.n_moe_layers());
    return true;
#else
    (void)cache_path; (void)source_path; (void)completed_layers;
    err = "STRATA_EXPERT_CACHE_FILE is currently supported on Windows only";
    return false;
#endif
}

bool Model::flush_expert_cache_layer(int il, std::string& err) {
#ifdef _WIN32
    const auto& l = layers[il];
    if (!FlushViewOfFile(l.experts, (SIZE_T)(l.blob_bytes * cfg.n_expert)) ||
        !FlushFileBuffers((HANDLE)expert_cache_file_)) {
        err = "cannot flush converted expert layer " + std::to_string(il);
        return false;
    }
    auto* h = (ExpertCacheHeader*)expert_cache_base_;
    h->completed_layers = (uint64_t)il + 1;
    if (!FlushViewOfFile(expert_cache_base_, kCacheHeaderBytes) || !FlushFileBuffers((HANDLE)expert_cache_file_)) {
        err = "cannot checkpoint expert cache progress";
        return false;
    }
    return true;
#else
    (void)il; (void)err;
    return false;
#endif
}

void* Model::dev_alloc(int64_t bytes) {
    bytes = align_up(bytes);
    SQ_CHECK(dev_used_ + bytes <= dev_cap_, "device weight arena overflow");
    void* p = dev_arena_ + dev_used_;
    dev_used_ += bytes;
    return p;
}

int64_t Model::dense_upload_bytes(const std::vector<const GgufTensor*>& parts) const {
    const int64_t cols = parts[0]->ne[0];
    int64_t rows = 0;
    bool all_q6 = native_dense_q6 && cols % 256 == 0;
    for (auto* t : parts) {
        rows += t->n_elements() / cols;
        all_q6 = all_q6 && t->type == T_Q6_K;
    }
    const int64_t q6_row = native_dense_q6_repack ? (int64_t)dense_q6_plane_row_bytes((int)cols) : cols / 256 * 210;
    return all_q6 ? align_up(rows * q6_row) : q8_dev_bytes(rows, cols);
}

DQ8 Model::upload_q8(const std::vector<const GgufTensor*>& parts, bool host) {
    const int64_t cols = parts[0]->ne[0];
    int64_t rows = 0;
    for (auto* t : parts) {
        // Q8_0 is copied as stored; any other type the dequantizer knows (Q4_K..Q6_K, F16, BF16, F32 - the dense
        // tensors of the other quantizations, e.g. Q4_K_M's Q6_K output head) is converted to Q8_0 at load
        SQ_CHECK(t->type == T_Q8_0 || can_dequant(t->type), "%s: unsupported type %s", t->name.c_str(),
                 type_name(t->type));
        SQ_CHECK(t->ne[0] == cols && cols % 32 == 0, "%s: column mismatch", t->name.c_str());
        rows += t->n_elements() / cols;
    }
    if (native_dense_q6 && !host && cols % 256 == 0 &&
        std::all_of(parts.begin(), parts.end(), [](const GgufTensor* t) { return t->type == T_Q6_K; })) {
        // Both layouts preserve every original Q6 quant/scale byte and fused tensor row order. The arena
        // prepass uses this identical all-parts rule and row stride; mixed Q6/Q8 fusions retain the old Q8 path.
        const int64_t source_row = cols / 256 * 210;
        const int64_t device_row = native_dense_q6_repack ? (int64_t)dense_q6_plane_row_bytes((int)cols) : source_row;
        uint8_t* destination = (uint8_t*)dev_alloc(rows * device_row);
        int64_t offset = 0;
        for (auto* t : parts) {
            const int64_t part_rows = t->n_elements() / cols;
            if (native_dense_q6_repack) {
                // Bound temporary CPU memory even for the full vocabulary head; no persistent duplicate
                // weights or expert-cache file changes are needed. cudaMemcpy consumes each pageable chunk.
                const int64_t chunk_rows = std::max<int64_t>(1, std::min<int64_t>(part_rows, (8ll << 20) / device_row));
                std::vector<uint8_t> staging((size_t)(chunk_rows * device_row));
                for (int64_t first = 0; first < part_rows; first += chunk_rows) {
                    const int64_t count = std::min(chunk_rows, part_rows - first);
                    parallel_for(count, [&](int64_t row) {
                        dense_q6_pack_row_planes(t->data + (first + row) * source_row,
                                                 staging.data() + row * device_row, (int)cols);
                    });
                    const size_t bytes = (size_t)(count * device_row);
                    CUDA_CHECK(cudaMemcpy(destination + offset, staging.data(), bytes, cudaMemcpyHostToDevice));
                    offset += bytes;
                }
            } else {
                const int64_t bytes = part_rows * source_row;
                CUDA_CHECK(cudaMemcpy(destination + offset, t->data, (size_t)bytes, cudaMemcpyHostToDevice));
                offset += bytes;
            }
        }
        SQ_CHECK(offset == rows * device_row, "native dense Q6 upload size differs from arena prepass");
        DQ8 matrix;
        matrix.rows = (int)rows;
        matrix.cols = (int)cols;
        matrix.q6 = destination;
        matrix.q6_row_planes = native_dense_q6_repack;
        return matrix;
    }
    const int64_t nb = cols / 32;
    std::vector<int8_t> qs((size_t)(rows * cols));
    std::vector<uint16_t> d((size_t)(rows * nb));
    int64_t r0 = 0;
    for (auto* t : parts) {
        const int64_t tr = t->n_elements() / cols;
        if (t->type == T_Q8_0) {
            const BlockQ8_0* src = (const BlockQ8_0*)t->data;
            parallel_for(tr, [&](int64_t r) {
                const BlockQ8_0* b = src + r * nb;
                int8_t* q = qs.data() + (r0 + r) * cols;
                uint16_t* dd = d.data() + (r0 + r) * nb;
                for (int64_t i = 0; i < nb; ++i) {
                    dd[i] = b[i].d;
                    std::memcpy(q + i * 32, b[i].qs, 32);
                }
            });
        } else {
            const int64_t rb = row_bytes(t->type, cols);
            parallel_for(tr, [&](int64_t r) {
                std::vector<float> f((size_t)cols);
                dequant_row(t->type, t->data + r * rb, f.data(), cols);
                quantize_row_q8_0(f.data(), qs.data() + (r0 + r) * cols, d.data() + (r0 + r) * nb, cols);
            });
            log("%s: %s converted to Q8_0 for the GPU", t->name.c_str(), type_name(t->type));
        }
        r0 += tr;
    }
    DQ8 m;
    m.rows = (int)rows;
    m.cols = (int)cols;
    if (host) {
        head_qs_ = std::move(qs);
        head_d_ = std::move(d);
        m.qs = head_qs_.data();
        m.d = head_d_.data();
        return m;
    }
    int8_t* dq = (int8_t*)dev_alloc(rows * cols);
    uint16_t* dd = (uint16_t*)dev_alloc(rows * nb * 2);
    CUDA_CHECK(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
    m.qs = dq;
    m.d = dd;
    return m;
}

DF32 Model::upload_f32(const std::vector<const GgufTensor*>& parts) {
    const int64_t cols = parts[0]->ne[0];
    int64_t rows = 0;
    for (auto* t : parts) {
        SQ_CHECK(t->type == T_F32 || t->type == T_BF16 || t->type == T_F16, "%s: expected F32", t->name.c_str());
        SQ_CHECK(t->ne[0] == cols || t->n_elements() == cols, "%s: column mismatch", t->name.c_str());
        rows += t->n_elements() / cols;
    }
    float* w = (float*)dev_alloc(rows * cols * 4);
    int64_t off = 0;
    for (auto* t : parts) {
        if (t->type == T_F32) {
            CUDA_CHECK(cudaMemcpy(w + off, t->data, t->n_elements() * 4, cudaMemcpyHostToDevice));
        } else {   // the MTP block's router is stored as BF16
            std::vector<float> f((size_t)t->n_elements());
            dequant_row(t->type, t->data, f.data(), t->n_elements());
            CUDA_CHECK(cudaMemcpy(w + off, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
        }
        off += t->n_elements();
    }
    DF32 m;
    m.w = w;
    m.rows = (int)rows;
    m.cols = (int)cols;
    return m;
}

const float* Model::upload_vec(const GgufTensor& t) {
    SQ_CHECK(t.type == T_F32 || t.type == T_F16 || t.type == T_BF16, "%s: expected F32, got %s", t.name.c_str(),
             type_name(t.type));
    float* w = (float*)dev_alloc(t.n_elements() * 4);
    if (t.type == T_F32) {
        CUDA_CHECK(cudaMemcpy(w, t.data, t.n_elements() * 4, cudaMemcpyHostToDevice));
    } else {
        std::vector<float> f((size_t)t.n_elements());
        dequant_row(t.type, t.data, f.data(), t.n_elements());
        CUDA_CHECK(cudaMemcpy(w, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    }
    return w;
}

bool Model::load(const std::string& path, bool with_mtp, std::string& err) {
    const double t0 = now_ms();
    const char* fast_env = std::getenv("STRATA_EXPERT_CONVERSION_FAST");
    fast_expert_conversion_ = fast_env && std::string(fast_env) == "1";
    if (!gguf.open(path, err)) return false;
    const std::string arch = gguf.str("general.architecture");
    if (arch != "qwen35moe") { err = "unsupported architecture '" + arch + "' (this engine runs qwen35moe only)"; return false; }
    Config& c = cfg;
    c.n_layer = (int)gguf.i64("qwen35moe.block_count");
    const int nextn = (int)gguf.num("qwen35moe.nextn_predict_layers", 0);
    c.n_layer -= nextn;   // the MTP blocks come after the main layers
    c.n_mtp = with_mtp && nextn > 0 && gguf.tensor(blk(c.n_layer, "nextn.eh_proj.weight")) ? 1 : 0;
    c.n_embd = (int)gguf.i64("qwen35moe.embedding_length");
    c.n_head = (int)gguf.i64("qwen35moe.attention.head_count");
    c.n_head_kv = (int)gguf.i64("qwen35moe.attention.head_count_kv");
    c.head_dim = (int)gguf.i64("qwen35moe.attention.key_length");
    c.n_rot = (int)gguf.i64("qwen35moe.rope.dimension_count");
    c.rope_base = (float)gguf.num("qwen35moe.rope.freq_base", 1e7);
    c.eps = (float)gguf.num("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6);
    c.n_expert = (int)gguf.i64("qwen35moe.expert_count");
    c.n_expert_used = (int)gguf.i64("qwen35moe.expert_used_count");
    c.n_ff_exp = (int)gguf.i64("qwen35moe.expert_feed_forward_length");
    c.n_ff_shexp = (int)gguf.i64("qwen35moe.expert_shared_feed_forward_length");
    c.ssm_d_conv = (int)gguf.i64("qwen35moe.ssm.conv_kernel");
    c.ssm_d_state = (int)gguf.i64("qwen35moe.ssm.state_size");
    c.ssm_n_kh = (int)gguf.i64("qwen35moe.ssm.group_count");
    c.ssm_n_vh = (int)gguf.i64("qwen35moe.ssm.time_step_rank");
    c.ssm_d_inner = (int)gguf.i64("qwen35moe.ssm.inner_size");
    c.full_attn_interval = (int)gguf.num("qwen35moe.full_attention_interval", 4);
    c.ctx_train = (int)gguf.num("qwen35moe.context_length", 262144);
    const GgufTensor& te = gguf.need("token_embd.weight");
    c.n_vocab = (int)te.ne[1];

    // The kernels are written for this model's shapes; refuse anything else rather than mis-index.
    SQ_CHECK(c.n_embd == 2048 && c.head_dim == 256 && c.n_head == 16 && c.n_head_kv == 2 && c.n_rot == 64 &&
             c.n_expert == 256 && c.n_expert_used == 8 && c.n_ff_exp == 512 && c.n_ff_shexp == 512 &&
             c.ssm_d_state == 128 && c.ssm_n_kh == 16 && c.ssm_n_vh == 32 && c.ssm_d_conv == 4,
             "model shape differs from Qwen3.6-35B-A3B; this engine is specialised for it");

    // ---------------------------------------------------------------- size the device arena
    const char* cpu_head_env = std::getenv("STRATA_CPU_LM_HEAD");
    cpu_lm_head = cpu_head_env && std::string(cpu_head_env) == "1";
    const char* phase_head_env = std::getenv("STRATA_PREFILL_HEAD_LOAN");
    phase_lm_head = !cpu_lm_head && phase_head_env && std::string(phase_head_env) == "1";
    const char* native_dense_env = std::getenv("STRATA_NATIVE_DENSE_Q6");
    native_dense_q6 = native_dense_env && std::string(native_dense_env) == "1";
    const char* repack_env = std::getenv("STRATA_DENSE_Q6_REPACK");
    native_dense_q6_repack = repack_env && std::string(repack_env) == "1";
    if (native_dense_q6_repack && !native_dense_q6) {
        err = "STRATA_DENSE_Q6_REPACK=1 requires STRATA_NATIVE_DENSE_Q6=1";
        return false;
    }
    if (native_dense_q6 && cpu_lm_head) {
        err = "STRATA_NATIVE_DENSE_Q6 requires the GPU LM head (STRATA_CPU_LM_HEAD=0)";
        return false;
    }
    int64_t need = 0;
    auto q8 = [&](std::initializer_list<const GgufTensor*> parts) { need += dense_upload_bytes(parts); };
    auto f32 = [&](int64_t n) { need += align_up(n * 4); };
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        f32(c.n_embd); f32(c.n_embd);
        if (c.is_attn(il)) {
            q8({&gguf.need(blk(il, "attn_q.weight")), &gguf.need(blk(il, "attn_k.weight")),
                &gguf.need(blk(il, "attn_v.weight"))});
            f32(c.head_dim); f32(c.head_dim);
            q8({&gguf.need(blk(il, "attn_output.weight"))});
        } else {
            q8({&gguf.need(blk(il, "attn_qkv.weight")), &gguf.need(blk(il, "attn_gate.weight"))});
            f32(2 * c.ssm_n_vh * c.n_embd);
            f32(c.conv_dim() * c.ssm_d_conv); f32(c.ssm_n_vh); f32(c.ssm_n_vh); f32(c.ssm_d_state);
            q8({&gguf.need(blk(il, "ssm_out.weight"))});
        }
        f32((c.n_expert + 1) * c.n_embd);
        q8({&gguf.need(blk(il, "ffn_gate_shexp.weight")), &gguf.need(blk(il, "ffn_up_shexp.weight"))});
        q8({&gguf.need(blk(il, "ffn_down_shexp.weight"))});
    }
    f32(c.n_embd);
    const GgufTensor* output = gguf.tensor("output.weight");
    if (!cpu_lm_head && !phase_lm_head) q8({output ? output : &te});
    if (c.n_mtp) {
        f32(c.n_embd); f32(c.n_embd); f32(c.n_embd);
        q8({&gguf.need(blk(c.n_layer, "nextn.eh_proj.weight"))});
    }
    dev_cap_ = need + (1 << 20);
    CUDA_CHECK(cudaMalloc(&dev_arena_, dev_cap_));
    dense_bytes = dev_cap_;
    if (native_dense_q6) log("dense Q6: original packed GGUF weights on GPU; other dense types retain Q8/F32 imports");
    if (native_dense_q6_repack) log("dense Q6: lossless row-word planes with 32-byte row alignment; original values/scales retained");

    tok_embd_type = te.type;
    tok_embd_row_bytes = row_bytes(te.type, c.n_embd);
    SQ_CHECK(te.type == T_Q8_0 || te.type == T_F32 || te.type == T_F16 || te.type == T_Q6_K || te.type == T_Q4_K ||
             te.type == T_Q5_K, "unsupported token_embd type");

    layers.resize(c.n_moe_layers());
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        L.attn = c.is_attn(il);
        L.attn_norm = upload_vec(gguf.need(blk(il, "attn_norm.weight")));
        L.post_norm = upload_vec(gguf.need(blk(il, "post_attention_norm.weight")));
        if (L.attn) {
            L.qkv = upload_q8({&gguf.need(blk(il, "attn_q.weight")), &gguf.need(blk(il, "attn_k.weight")),
                               &gguf.need(blk(il, "attn_v.weight"))});
            L.q_norm = upload_vec(gguf.need(blk(il, "attn_q_norm.weight")));
            L.k_norm = upload_vec(gguf.need(blk(il, "attn_k_norm.weight")));
            L.wo = upload_q8({&gguf.need(blk(il, "attn_output.weight"))});
        } else {
            L.qkvz = upload_q8({&gguf.need(blk(il, "attn_qkv.weight")), &gguf.need(blk(il, "attn_gate.weight"))});
            L.ba = upload_f32({&gguf.need(blk(il, "ssm_beta.weight")), &gguf.need(blk(il, "ssm_alpha.weight"))});
            L.conv_w = upload_vec(gguf.need(blk(il, "ssm_conv1d.weight")));
            L.dt_bias = upload_vec(gguf.need(blk(il, "ssm_dt.bias")));
            L.ssm_a = upload_vec(gguf.need(blk(il, "ssm_a")));
            L.ssm_norm = upload_vec(gguf.need(blk(il, "ssm_norm.weight")));
            L.ssm_out = upload_q8({&gguf.need(blk(il, "ssm_out.weight"))});
        }
        L.router = upload_f32({&gguf.need(blk(il, "ffn_gate_inp.weight")), &gguf.need(blk(il, "ffn_gate_inp_shexp.weight"))});
        L.sh_gu = upload_q8({&gguf.need(blk(il, "ffn_gate_shexp.weight")), &gguf.need(blk(il, "ffn_up_shexp.weight"))});
        L.sh_down = upload_q8({&gguf.need(blk(il, "ffn_down_shexp.weight"))});
    }
    output_norm = upload_vec(gguf.need("output_norm.weight"));
    const GgufTensor* out = gguf.tensor("output.weight");
    lm_head = upload_q8({out ? out : &te}, cpu_lm_head || phase_lm_head);
    if (phase_lm_head) {
        restore_lm_head();
        dense_bytes += head_dev_bytes_;
        log("LM head: %.1f MiB phase-loan allocation with host Q8 restore copy", head_dev_bytes_ / 1048576.0);
    }
    if (cpu_lm_head) log("LM head: Q8_0 in host RAM (STRATA_CPU_LM_HEAD=1)");
    if (c.n_mtp) {
        const int il = c.n_layer;
        mtp.enorm = upload_vec(gguf.need(blk(il, "nextn.enorm.weight")));
        mtp.hnorm = upload_vec(gguf.need(blk(il, "nextn.hnorm.weight")));
        const GgufTensor* hn = gguf.tensor(blk(il, "nextn.shared_head_norm.weight"));
        mtp.head_norm = hn ? upload_vec(*hn) : output_norm;
        mtp.eh_proj = upload_q8({&gguf.need(blk(il, "nextn.eh_proj.weight"))});
        SQ_CHECK(mtp.eh_proj.rows == c.n_embd && mtp.eh_proj.cols == 2 * c.n_embd, "MTP eh_proj shape");
        SQ_CHECK(!gguf.tensor(blk(il, "nextn.embed_tokens.weight")) && !gguf.tensor(blk(il, "nextn.shared_head_head.weight")),
                 "MTP block with its own embedding / head is not supported");
        log("MTP block loaded (layer %d)", il);
    }
    const double t1 = now_ms();
    log("dense weights on GPU: %.2f GiB (%.1f s)", dense_bytes / 1073741824.0, (t1 - t0) / 1000);

    // ------------------------------------------------ routed experts -> pinned RAM or a pageable disk cache
    expert_bytes = 0;
    const char* native_env = std::getenv("STRATA_NATIVE_EXPERTS");
    const bool native_iq = native_env && std::atoi(native_env) != 0;
    auto compute_type = [native_iq](uint32_t t) {
        if (native_iq) {
            SQ_CHECK(t == T_IQ2_S || t == T_IQ3_S || t == T_IQ4_XS || t == T_Q2_K || t == T_Q3_K ||
                     t == T_Q4_K || t == T_Q5_K || t == T_Q6_K,
                     "native expert kernel unavailable for %s; refusing implicit requantization", type_name(t));
            return t;
        }
        return t == T_Q4_K || t == T_Q5_K || t == T_Q6_K ? t : (uint32_t)T_Q6_K;
    };
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        const GgufTensor& g = gguf.need(blk(il, "ffn_gate_exps.weight"));
        const GgufTensor& u = gguf.need(blk(il, "ffn_up_exps.weight"));
        const GgufTensor& d = gguf.need(blk(il, "ffn_down_exps.weight"));
        SQ_CHECK(g.type == u.type, "layer %d: gate/up types differ", il);
        for (uint32_t t : {g.type, d.type})
            SQ_CHECK(can_dequant(t), "layer %d: unsupported expert import type %s", il, type_name(t));
        SQ_CHECK(g.ne[0] == c.n_embd && g.ne[1] == c.n_ff_exp && d.ne[0] == c.n_ff_exp && d.ne[1] == c.n_embd, "expert shape");
        L.t_gu = compute_type(g.type);
        L.t_down = compute_type(d.type);
        L.gu_bytes = row_bytes(L.t_gu, c.n_embd) * c.n_ff_exp;
        L.down_bytes = row_bytes(L.t_down, c.n_ff_exp) * c.n_embd;
        L.blob_bytes = 2 * L.gu_bytes + L.down_bytes;
        expert_bytes += L.blob_bytes * c.n_expert;
    }
    // "expert arena:" and "loaded ... GiB at" are what serve/server.py narrates while the engine starts
    const char* cache_path = std::getenv("STRATA_EXPERT_CACHE_FILE");
    const bool mapped_cache = cache_path && *cache_path;
    int completed_layers = 0;
    if (native_iq) log("expert storage: original native packed weights (IQ2_S/IQ3_S/IQ4_XS/Q2_K/Q3_K/K-quants); no requantization");
    else log("expert storage: legacy K-quants; fallback Q6_K %s",
             fast_expert_conversion_ ? "fast weighted fit" : "reference scale search");
    if (mapped_cache && !open_expert_cache(cache_path, path, completed_layers, err)) return false;
    log("expert arena: %.2f GiB of %s for %d layers x %d experts", expert_bytes / 1073741824.0,
        mapped_cache ? "pageable file-backed storage" : "pinned RAM", c.n_moe_layers(), c.n_expert);
    int64_t cache_offset = kCacheHeaderBytes;
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        if (mapped_cache) {
            L.experts = expert_cache_base_ + cache_offset;
            L.experts_pinned = false;
            cache_offset += L.blob_bytes * c.n_expert;
            continue;
        }
        uint8_t* p = nullptr;
        cudaError_t e = cudaHostAlloc((void**)&p, (size_t)(L.blob_bytes * c.n_expert), cudaHostAllocPortable);
        if (e != cudaSuccess) {
            err = "cannot pin " + std::to_string(L.blob_bytes * c.n_expert >> 20) + " MiB of RAM for layer " +
                  std::to_string(il) + " experts: " + cudaGetErrorString(e);
            return false;
        }
        pinned_.push_back(p);
        L.experts = p;
        L.experts_pinned = true;
    }
    // Copy supported types; import other types using the GGML reference Q6_K quantizer.
    // One row of float scratch per worker avoids ever materializing an F32 expert/model.
    // The source GGUF is read-only. A flushed completion marker makes interrupted cache creation resumable.
    for (int il = completed_layers; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        const GgufTensor& g = gguf.need(blk(il, "ffn_gate_exps.weight"));
        const GgufTensor& u = gguf.need(blk(il, "ffn_up_exps.weight"));
        const GgufTensor& d = gguf.need(blk(il, "ffn_down_exps.weight"));
        const double layer_start = now_ms();
        log("expert layer %d/%d: %s/%s -> %s/%s", il + 1, c.n_moe_layers(),
            type_name(g.type), type_name(d.type), type_name(L.t_gu), type_name(L.t_down));
        auto copy_matrix = [&](const GgufTensor& t, uint32_t target, int64_t e, uint8_t* dst) {
            const int64_t cols = t.ne[0], rows = t.ne[1];
            const int64_t src_rb = row_bytes(t.type, cols), dst_rb = row_bytes(target, cols);
            const uint8_t* src = t.data + e * src_rb * rows;
            if (t.type == target) {
                std::memcpy(dst, src, (size_t)(src_rb * rows));
            } else {
                std::vector<float> f((size_t)cols);
                for (int64_t r = 0; r < rows; ++r) {
                    dequant_row(t.type, src + r * src_rb, f.data(), cols);
                    quantize_row_q6_K(f.data(), dst + r * dst_rb, cols, fast_expert_conversion_);
                }
            }
        };
        parallel_for(c.n_expert, [&](int64_t e) {
            uint8_t* dst = L.experts + e * L.blob_bytes;
            copy_matrix(g, L.t_gu, e, dst);
            copy_matrix(u, L.t_gu, e, dst + L.gu_bytes);
            copy_matrix(d, L.t_down, e, dst + 2 * L.gu_bytes);
        });
        if (mapped_cache && !flush_expert_cache_layer(il, err)) return false;
        log("expert layer %d ready (%.1f s)", il + 1, (now_ms() - layer_start) / 1000);
    }
    // Everything the engine needs from the file is copied now, so the mapping is closed: it would otherwise keep the
    // ~20 GB of pages the load touched in this process's working set (reclaimable, but counted as used RAM and
    // pushing other programs out).  The embedding is the one tensor still read per token: a copy of it stays.
    const GgufTensor& emb = gguf.need("token_embd.weight");
    tok_embd_copy_.assign(emb.data, emb.data + (size_t)tok_embd_row_bytes * c.n_vocab);
    tok_embd = tok_embd_copy_.data();
    gguf.close();
    const double secs = (now_ms() - t1) / 1000;
    log("experts loaded %.2f GiB at %.1f GiB/s (%.1f s)", expert_bytes / 1073741824.0,
        secs > 0 ? expert_bytes / 1073741824.0 / secs : 0.0, secs);
    return true;
}

void Model::embed(int token, float* out) const {
    SQ_CHECK(token >= 0 && token < cfg.n_vocab, "token id %d out of range", token);
    dequant_row(tok_embd_type, tok_embd + (int64_t)token * tok_embd_row_bytes, out, cfg.n_embd);
}

}  // namespace sq
