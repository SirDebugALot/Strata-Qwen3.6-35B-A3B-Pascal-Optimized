// gguf.cpp - see gguf.hpp.
#include "gguf.hpp"

#include "common.hpp"
#include "quant.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstring>

namespace sq {

namespace {

enum : uint32_t {
    V_U8 = 0, V_I8, V_U16, V_I16, V_U32, V_I32, V_F32, V_BOOL, V_STR, V_ARR, V_U64, V_I64, V_F64,
};

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <typename T> T get() {
        T v{};
        if (p + sizeof(T) > end) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || p + n > end) { ok = false; return {}; }
        std::string s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
};

double read_num(Cursor& c, uint32_t t) {
    switch (t) {
        case V_U8: return c.get<uint8_t>();
        case V_I8: return c.get<int8_t>();
        case V_U16: return c.get<uint16_t>();
        case V_I16: return c.get<int16_t>();
        case V_U32: return c.get<uint32_t>();
        case V_I32: return c.get<int32_t>();
        case V_F32: return c.get<float>();
        case V_BOOL: return c.get<uint8_t>();
        case V_U64: return (double)c.get<uint64_t>();
        case V_I64: return (double)c.get<int64_t>();
        case V_F64: return c.get<double>();
        default: c.ok = false; return 0;
    }
}

}  // namespace

GgufFile::~GgufFile() { close(); }

void GgufFile::close() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (hmap_) CloseHandle((HANDLE)hmap_);
    if (hfile_) CloseHandle((HANDLE)hfile_);
#else
    if (base_) munmap((void*)base_, size_);
    if (fd_ >= 0) ::close(fd_);
#endif
    base_ = nullptr;
    hmap_ = hfile_ = nullptr;
    fd_ = -1;
    for (auto& t : tensors_) t.data = nullptr;
}

bool GgufFile::open(const std::string& path, std::string& err) {
#ifdef _WIN32
    const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wn, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wn);
    HANDLE f = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) { err = "cannot open " + path; return false; }
    hfile_ = f;
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    size_ = (uint64_t)sz.QuadPart;
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { err = "CreateFileMapping failed"; return false; }
    hmap_ = m;
    base_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base_) { err = "MapViewOfFile failed"; return false; }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "cannot open " + path; return false; }
    struct stat st;
    fstat(fd_, &st);
    size_ = (uint64_t)st.st_size;
    void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) { err = "mmap failed"; return false; }
    base_ = (const uint8_t*)p;
#endif
    Cursor c{base_, base_ + size_};
    if (c.get<uint32_t>() != 0x46554747u) { err = "not a GGUF file"; return false; }
    const uint32_t version = c.get<uint32_t>();
    if (version != 3) { err = "unsupported GGUF version " + std::to_string(version); return false; }
    const uint64_t n_tensors = c.get<uint64_t>();
    const uint64_t n_kv = c.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv && c.ok; ++i) {
        std::string key = c.str();
        GgufValue v;
        v.type = c.get<uint32_t>();
        if (v.type == V_STR) {
            v.str = c.str();
        } else if (v.type == V_ARR) {
            v.is_array = true;
            v.type = c.get<uint32_t>();
            v.arr_len = c.get<uint64_t>();
            if (v.type == V_STR) {
                for (uint64_t j = 0; j < v.arr_len && c.ok; ++j) {
                    const uint64_t n = c.get<uint64_t>();
                    c.p += n;
                }
            } else {
                v.arr.reserve((size_t)v.arr_len);
                for (uint64_t j = 0; j < v.arr_len && c.ok; ++j) v.arr.push_back(read_num(c, v.type));
            }
        } else {
            v.num = read_num(c, v.type);
        }
        kv_[key] = std::move(v);
    }
    tensors_.resize((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors && c.ok; ++i) {
        GgufTensor& t = tensors_[(size_t)i];
        t.name = c.str();
        const uint32_t nd = c.get<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.ne.push_back((int64_t)c.get<uint64_t>());
        t.type = c.get<uint32_t>();
        t.offset = c.get<uint64_t>();
        by_name_[t.name] = (size_t)i;
    }
    if (!c.ok) { err = "truncated GGUF header"; return false; }
    const uint64_t align = (uint64_t)num("general.alignment", 32);
    uint64_t data_off = (uint64_t)(c.p - base_);
    data_off = (data_off + align - 1) / align * align;
    for (auto& t : tensors_) {
        t.data = base_ + data_off + t.offset;
        t.bytes = row_bytes(t.type, t.ne[0]);
        if (t.bytes < 0) { err = "unsupported tensor type " + std::to_string(t.type) + " for " + t.name; return false; }
        t.bytes *= t.n_elements() / t.ne[0];
        if (data_off + t.offset + (uint64_t)t.bytes > size_) { err = "tensor " + t.name + " out of file bounds"; return false; }
    }
    return true;
}

const GgufTensor* GgufFile::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const GgufTensor& GgufFile::need(const std::string& name) const {
    const GgufTensor* t = tensor(name);
    if (!t) die("missing tensor %s", name.c_str());
    return *t;
}

double GgufFile::num(const std::string& key, double def) const {
    auto it = kv_.find(key);
    return (it == kv_.end() || it->second.is_array) ? def : it->second.num;
}

int64_t GgufFile::i64(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) die("missing metadata key %s", key.c_str());
    return (int64_t)it->second.num;
}

std::string GgufFile::str(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second.str;
}

std::vector<double> GgufFile::arr(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? std::vector<double>{} : it->second.arr;
}

}  // namespace sq
