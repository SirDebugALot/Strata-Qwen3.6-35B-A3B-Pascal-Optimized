// gguf.hpp - a memory-mapped GGUF v3 reader: metadata scalars/arrays and the tensor directory.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sq {

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;   // ggml order: ne[0] is the contiguous (row) dimension
    uint64_t offset = 0;       // relative to the data section
    const uint8_t* data = nullptr;
    int64_t n_elements() const { int64_t n = 1; for (auto v : ne) n *= v; return n; }
    int64_t bytes = 0;
};

struct GgufValue {
    uint32_t type = 0;           // gguf value type of the scalar or of the array elements
    bool is_array = false;
    double num = 0;              // numeric scalar
    std::string str;             // string scalar
    std::vector<double> arr;     // numeric array (string arrays are skipped, only their length is kept)
    uint64_t arr_len = 0;
};

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    bool open(const std::string& path, std::string& err);
    /// Unmaps the file (the metadata stays readable; tensor data pointers become null).
    void close();

    const GgufTensor* tensor(const std::string& name) const;
    const GgufTensor& need(const std::string& name) const;
    bool has(const std::string& key) const { return kv_.count(key) != 0; }
    double num(const std::string& key, double def) const;
    int64_t i64(const std::string& key) const;
    std::string str(const std::string& key, const std::string& def = "") const;
    std::vector<double> arr(const std::string& key) const;

    const std::vector<GgufTensor>& tensors() const { return tensors_; }
    uint64_t file_size() const { return size_; }
    const uint8_t* base() const { return base_; }

private:
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    void* hfile_ = nullptr;
    void* hmap_ = nullptr;
    int fd_ = -1;
    std::map<std::string, GgufValue> kv_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> by_name_;
};

}  // namespace sq
