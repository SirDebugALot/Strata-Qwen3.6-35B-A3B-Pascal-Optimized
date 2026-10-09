#pragma once
#include "cpu_iq3_nibble.hpp"
#include <memory>
#include <string>
#include <vector>
namespace sq {
struct ExpertLayerDesc;

// Immutable pageable storage owned by CpuMoe. Build before worker creation; release after worker join.
class CpuIq3DownCache {
public:
    static constexpr size_t BlocksPerExpert = 2048 * 2;
    bool build(const std::vector<ExpertLayerDesc>& layers, int workers, std::string& error);
    const native_iq::Iq3NibbleBlock* layer(int index) const noexcept { return layers_[index].data.get(); }
    size_t bytes() const noexcept { return bytes_; }
    static bool estimate_bytes(const std::vector<ExpertLayerDesc>& layers, size_t& bytes, std::string& error);
private:
    struct Layer { std::unique_ptr<native_iq::Iq3NibbleBlock[]> data; };
    std::vector<Layer> layers_;
    size_t bytes_ = 0;
};
}
