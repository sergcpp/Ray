#pragma once

#include "../SmallVector.h"

struct ID3D12DescriptorHeap;

namespace Ray {
namespace Dx {
class Context;

enum class eDescrType : uint8_t {
    CBV_SRV_UAV,
    Sampler,
    RTV,
    DSV,
    _Count
};

struct DescrTable {
    eDescrType type;
    ID3D12DescriptorHeap *heap = nullptr;
    uint64_t ptr = 0;
    int count = 0;
};

class DescrPool {
    Context *ctx_ = nullptr;
    eDescrType type_;
    ID3D12DescriptorHeap *heap_ = nullptr;
    uint32_t descr_count_ = 0, next_free_ = 0;

  public:
    DescrPool(Context *ctx, const eDescrType type) : ctx_(ctx), type_(type) {}
    DescrPool(const DescrPool &rhs) = delete;
    DescrPool(DescrPool &&rhs) noexcept { (*this) = std::move(rhs); }
    ~DescrPool() { Destroy(); }

    DescrPool &operator=(const DescrPool &rhs) = delete;
    DescrPool &operator=(DescrPool &&rhs) noexcept;

    Context *ctx() { return ctx_; }
    ID3D12DescriptorHeap *heap() { return heap_; }

    uint32_t free_count() const { return descr_count_ - next_free_; }
    uint32_t descr_count() const { return descr_count_; }

    bool Init(uint32_t descr_count, bool shader_visible = true);
    void Destroy();

    uint32_t Alloc(uint32_t descr_count);
    void Reset();
};

struct PoolRef {
    ID3D12DescriptorHeap *heap;
    uint32_t offset;
    uint32_t _pad;
};

class DescrPoolAlloc {
    Context *ctx_ = nullptr;
    eDescrType type_;
    uint32_t initial_descr_count_ = 0;

    SmallVector<DescrPool, 256> pools_;
    int next_free_pool_ = -1;

  public:
    DescrPoolAlloc(Context *ctx, const eDescrType type, const uint32_t initial_descr_count)
        : ctx_(ctx), type_(type), initial_descr_count_(initial_descr_count) {}

    Context *ctx() { return ctx_; }

    PoolRef Alloc(uint32_t descr_count);
    void Reset();
};

struct DescrSizes {
    union {
        struct {
            uint32_t cbv_srv_uav_count;
            uint32_t sampler_count;
            uint32_t rtv_count;
            uint32_t dsv_count;
        };
        uint32_t counts[4] = {};
    };
};
static_assert(offsetof(DescrSizes, cbv_srv_uav_count) == sizeof(uint32_t) * int(eDescrType::CBV_SRV_UAV), "!");
static_assert(offsetof(DescrSizes, sampler_count) == sizeof(uint32_t) * int(eDescrType::Sampler), "!");
static_assert(offsetof(DescrSizes, rtv_count) == sizeof(uint32_t) * int(eDescrType::RTV), "!");
static_assert(offsetof(DescrSizes, dsv_count) == sizeof(uint32_t) * int(eDescrType::DSV), "!");

struct PoolRefs {
    union {
        struct {
            PoolRef cbv_srv_uav;
            PoolRef sampler;
            PoolRef rtv;
            PoolRef dsv;
        };
        PoolRef refs[4] = {};
    };
};

class DescrMultiPoolAlloc {
    SmallVector<DescrPoolAlloc, int(eDescrType::_Count)> pools_;

  public:
    DescrMultiPoolAlloc(Context *ctx, uint32_t initial_descr_count);

    Context *ctx() { return pools_.front().ctx(); }

    PoolRefs Alloc(const DescrSizes &sizes);
    void Reset();
};
} // namespace Vk
} // namespace Ray