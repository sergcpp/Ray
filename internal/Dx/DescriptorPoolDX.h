#pragma once

#include "../FreelistAlloc.h"
#include "../SmallVector.h"

struct ID3D12DescriptorHeap;

namespace Ray {
namespace Dx {
class Context;

enum class eDescrType : uint8_t { CBV_SRV_UAV, Sampler, RTV, DSV, _Count };

struct DescrTable {
    eDescrType type;
    ID3D12DescriptorHeap *cpu_heap = nullptr;
    mutable ID3D12DescriptorHeap *gpu_heap = nullptr;
    uint64_t cpu_ptr = 0;
    mutable uint64_t gpu_ptr = 0;
    int count = 0;
};

class BumpAlloc {
    uint32_t capacity_ = 0, next_free_ = 0;

  public:
    BumpAlloc() = default;
    explicit BumpAlloc(const uint32_t capacity) : capacity_(capacity) {}

    uint32_t capacity() const { return capacity_; }

    std::pair<uint32_t, uint32_t> Alloc(const uint32_t count) {
        if (next_free_ + count >= capacity_) {
            return std::make_pair(0xffffffff, 0xffffffff);
        }

        const uint32_t ret = next_free_;
        next_free_ += count;
        return std::make_pair(ret, count);
    }
    void Free(const uint32_t offset, const uint32_t size) {
        if (offset + size == next_free_) {
            next_free_ -= size;
        }
    }
    void Reset() { next_free_ = 0; }
};

class FreelistAllocAdapted : public FreelistAlloc {
    uint32_t capacity_ = 0;

  public:
    FreelistAllocAdapted() = default;
    explicit FreelistAllocAdapted(const uint32_t capacity) : FreelistAlloc(capacity), capacity_(capacity) {}

    uint32_t capacity() const { return capacity_; }

    std::pair<uint32_t, uint32_t> Alloc(const uint32_t count) {
        const auto alloc = FreelistAlloc::Alloc(count);
        assert(FreelistAlloc::IntegrityCheck());
        return std::make_pair(alloc.offset, alloc.block);
    }
    void Free(const uint32_t offset, const uint32_t block) {
        FreelistAlloc::Free(block);
        assert(FreelistAlloc::IntegrityCheck());
    }
    void Reset() {}
};

template <class Allocator> class DescrPool {
    Allocator alloc_;
    Context *ctx_ = nullptr;
    eDescrType type_;
    ID3D12DescriptorHeap *heap_ = nullptr;

  public:
    DescrPool(Context *ctx, const eDescrType type) : ctx_(ctx), type_(type) {}
    DescrPool(const DescrPool &rhs) = delete;
    DescrPool(DescrPool &&rhs) noexcept { (*this) = std::move(rhs); }
    ~DescrPool() { Destroy(); }

    DescrPool &operator=(const DescrPool &rhs) = delete;
    DescrPool &operator=(DescrPool &&rhs) noexcept;

    Context *ctx() { return ctx_; }
    ID3D12DescriptorHeap *heap() { return heap_; }

    uint32_t capacity() const { return alloc_.capacity(); }

    bool Init(uint32_t descr_count, bool shader_visible);
    void Destroy();

    std::pair<uint32_t, uint32_t> Alloc(const uint32_t descr_count) { return alloc_.Alloc(descr_count); }
    void Free(const uint32_t offset, const uint32_t block) { alloc_.Free(offset, block); }
    void Reset();
};

extern template class DescrPool<BumpAlloc>;
extern template class DescrPool<FreelistAllocAdapted>;

struct PoolRef {
    ID3D12DescriptorHeap *heap = nullptr;
    uint32_t offset = 0xffffffff, block = 0xffffffff;

    operator bool() const { return heap != nullptr; }
};

template <class Allocator> class DescrPoolAlloc {
    Context *ctx_ = nullptr;
    eDescrType type_;
    bool shader_visible_ = false;
    uint32_t initial_descr_count_ = 0;

    SmallVector<DescrPool<Allocator>, 256> pools_;

  public:
    DescrPoolAlloc(Context *ctx, const eDescrType type, const bool shader_visible, const uint32_t initial_descr_count)
        : ctx_(ctx), type_(type), shader_visible_(shader_visible), initial_descr_count_(initial_descr_count) {}

    Context *ctx() { return ctx_; }

    PoolRef Alloc(uint32_t descr_count);
    void Free(const PoolRef &ref);
    void Reset();
};

extern template class DescrPoolAlloc<BumpAlloc>;
extern template class DescrPoolAlloc<FreelistAllocAdapted>;

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

template <class Allocator> class DescrMultiPoolAlloc {
    SmallVector<DescrPoolAlloc<Allocator>, int(eDescrType::_Count)> pools_;

  public:
    DescrMultiPoolAlloc(Context *ctx, bool shader_visible, uint32_t initial_descr_count);

    Context *ctx() { return pools_.front().ctx(); }

    PoolRef Alloc(eDescrType type, uint32_t descr_count);
    PoolRefs Alloc(const DescrSizes &sizes);
    void Free(eDescrType type, const PoolRef &ref);
    void Free(const PoolRefs &refs);
    void Reset();
};

extern template class DescrMultiPoolAlloc<BumpAlloc>;
extern template class DescrMultiPoolAlloc<FreelistAllocAdapted>;

} // namespace Dx
} // namespace Ray