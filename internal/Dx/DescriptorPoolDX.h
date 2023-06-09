#pragma once

#include "../LinearAlloc.h"
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

    uint32_t Alloc(const uint32_t count) {
        if (next_free_ + count >= capacity_) {
            return 0xffffffff;
        }

        const uint32_t ret = next_free_;
        next_free_ += count;
        return ret;
    }
    void Free(uint32_t offset, uint32_t size) {}
    void Reset() { next_free_ = 0; }
};

class LinearAllocAdapted : public LinearAlloc {
  public:
    LinearAllocAdapted() = default;
    explicit LinearAllocAdapted(const uint32_t capacity) : LinearAlloc(1, capacity) {}
    /*LinearAllocAdapted(LinearAllocAdapted &&rhs) noexcept = default;
    ~LinearAllocAdapted() {
        for (uint32_t i = 0; i < size(); ++i) {
            assert(!IsSet(i));
        }
    }
    LinearAllocAdapted &operator=(LinearAllocAdapted &rhs) noexcept = default;*/

    uint32_t capacity() const { return size(); }

    uint32_t Alloc(const uint32_t count) { return LinearAlloc::Alloc(count, nullptr); }
    void Reset() { LinearAlloc::Clear(); }
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

    uint32_t Alloc(const uint32_t descr_count) { return alloc_.Alloc(descr_count); }
    void Free(const uint32_t offset, const uint32_t size) { alloc_.Free(offset, size); }
    void Reset();
};

extern template class DescrPool<BumpAlloc>;
extern template class DescrPool<LinearAllocAdapted>;

struct PoolRef {
    ID3D12DescriptorHeap *heap = nullptr;
    uint32_t offset = 0xffffffff;
    uint32_t count = 0;

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
extern template class DescrPoolAlloc<LinearAllocAdapted>;

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
extern template class DescrMultiPoolAlloc<LinearAllocAdapted>;

} // namespace Dx
} // namespace Ray