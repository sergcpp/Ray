#pragma once

#include "../SmallVector.h"
#include "Api.h"

namespace Ray::Vk {
class Context;

enum class eDescrType : uint8_t {
    CombinedImageSampler,
    SampledImage,
    Sampler,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    UniformTexBuffer,
    AccStructure,
    _Count
};

struct DescrSizes {
    uint32_t img_sampler_count = 0;
    uint32_t img_count = 0;
    uint32_t sampler_count = 0;
    uint32_t store_img_count = 0;
    uint32_t ubuf_count = 0;
    uint32_t sbuf_count = 0;
    uint32_t tbuf_count = 0;
    uint32_t acc_count = 0;
};

//
// DescrPool is able to allocate up to fixed amount of sets of specific size
//
class DescrPool {
    Context *ctx_ = nullptr;
    VkDescriptorPool handle_ = VK_NULL_HANDLE;
    uint32_t sets_count_ = 0, next_free_ = 0;

    uint32_t descr_counts_[int(eDescrType::_Count)] = {};

  public:
    explicit DescrPool(Context *ctx) : ctx_(ctx) {}
    DescrPool(const DescrPool &rhs) = delete;
    DescrPool(DescrPool &&rhs) noexcept { (*this) = std::move(rhs); }
    ~DescrPool() { Destroy(); }

    DescrPool &operator=(const DescrPool &rhs) = delete;
    DescrPool &operator=(DescrPool &&rhs) noexcept;

    Context *ctx() { return ctx_; }

    uint32_t free_count() const { return sets_count_ - next_free_; }
    uint32_t descr_count(const eDescrType type) const { return descr_counts_[int(type)]; }

    bool Init(const DescrSizes &sizes, uint32_t sets_count);
    void Destroy();

    VkDescriptorSet Alloc(VkDescriptorSetLayout layout);
    bool Reset();
};

//
// DescrPoolAlloc is able to allocate any amount of sets of specific size
//
class DescrPoolAlloc {
    Context *ctx_ = nullptr;
    DescrSizes sizes_;
    uint32_t initial_sets_count_ = 0;

    SmallVector<DescrPool, 256> pools_;
    int next_free_pool_ = -1;

  public:
    DescrPoolAlloc(Context *ctx, const DescrSizes &sizes, const uint32_t initial_sets_count)
        : ctx_(ctx), sizes_(sizes), initial_sets_count_(initial_sets_count) {}

    Context *ctx() { return ctx_; }

    VkDescriptorSet Alloc(VkDescriptorSetLayout layout);
    bool Reset();
};

//
// DescrMultiPoolAlloc is able to allocate any amount of sets of any size
//
class DescrMultiPoolAlloc {
    uint32_t pool_step_ = 0;
    uint32_t img_sampler_based_count_ = 0, img_based_count_ = 0, sampler_based_count_ = 0, store_img_based_count_ = 0,
             ubuf_based_count_ = 0, sbuf_based_count_ = 0, tbuf_based_count_ = 0, acc_based_count_;
    uint32_t max_img_sampler_count_ = 0, max_img_count_ = 0, max_sampler_count_ = 0, max_store_img_count_ = 0,
             max_ubuf_count_ = 0, max_sbuf_count_ = 0, max_tbuf_count_ = 0, max_acc_count_ = 0;
    SmallVector<DescrPoolAlloc, 16> pools_;

  public:
    DescrMultiPoolAlloc(Context *ctx, uint32_t pool_step, uint32_t max_img_sampler_count, uint32_t max_img_count,
                        uint32_t max_sampler_count, uint32_t max_store_img_count, uint32_t max_ubuf_count,
                        uint32_t max_sbuf_count, uint32_t max_tbuf_count, uint32_t max_acc_count,
                        uint32_t initial_sets_count);

    Context *ctx() { return pools_.front().ctx(); }

    VkDescriptorSet Alloc(const DescrSizes &sizes, VkDescriptorSetLayout layout);
    bool Reset();
};
} // namespace Ray::Vk