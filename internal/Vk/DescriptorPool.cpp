#include "DescriptorPool.h"

#include "Context.h"

namespace Ray {
namespace Vk {
const VkDescriptorType g_descr_types_vk[] = {
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR};
static_assert(COUNT_OF(g_descr_types_vk) == int(eDescrType::_Count), "!");
} // namespace Vk
} // namespace Ray

Ray::Vk::DescrPool &Ray::Vk::DescrPool::operator=(DescrPool &&rhs) noexcept {
    if (this == &rhs) {
        return (*this);
    }

    Destroy();

    ctx_ = exchange(rhs.ctx_, nullptr);
    handle_ = exchange(rhs.handle_, {});
    sets_count_ = exchange(rhs.sets_count_, 0);
    next_free_ = exchange(rhs.next_free_, 0);
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        descr_counts_[i] = rhs.descr_counts_[i];
    }

    return (*this);
}

bool Ray::Vk::DescrPool::Init(const DescrSizes &sizes, const uint32_t sets_count) {
    Destroy();

    descr_counts_[int(eDescrType::CombinedImageSampler)] = sizes.img_sampler_count;
    descr_counts_[int(eDescrType::StorageImage)] = sizes.store_img_count;
    descr_counts_[int(eDescrType::UniformBuffer)] = sizes.ubuf_count;
    descr_counts_[int(eDescrType::StorageBuffer)] = sizes.sbuf_count;
    descr_counts_[int(eDescrType::UniformTexBuffer)] = sizes.tbuf_count;
    descr_counts_[int(eDescrType::AccStructure)] = sizes.acc_count;

    SmallVector<VkDescriptorPoolSize, int(eDescrType::_Count)> pool_sizes;
    for (int i = 0; i < int(eDescrType::_Count); ++i) {
        if (descr_counts_[i]) {
            pool_sizes.push_back({g_descr_types_vk[i], sets_count * descr_counts_[i]});
        }
    }

    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.poolSizeCount = uint32_t(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.cdata();
    pool_info.maxSets = sets_count;

    sets_count_ = sets_count;

    const VkResult res = vkCreateDescriptorPool(ctx_->device(), &pool_info, nullptr, &handle_);
    return res == VK_SUCCESS;
}

void Ray::Vk::DescrPool::Destroy() {
    if (handle_) {
        ctx_->descriptor_pools_to_destroy[ctx_->backend_frame].emplace_back(handle_);
        handle_ = {};
    }
}

VkDescriptorSet Ray::Vk::DescrPool::Alloc(const VkDescriptorSetLayout layout) {
    if (next_free_ >= sets_count_) {
        return VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.descriptorPool = handle_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet descr_set = VK_NULL_HANDLE;
    const VkResult res = vkAllocateDescriptorSets(ctx_->device(), &alloc_info, &descr_set);
    assert(res == VK_SUCCESS);

    ++next_free_;

    return descr_set;
}

bool Ray::Vk::DescrPool::Reset() {
    next_free_ = 0;
    const VkResult res = vkResetDescriptorPool(ctx_->device(), handle_, 0);
    return res == VK_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

VkDescriptorSet Ray::Vk::DescrPoolAlloc::Alloc(const VkDescriptorSetLayout layout) {
    if (next_free_pool_ == -1 || pools_[next_free_pool_].free_count() == 0) {
        ++next_free_pool_;

        if (next_free_pool_ == pools_.size()) {
            // allocate twice more sets each time
            const uint32_t count_mul = (1u << pools_.size());

            DescrPool &new_pool = pools_.emplace_back(ctx_);
            if (!new_pool.Init(sizes_, count_mul * initial_sets_count_)) {
                return VK_NULL_HANDLE;
            }
        }
    }
    return pools_[next_free_pool_].Alloc(layout);
}

bool Ray::Vk::DescrPoolAlloc::Reset() {
    if (pools_.empty()) {
        return true;
    }

    bool result = true;
    for (auto &pool : pools_) {
        result &= pool.Reset();
    }
    next_free_pool_ = 0;
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Ray::Vk::DescrMultiPoolAlloc::DescrMultiPoolAlloc(Context *ctx, const uint32_t pool_step,
                                                  const uint32_t max_img_sampler_count,
                                                  const uint32_t max_store_img_count, const uint32_t max_ubuf_count,
                                                  const uint32_t max_sbuf_count, const uint32_t max_tbuf_count,
                                                  const uint32_t max_acc_count, const uint32_t initial_sets_count)
    : pool_step_(pool_step) {
    img_sampler_based_count_ = (max_img_sampler_count + pool_step - 1) / pool_step;
    store_img_based_count_ = (max_store_img_count + pool_step - 1) / pool_step;
    ubuf_based_count_ = (max_ubuf_count + pool_step - 1) / pool_step;
    sbuf_based_count_ = (max_sbuf_count + pool_step - 1) / pool_step;
    tbuf_based_count_ = (max_tbuf_count + pool_step - 1) / pool_step;
    acc_based_count_ = (max_acc_count + pool_step - 1) / pool_step;
    const uint32_t required_pools_count = img_sampler_based_count_ * store_img_based_count_ * ubuf_based_count_ *
                                          sbuf_based_count_ * tbuf_based_count_ * acc_based_count_;

    // store rounded values
    max_img_sampler_count_ = pool_step * img_sampler_based_count_;
    max_store_img_count_ = pool_step * store_img_based_count_;
    max_ubuf_count_ = pool_step * ubuf_based_count_;
    max_sbuf_count_ = pool_step * sbuf_based_count_;
    max_tbuf_count_ = pool_step * tbuf_based_count_;
    max_acc_count_ = pool_step * acc_based_count_;

    for (uint32_t i = 0; i < required_pools_count; ++i) {
        uint32_t index = i;

        DescrSizes pool_sizes;
        pool_sizes.acc_count = pool_step * ((index % acc_based_count_) + 1);
        index /= acc_based_count_;
        pool_sizes.tbuf_count = pool_step * ((index % tbuf_based_count_) + 1);
        index /= tbuf_based_count_;
        pool_sizes.sbuf_count = pool_step * ((index % sbuf_based_count_) + 1);
        index /= sbuf_based_count_;
        pool_sizes.ubuf_count = pool_step * ((index % ubuf_based_count_) + 1);
        index /= ubuf_based_count_;
        pool_sizes.img_sampler_count = pool_step * ((index % img_sampler_based_count_) + 1);
        index /= img_sampler_based_count_;
        pool_sizes.store_img_count = pool_step * ((index % store_img_based_count_) + 1);
        index /= store_img_based_count_;

        pools_.emplace_back(ctx, pool_sizes, initial_sets_count);
    }
    assert(pools_.size() == required_pools_count);
}

VkDescriptorSet Ray::Vk::DescrMultiPoolAlloc::Alloc(const DescrSizes &sizes, const VkDescriptorSetLayout layout) {
    const uint32_t img_sampler_based_index =
        sizes.img_sampler_count ? ((sizes.img_sampler_count + pool_step_ - 1) / pool_step_ - 1) : 0;
    const uint32_t store_img_based_index =
        sizes.store_img_count ? ((sizes.store_img_count + pool_step_ - 1) / pool_step_ - 1) : 0;
    const uint32_t ubuf_based_index = sizes.ubuf_count ? ((sizes.ubuf_count + pool_step_ - 1) / pool_step_ - 1) : 0;
    const uint32_t sbuf_based_index = sizes.sbuf_count ? ((sizes.sbuf_count + pool_step_ - 1) / pool_step_ - 1) : 0;
    const uint32_t tbuf_based_index = sizes.tbuf_count ? ((sizes.tbuf_count + pool_step_ - 1) / pool_step_ - 1) : 0;
    const uint32_t acc_based_index = sizes.acc_count ? ((sizes.acc_count + pool_step_ - 1) / pool_step_ - 1) : 0;

    uint32_t pool_index = img_sampler_based_index * store_img_based_count_ * ubuf_based_count_ * sbuf_based_count_ *
                          tbuf_based_count_ * acc_based_count_;
    pool_index += store_img_based_index * ubuf_based_count_ * sbuf_based_count_ * tbuf_based_count_ * acc_based_count_;
    pool_index += ubuf_based_index * sbuf_based_count_ * tbuf_based_count_ * acc_based_count_;
    pool_index += sbuf_based_index * tbuf_based_count_ * acc_based_count_;
    pool_index += tbuf_based_index * acc_based_count_;
    pool_index += acc_based_index;
    return pools_[pool_index].Alloc(layout);
}

bool Ray::Vk::DescrMultiPoolAlloc::Reset() {
    bool result = true;
    for (auto &pool : pools_) {
        result &= pool.Reset();
    }
    return result;
}