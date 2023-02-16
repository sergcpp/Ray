#pragma once

#include <cstdint>

#include "../Span.h"
#include "VK.h"

#if (VK_USE_64_BIT_PTR_DEFINES == 1)
struct VkDescriptorSet_T;
typedef VkDescriptorSet_T *VkDescriptorSet;
struct VkDescriptorSetLayout_T;
typedef VkDescriptorSetLayout_T *VkDescriptorSetLayout;
#else
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkDescriptorSetLayout;
#endif

namespace Ray {
class ILog;
namespace Vk {
class AccStructure;
class Context;

class Buffer;
class DescrMultiPoolAlloc;
class Pipeline;
class TextureAtlas;
class Texture1D;
class Texture2D;

enum class eBindTarget : uint16_t {
    Tex2D,
    Tex2DMs,
    Tex2DArray,
    TexCubeArray,
    TBuf,
    UBuf,
    SBuf,
    Image,
    AccStruct,
    _Count
};

struct OpaqueHandle {
    union {
        const Texture2D *tex;
        const Buffer *buf;
        const Texture1D *tex_buf;
        const TextureAtlas *tex_arr;
        const AccStructure *acc_struct;
    };
    int count = 0;
    OpaqueHandle() = default;
    OpaqueHandle(const Texture2D &_tex) : tex(&_tex), count(1) {}
    OpaqueHandle(const Buffer &_buf) : buf(&_buf), count(1) {}
    OpaqueHandle(const Texture1D &_tex) : tex_buf(&_tex), count(1) {}
    OpaqueHandle(const TextureAtlas &_tex_arr) : tex_arr(&_tex_arr), count(1) {}
    OpaqueHandle(const TextureAtlas *_tex_arr, int _count = 1) : tex_arr(_tex_arr), count(_count) {}
    OpaqueHandle(Span<const TextureAtlas> tex_arrs) : tex_arr(tex_arrs.data()), count(int(tex_arrs.size())) {}
    OpaqueHandle(const AccStructure &_acc_struct) : acc_struct(&_acc_struct), count(1) {}
};

struct Binding {
    eBindTarget trg;
    uint16_t loc = 0;
    uint16_t offset = 0;
    uint16_t size = 0;
    OpaqueHandle handle;

    Binding() = default;
    Binding(eBindTarget _trg, int _loc, OpaqueHandle _handle) : trg(_trg), loc(_loc), handle(_handle) {}
    Binding(eBindTarget _trg, int _loc, size_t _offset, OpaqueHandle _handle)
        : trg(_trg), loc(_loc), offset(uint16_t(_offset)), handle(_handle) {}
    Binding(eBindTarget _trg, uint16_t _loc, size_t _offset, size_t _size, OpaqueHandle _handle)
        : trg(_trg), loc(_loc), offset(uint16_t(_offset)), size(uint16_t(_size)), handle(_handle) {}
};
// static_assert(sizeof(Binding) == sizeof(void *) + 8 + 8, "!");

VkDescriptorSet PrepareDescriptorSet(Context *ctx, VkDescriptorSetLayout layout, Span<const Binding> bindings,
                                     DescrMultiPoolAlloc *descr_alloc, ILog *log);

void DispatchCompute(VkCommandBuffer cmd_buf, const Pipeline &comp_pipeline, const uint32_t grp_count[3],
                     Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
                     DescrMultiPoolAlloc *descr_alloc, ILog *log);
void DispatchComputeIndirect(VkCommandBuffer cmd_buf, const Pipeline &comp_pipeline, const Buffer &indir_buf,
                             uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
                             int uniform_data_len, DescrMultiPoolAlloc *descr_alloc, ILog *log);
} // namespace Vk
} // namespace Ray