#pragma once

#include <cstdint>

#include "../../Span.h"
#include "ApiMtl.h"

namespace Ray {
class ILog;
namespace Mtl {
class AccStructure;
class Context;

class Buffer;
class DescrMultiPoolAlloc;
class Pipeline;
class ImageAtlas;
class Image;
class Sampler;

// must stay value-identical to Vk/Dx eBindTarget (shared kernel bodies bind against it)
enum class eBindTarget : uint16_t {
    Tex,
    TexSampled,
    TexArray,
    TexArraySampled,
    UBuf,
    SBufRO,
    SBufRW,
    Image,
    AccStruct,
    Sampler,
    DescrTable, // DX-only: bindless SRV/sampler tables; VK binds pre-built descriptor sets in-scene
    _Count
};

struct OpaqueHandle {
    union {
        const Image *tex;
        const Buffer *buf;
        const ImageAtlas *tex_arr;
        const Sampler *sampler;
        const AccStructure *acc_struct;
    };
    int count = 0;
    OpaqueHandle() = default;
    OpaqueHandle(const Image &_tex) : tex(&_tex), count(1) {}
    OpaqueHandle(const Buffer &_buf) : buf(&_buf), count(1) {}
    OpaqueHandle(const ImageAtlas &_tex_arr) : tex_arr(&_tex_arr), count(1) {}
    OpaqueHandle(const ImageAtlas *_tex_arr, int _count = 1) : tex_arr(_tex_arr), count(_count) {}
    OpaqueHandle(Span<const ImageAtlas> tex_arrs) : tex_arr(tex_arrs.data()), count(int(tex_arrs.size())) {}
    OpaqueHandle(const Sampler &_sampler) : sampler(&_sampler), count(1) {}
    OpaqueHandle(const AccStructure &_acc_struct) : acc_struct(&_acc_struct), count(1) {}
};

struct Binding {
    eBindTarget trg;
    uint16_t loc = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
    OpaqueHandle handle;

    Binding() = default;
    Binding(eBindTarget _trg, int _loc, OpaqueHandle _handle) : trg(_trg), loc(_loc), handle(_handle) {}
    Binding(eBindTarget _trg, int _loc, size_t _offset, OpaqueHandle _handle)
        : trg(_trg), loc(_loc), offset(uint32_t(_offset)), handle(_handle) {}
    Binding(eBindTarget _trg, int _loc, size_t _offset, size_t _size, OpaqueHandle _handle)
        : trg(_trg), loc(_loc), offset(uint32_t(_offset)), size(uint32_t(_size)), handle(_handle) {}
};

// Metal needs no descriptor sets: bindings are resolved straight onto the compute encoder
void DispatchCompute(CommandBuffer cmd_buf, const Pipeline &comp_pipeline, const uint32_t grp_count[3],
                     Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
                     DescrMultiPoolAlloc *descr_alloc, ILog *log);
void DispatchComputeIndirect(CommandBuffer cmd_buf, const Pipeline &comp_pipeline, const Buffer &indir_buf,
                             uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
                             int uniform_data_len, DescrMultiPoolAlloc *descr_alloc, ILog *log);
} // namespace Mtl
} // namespace Ray
