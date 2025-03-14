#pragma once

#include <cstdint>

#include "../../Span.h"

struct ID3D12GraphicsCommandList;

namespace Ray {
class ILog;
namespace Dx {
class AccStructure;
class Context;

class Buffer;
class BumpAlloc;
template <class Allocator> class DescrMultiPoolAlloc;
class Pipeline;
class Program;
class TextureAtlas;
class Texture;
struct DescrTable;
class Sampler;

enum class eBindTarget : uint16_t {
    Tex2D,
    Tex2DSampled,
    Tex2DMs,
    Tex2DArray,
    Tex2DArraySampled,
    TexCubeArray,
    Tex3D,
    Tex3DSampled,
    TBuf,
    UBuf,
    SBufRO,
    SBufRW,
    Image,
    AccStruct,
    Sampler,
    DescrTable, // TODO: This does not belong here!
    _Count
};

struct OpaqueHandle {
    union {
        const Texture *tex;
        const Buffer *buf;
        const TextureAtlas *tex_arr;
        const AccStructure *acc_struct;
        const Sampler *sampler;
        const DescrTable *descr_table;
    };
    int count = 0;
    OpaqueHandle() = default;
    OpaqueHandle(const Texture &_tex) : tex(&_tex), count(1) {}
    OpaqueHandle(const Buffer &_buf) : buf(&_buf), count(1) {}
    OpaqueHandle(const TextureAtlas &_tex_arr) : tex_arr(&_tex_arr), count(1) {}
    OpaqueHandle(const TextureAtlas *_tex_arr, int _count = 1) : tex_arr(_tex_arr), count(_count) {}
    OpaqueHandle(Span<const TextureAtlas> tex_arrs) : tex_arr(tex_arrs.data()), count(int(tex_arrs.size())) {}
    OpaqueHandle(const Sampler &_sampler) : sampler(&_sampler), count(1) {}
    OpaqueHandle(const AccStructure &_acc_struct) : acc_struct(&_acc_struct), count(1) {}
    OpaqueHandle(const DescrTable &_descr_table) : descr_table(&_descr_table), count(1) {}
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
// static_assert(sizeof(Binding) == sizeof(void *) + 8 + 8, "!");

void PrepareDescriptors(Context *ctx, ID3D12GraphicsCommandList *cmd_buf, Span<const Binding> bindings,
                        const void *uniform_data, int uniform_data_len, const Program *prog,
                        DescrMultiPoolAlloc<BumpAlloc> *descr_alloc, ILog *log);

void DispatchCompute(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &comp_pipeline, const uint32_t grp_count[3],
                     Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
                     DescrMultiPoolAlloc<BumpAlloc> *descr_alloc, ILog *log);
void DispatchComputeIndirect(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &comp_pipeline, const Buffer &indir_buf,
                             uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
                             int uniform_data_len, DescrMultiPoolAlloc<BumpAlloc> *descr_alloc, ILog *log);

// void TraceRays(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &rt_pipeline, const uint32_t dims[3],
//                Span<const Binding> bindings, const void *uniform_data, int uniform_data_len,
//                DescrMultiPoolAlloc *descr_alloc, ILog *log);
// void TraceRaysIndirect(ID3D12GraphicsCommandList *cmd_buf, const Pipeline &rt_pipeline, const Buffer &indir_buf,
//                        uint32_t indir_buf_offset, Span<const Binding> bindings, const void *uniform_data,
//                        int uniform_data_len, DescrMultiPoolAlloc *descr_alloc, ILog *log);
} // namespace Dx
} // namespace Ray