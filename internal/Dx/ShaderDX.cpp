#include "ShaderDX.h"

#include <stdexcept>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// #include <d3d12shader.h>
#include <d3dcommon.h>
// #include <dxcapi.h>

#include "../../Log.h"
#include "../ScopeExit.h"
#include "ContextDX.h"

namespace Ray {
namespace Dx {
// extern const VkShaderStageFlagBits g_shader_stages_vk[] = {
//     VK_SHADER_STAGE_VERTEX_BIT,                  // Vert
//     VK_SHADER_STAGE_FRAGMENT_BIT,                // Frag
//     VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,    // Tesc
//     VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, // Tese
//     VK_SHADER_STAGE_COMPUTE_BIT,                 // Comp
//     VK_SHADER_STAGE_RAYGEN_BIT_KHR,              // RayGen
//     VK_SHADER_STAGE_MISS_BIT_KHR,                // Miss
//     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,         // ClosestHit
//     VK_SHADER_STAGE_ANY_HIT_BIT_KHR,             // AnyHit
//     VK_SHADER_STAGE_INTERSECTION_BIT_KHR         // Intersection
// };
// static_assert(std::size(g_shader_stages_vk) == int(eShaderType::_Count), "!");

// TODO: not rely on this order somehow
// static_assert(int(eShaderType::RayGen) < int(eShaderType::Miss), "!");
// static_assert(int(eShaderType::Miss) < int(eShaderType::ClosestHit), "!");
// static_assert(int(eShaderType::ClosestHit) < int(eShaderType::AnyHit), "!");
// static_assert(int(eShaderType::AnyHit) < int(eShaderType::Intersection), "!");

#define _DXC_FOURCC(ch0, ch1, ch2, ch3) (uint32_t(ch0) | uint32_t(ch1) << 8 | uint32_t(ch2) << 16 | uint32_t(ch3) << 24)

// const uint32_t _DXC_PART_PDB = _DXC_FOURCC('I', 'L', 'D', 'B');
// const uint32_t _DXC_PART_PDB_NAME = _DXC_FOURCC('I', 'L', 'D', 'N');
// const uint32_t _DXC_PART_PRIVATE_DATA = _DXC_FOURCC('P', 'R', 'I', 'V');
// const uint32_t _DXC_PART_ROOT_SIGNATURE = _DXC_FOURCC('R', 'T', 'S', '0');
// const uint32_t _DXC_PART_DXIL = _DXC_FOURCC('D', 'X', 'I', 'L');
const uint32_t _DXC_PART_REFLECTION_DATA = _DXC_FOURCC('S', 'T', 'A', 'T');
// const uint32_t _DXC_PART_SHADER_HASH = _DXC_FOURCC('H', 'A', 'S', 'H');

#undef _DXC_FOURCC

/*struct ShaderBlobAdapter : public IDxcBlob {
    LPVOID pointer_;
    SIZE_T size_;

    ShaderBlobAdapter(const uint8_t *pointer, const int size) : pointer_((void *)pointer), size_(size) {}

    LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return pointer_; }
    SIZE_T STDMETHODCALLTYPE GetBufferSize() override { return size_; }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, _COM_Outptr_ void __RPC_FAR *__RPC_FAR *ppvObject) override {
        if (ppvObject == nullptr) {
            return E_POINTER;
        }

        if (IsEqualIID(riid, __uuidof(IUnknown)) || IsEqualIID(riid, __uuidof(INoMarshal))) {
            *ppvObject = static_cast<IUnknown *>(this);
            return NOERROR;
        } else if (IsEqualIID(riid, __uuidof(IDxcBlob))) {
            *ppvObject = static_cast<IDxcBlob *>(this);
            return NOERROR;
        }

        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 0; }
};*/
} // namespace Dx
} // namespace Ray

Ray::Dx::Shader::Shader(std::string_view name, Context *ctx, Span<const uint8_t> shader_code, const eShaderType type,
                        ILog *log) {
    if (!Init(name, ctx, shader_code, type, log)) {
        throw std::runtime_error("Shader Init error!");
    }
}

Ray::Dx::Shader::~Shader() {}

Ray::Dx::Shader &Ray::Dx::Shader::operator=(Shader &&rhs) noexcept {
    device_ = std::exchange(rhs.device_, {});
    shader_code_ = std::move(rhs.shader_code_);
    type_ = rhs.type_;
    name_ = std::move(rhs.name_);

    attr_bindings = std::move(rhs.attr_bindings);
    unif_bindings = std::move(rhs.unif_bindings);
    pc_ranges = std::move(rhs.pc_ranges);

    return (*this);
}

bool Ray::Dx::Shader::Init(std::string_view name, Context *ctx, Span<const uint8_t> shader_code, const eShaderType type,
                           ILog *log) {
    name_ = name;
    device_ = ctx->device();

    if (!InitFromCSO(shader_code, type, log)) {
        return false;
    }

    return true;
}

bool Ray::Dx::Shader::InitFromCSO(Span<const uint8_t> shader_code, const eShaderType type, ILog *log) {
    if (shader_code.empty()) {
        return false;
    }

    type_ = type;

    attr_bindings.clear();
    unif_bindings.clear();
    pc_ranges.clear();

    struct header_t {
        char sig[4];
        uint8_t checksum[16];
        uint32_t one;
        uint32_t total_size;
        uint32_t chunk_count;
    };

    uint32_t data_off = 0;

    header_t header = {};
    memcpy(&header, shader_code.data(), sizeof(header_t));
    data_off += sizeof(header_t);

    shader_code_.assign(shader_code.data(), shader_code.data() + header.total_size);

#if 0 // Unfinished manual parsing
    assert(header.chunk_count < 16);
    uint32_t chunk_offsets[16] = {};
    memcpy(chunk_offsets, shader_code + data_off, header.chunk_count * sizeof(uint32_t));
    data_off += header.chunk_count * sizeof(uint32_t);

    uint32_t stat_chunk_index = 0xffffffff;
    for (uint32_t i = 0; i < header.chunk_count; ++i) {
        uint32_t chunk_type = 0;
        memcpy(&chunk_type, shader_code + chunk_offsets[i], sizeof(uint32_t));

        if (chunk_type == _DXC_PART_REFLECTION_DATA) {
            stat_chunk_index = i;
            break;
        }
    }

    if (stat_chunk_index == 0xffffffff) {
        log->Error("Failed to find reflection data");
        return false;
    }

    struct stat_chunk_t {
        uint32_t chunk_type;
        uint32_t chunk_size;
        uint32_t version;
    };
#endif

#if 1 // read custom reflection data
    data_off = header.total_size;

    uint32_t pc_count = 0;
    memcpy(&pc_count, shader_code.data() + data_off, sizeof(uint32_t));
    data_off += sizeof(uint32_t);

    if (pc_count) {
        pc_ranges.resize(pc_count);
        memcpy(pc_ranges.data(), shader_code.data() + data_off, pc_count * sizeof(Range));
        data_off += pc_count * sizeof(Range);
    }

    uint32_t ub_count = 0;
    memcpy(&ub_count, shader_code.data() + data_off, sizeof(uint32_t));
    data_off += sizeof(uint32_t);

    if (ub_count) {
        unif_bindings.resize(ub_count);
        memcpy(unif_bindings.data(), shader_code.data() + data_off, ub_count * sizeof(Descr));
        data_off += ub_count * sizeof(Descr);
    }

    (void)_DXC_PART_REFLECTION_DATA;
#else
    IDxcContainerReflection *container_reflection = {};
    HRESULT hr = DxcCreateInstance(CLSID_DxcContainerReflection, IID_PPV_ARGS(&container_reflection));
    if (FAILED(hr)) {
        log->Error("Failed to create DxcContainerReflection");
        return false;
    }
    SCOPE_EXIT(container_reflection->Release());

    ShaderBlobAdapter shader_blob(shader_code, code_size);

    hr = container_reflection->Load(&shader_blob);
    if (FAILED(hr)) {
        log->Error("Failed to load shader blob");
        return false;
    }

    uint32_t reflection_part = 0;
    hr = container_reflection->FindFirstPartKind(_DXC_PART_REFLECTION_DATA, &reflection_part);
    if (FAILED(hr)) {
        log->Error("Failed to find reflection data");
        return false;
    }

    ID3D12ShaderReflection *shader_reflection = {};
    hr = container_reflection->GetPartReflection(reflection_part, IID_PPV_ARGS(&shader_reflection));
    if (FAILED(hr)) {
        log->Error("Failed to get reflection data");
        return false;
    }
    SCOPE_EXIT(shader_reflection->Release());

    D3D12_SHADER_DESC shader_desc = {};
    hr = shader_reflection->GetDesc(&shader_desc);
    if (FAILED(hr)) {
        log->Error("Failed to get shader description");
        return false;
    }

    for (uint32_t i = 0; i < shader_desc.InputParameters; ++i) {
        // TODO: fill input attributes
    }

    for (uint32_t i = 0; i < shader_desc.BoundResources; ++i) {
        D3D12_SHADER_INPUT_BIND_DESC shader_input_bind_desc = {};
        hr = shader_reflection->GetResourceBindingDesc(i, &shader_input_bind_desc);
        if (FAILED(hr)) {
            log->Error("Failed to get resource binding description");
            return false;
        }

        // TODO: Do not rely on the name here!
        if (strcmp(shader_input_bind_desc.Name, "UniformParams") == 0) {
            pc_ranges.push_back({shader_input_bind_desc.BindPoint, shader_input_bind_desc.BindCount});
        } else {
            Descr &new_item = unif_bindings.emplace_back();
            strncpy_s(new_item.name, sizeof(new_item.name), shader_input_bind_desc.Name, sizeof(new_item.name) - 1);
            new_item.input_type = shader_input_bind_desc.Type;
            new_item.loc = int(shader_input_bind_desc.BindPoint);
            new_item.space = int(shader_input_bind_desc.Space);
            new_item.count = int(shader_input_bind_desc.BindCount);
        }
    }
#endif
    return true;
}
