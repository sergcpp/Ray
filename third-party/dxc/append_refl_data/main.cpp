
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12shader.h>
#include <d3dcommon.h>
#include <dxcapi.h>

#include "../../../internal/Dx/ShaderDX.h"
#include "../../../internal/ScopeExit.h"

namespace {
#define _DXC_FOURCC(ch0, ch1, ch2, ch3) (uint32_t(ch0) | uint32_t(ch1) << 8 | uint32_t(ch2) << 16 | uint32_t(ch3) << 24)

// const uint32_t _DXC_PART_PDB = _DXC_FOURCC('I', 'L', 'D', 'B');
// const uint32_t _DXC_PART_PDB_NAME = _DXC_FOURCC('I', 'L', 'D', 'N');
// const uint32_t _DXC_PART_PRIVATE_DATA = _DXC_FOURCC('P', 'R', 'I', 'V');
// const uint32_t _DXC_PART_ROOT_SIGNATURE = _DXC_FOURCC('R', 'T', 'S', '0');
// const uint32_t _DXC_PART_DXIL = _DXC_FOURCC('D', 'X', 'I', 'L');
const uint32_t _DXC_PART_REFLECTION_DATA = _DXC_FOURCC('S', 'T', 'A', 'T');
// const uint32_t _DXC_PART_SHADER_HASH = _DXC_FOURCC('H', 'A', 'S', 'H');

#undef _DXC_FOURCC

struct ShaderBlobAdapter : public IDxcBlob {
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
};
} // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        return -1;
    }

    std::fstream inout_file(argv[1], std::ios_base::in | std::ios_base::out | std::ios::ate | std::ios::binary);
    if (!inout_file.good()) {
        printf("Failed to open %s\n", argv[1]);
        return -1;
    }
    const size_t file_size = size_t(inout_file.tellg());
    inout_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> inout_file_data(file_size);
    inout_file.read((char *)inout_file_data.data(), file_size);

    IDxcContainerReflection *container_reflection = {};
    HRESULT hr = DxcCreateInstance(CLSID_DxcContainerReflection, IID_PPV_ARGS(&container_reflection));
    if (FAILED(hr)) {
        printf("Failed to create DxcContainerReflection\n");
        return -1;
    }
    SCOPE_EXIT(container_reflection->Release());

    ShaderBlobAdapter shader_blob(inout_file_data.data(), int(inout_file_data.size()));

    hr = container_reflection->Load(&shader_blob);
    if (FAILED(hr)) {
        printf("Failed to load shader blob\n");
        return -1;
    }

    uint32_t reflection_part = 0;
    hr = container_reflection->FindFirstPartKind(_DXC_PART_REFLECTION_DATA, &reflection_part);
    if (FAILED(hr)) {
        printf("Failed to find reflection data\n");
        return -1;
    }

    ID3D12ShaderReflection *shader_reflection = {};
    hr = container_reflection->GetPartReflection(reflection_part, IID_PPV_ARGS(&shader_reflection));
    if (FAILED(hr)) {
        printf("Failed to get reflection data\n");
        return -1;
    }
    SCOPE_EXIT(shader_reflection->Release());

    ////////////////////////////

    std::vector<Ray::Dx::Descr> unif_bindings;
    std::vector<Ray::Dx::Range> pc_ranges;

    D3D12_SHADER_DESC shader_desc = {};
    hr = shader_reflection->GetDesc(&shader_desc);
    if (FAILED(hr)) {
        printf("Failed to get shader description\n");
        return -1;
    }

    for (uint32_t i = 0; i < shader_desc.InputParameters; ++i) {
        // TODO: fill input attributes
    }

    for (uint32_t i = 0; i < shader_desc.BoundResources; ++i) {
        D3D12_SHADER_INPUT_BIND_DESC shader_input_bind_desc = {};
        hr = shader_reflection->GetResourceBindingDesc(i, &shader_input_bind_desc);
        if (FAILED(hr)) {
            printf("Failed to get resource binding description\n");
            return false;
        }

        // TODO: Do not rely on the name here!
        if (strcmp(shader_input_bind_desc.Name, "UniformParams") == 0) {
            pc_ranges.push_back({shader_input_bind_desc.BindPoint, shader_input_bind_desc.BindCount});
        } else {
            unif_bindings.emplace_back();
            Ray::Dx::Descr &new_item = unif_bindings.back();
            strncpy_s(new_item.name, sizeof(new_item.name), shader_input_bind_desc.Name, sizeof(new_item.name) - 1);
            new_item.input_type = shader_input_bind_desc.Type;
            new_item.loc = int(shader_input_bind_desc.BindPoint);
            new_item.space = int(shader_input_bind_desc.Space);
            new_item.count = int(shader_input_bind_desc.BindCount);
        }
    }

    inout_file.seekg(file_size, std::ios::beg);

    const uint32_t pc_count = uint32_t(pc_ranges.size());
    inout_file.write((const char *)&pc_count, sizeof(uint32_t));
    if (pc_count) {
        inout_file.write((const char *)pc_ranges.data(), pc_count * sizeof(Ray::Dx::Range));
    }

    const uint32_t ub_count = uint32_t(unif_bindings.size());
    inout_file.write((const char *)&ub_count, sizeof(uint32_t));
    if (ub_count) {
        inout_file.write((const char *)unif_bindings.data(), ub_count * sizeof(Ray::Dx::Descr));
    }

    return 0;
}
