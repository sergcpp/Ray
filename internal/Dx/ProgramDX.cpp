#include "ProgramDX.h"

#include <stdexcept>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

#include "../../Log.h"
#include "ContextDX.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

Ray::Dx::Program::Program(std::string_view name, Context *ctx, Shader *vs_ref, Shader *fs_ref, Shader *tcs_ref,
                          Shader *tes_ref, ILog *log)
    : name_(name), ctx_(ctx) {
    if (!Init(vs_ref, fs_ref, tcs_ref, tes_ref, log)) {
        throw std::runtime_error("Program Init error!");
    }
}

Ray::Dx::Program::Program(std::string_view name, Context *ctx, Shader *cs_ref, ILog *log) : name_(name), ctx_(ctx) {
    if (!Init(cs_ref, log)) {
        throw std::runtime_error("Program Init error!");
    }
}

Ray::Dx::Program::Program(std::string_view name, Context *ctx, Shader *raygen_ref, Shader *closesthit_ref,
                          Shader *anyhit_ref, Shader *miss_ref, Shader *intersection_ref, ILog *log)
    : name_(name), ctx_(ctx) {
    if (!Init(raygen_ref, closesthit_ref, anyhit_ref, miss_ref, intersection_ref, log)) {
        throw std::runtime_error("Program Init error!");
    }
}

Ray::Dx::Program::~Program() { Destroy(); }

Ray::Dx::Program &Ray::Dx::Program::operator=(Program &&rhs) noexcept {
    Destroy();

    shaders_ = rhs.shaders_;
    attributes_ = std::move(rhs.attributes_);
    uniforms_ = std::move(rhs.uniforms_);
    for (int i = 0; i < std::size(descr_indices_); ++i) {
        descr_indices_[i] = std::move(rhs.descr_indices_[i]);
    }
    pc_param_index_ = std::exchange(rhs.pc_param_index_, -1);
    name_ = std::move(rhs.name_);

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    root_signature_ = std::exchange(rhs.root_signature_, nullptr);
    // descr_set_layouts_ = std::move(rhs.descr_set_layouts_);

    return *this;
}

void Ray::Dx::Program::Destroy() {
    // for (VkDescriptorSetLayout &l : descr_set_layouts_) {
    //     if (l) {
    //         vkDestroyDescriptorSetLayout(ctx_->device(), l, nullptr);
    //     }
    // }
    // descr_set_layouts_.clear();

    if (root_signature_) {
        root_signature_->Release();
        root_signature_ = nullptr;
    }
}

bool Ray::Dx::Program::Init(Shader *vs_ref, Shader *fs_ref, Shader *tcs_ref, Shader *tes_ref, ILog *log) {
    if (!vs_ref || !fs_ref) {
        return false;
    }

    // store shaders
    shaders_[int(eShaderType::Vert)] = vs_ref;
    shaders_[int(eShaderType::Frag)] = fs_ref;
    shaders_[int(eShaderType::Tesc)] = tcs_ref;
    shaders_[int(eShaderType::Tese)] = tes_ref;

    if (!InitRootSignature(log)) {
        log->Error("Failed to initialize descriptor set layouts! (%s)", name_.c_str());
        return false;
    }
    InitBindings(log);

    return true;
}

bool Ray::Dx::Program::Init(Shader *cs_ref, ILog *log) {
    if (!cs_ref) {
        return false;
    }

    // store shader
    shaders_[int(eShaderType::Comp)] = cs_ref;

    if (!InitRootSignature(log)) {
        log->Error("Failed to initialize descriptor set layouts! (%s)", name_.c_str());
        return false;
    }
    InitBindings(log);

    return true;
}

bool Ray::Dx::Program::Init(Shader *raygen_ref, Shader *closesthit_ref, Shader *anyhit_ref, Shader *miss_ref,
                            Shader *intersection_ref, ILog *log) {
    if (!raygen_ref || (!closesthit_ref && !anyhit_ref) || !miss_ref) {
        return false;
    }

    // store shaders
    shaders_[int(eShaderType::RayGen)] = raygen_ref;
    shaders_[int(eShaderType::ClosestHit)] = closesthit_ref;
    shaders_[int(eShaderType::AnyHit)] = anyhit_ref;
    shaders_[int(eShaderType::Miss)] = miss_ref;
    shaders_[int(eShaderType::Intersection)] = intersection_ref;

    if (!InitRootSignature(log)) {
        log->Error("Failed to initialize descriptor set layouts! (%s)", name_.c_str());
        return false;
    }
    InitBindings(log);

    return true;
}

bool Ray::Dx::Program::InitRootSignature(ILog *log) {
    SmallVector<D3D12_STATIC_SAMPLER_DESC, 16> static_samplers;

    ID3D12Device *device = nullptr;

    D3D12_ROOT_PARAMETER root_parameters[3] = {};
    SmallVector<D3D12_DESCRIPTOR_RANGE, 32> descriptor_ranges[3];
    short descriptor_count[3] = {};

    for (int i = 0; i < int(eShaderType::_Count); ++i) {
        const Shader *sh_ref = shaders_[i];
        if (!sh_ref) {
            continue;
        }

        const Shader &sh = (*sh_ref);
        assert(!device || device == sh.device());
        device = sh.device();

        for (const Descr &u : sh.unif_bindings) {
            const int rp_index = (u.input_type == D3D_SIT_SAMPLER) ? 1 : (u.space != 0) ? 2 : 0;

            if (u.loc >= int(descr_indices_[rp_index].size())) {
                descr_indices_[rp_index].resize(std::max(u.loc + 1, int(descr_indices_[rp_index].size())), -1);
            }
            descr_indices_[rp_index][u.loc] = descriptor_count[rp_index];

            D3D12_DESCRIPTOR_RANGE_TYPE range_type = {};
            if (u.input_type == D3D_SIT_BYTEADDRESS || u.input_type == D3D_SIT_TEXTURE ||
                u.input_type == D3D_SIT_RTACCELERATIONSTRUCTURE) {
                range_type = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            } else if (u.input_type == D3D_SIT_UAV_RWBYTEADDRESS || u.input_type == D3D_SIT_UAV_RWTYPED ||
                       u.input_type == D3D_SIT_UAV_RWSTRUCTURED) {
                range_type = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            } else if (u.input_type == D3D_SIT_CBUFFER) {
                range_type = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            } else if (u.input_type == D3D_SIT_SAMPLER) {
                range_type = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            }

            D3D12_DESCRIPTOR_RANGE &desc_range = descriptor_ranges[rp_index].emplace_back();
            desc_range.RangeType = range_type;
            desc_range.NumDescriptors = u.count ? u.count : -1;
            desc_range.BaseShaderRegister = u.loc;
            desc_range.RegisterSpace = u.space;
            desc_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            descriptor_count[rp_index] += desc_range.NumDescriptors;

            /*auto &bindings = layout_bindings[u.set];

            auto it = std::find_if(std::begin(bindings), std::end(bindings),
                                   [&u](const VkDescriptorSetLayoutBinding &b) { return u.loc == b.binding; });

            if (it == std::end(bindings)) {
                auto &new_binding = bindings.emplace_back();
                new_binding.binding = u.loc;
                new_binding.descriptorType = u.desc_type;

                if (u.unbounded_array && u.desc_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    assert(u.count == 1);
                    new_binding.descriptorCount = ctx_->max_combined_image_samplers();
                } else {
                    new_binding.descriptorCount = u.count;
                }

                new_binding.stageFlags = g_shader_stages_vk[i];
                new_binding.pImmutableSamplers = nullptr;
            } else {
                it->stageFlags |= g_shader_stages_vk[i];
            }*/
        }

        for (const Range &u : sh.pc_ranges) {
            pc_param_index_ = descriptor_count[0]++;

            D3D12_DESCRIPTOR_RANGE &desc_range = descriptor_ranges[0].emplace_back();
            desc_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            desc_range.NumDescriptors = 1;
            desc_range.BaseShaderRegister = u.offset;
            desc_range.RegisterSpace = 0;
            desc_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        }
    }

    int rp_count = 0;
    for (int i = 0; i < std::size(root_parameters); ++i) {
        if (descriptor_ranges[i].empty()) {
            continue;
        }

        ++rp_count;

        D3D12_ROOT_PARAMETER &rp = root_parameters[i];
        rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rp.DescriptorTable.pDescriptorRanges = descriptor_ranges[i].data();
        rp.DescriptorTable.NumDescriptorRanges = UINT(descriptor_ranges[i].size());
    }

    if (!device) {
        return false;
    }

    D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.pParameters = root_parameters;
    root_signature_desc.NumParameters = rp_count;
    root_signature_desc.pStaticSamplers = static_samplers.data();
    root_signature_desc.NumStaticSamplers = UINT(static_samplers.size());
    root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob *signature_blob;
    HRESULT hr =
        D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature_blob, nullptr);
    if (FAILED(hr)) {
        return false;
    }

    hr = device->CreateRootSignature(0, signature_blob->GetBufferPointer(), signature_blob->GetBufferSize(),
                                     IID_PPV_ARGS(&root_signature_));
    if (FAILED(hr)) {
        return false;
    }

    if (signature_blob) {
        signature_blob->Release();
        signature_blob = nullptr;
    }

    /*for (int i = 0; i < 4; ++i) {
        if (layout_bindings[i].empty()) {
            continue;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = uint32_t(layout_bindings[i].size());
        layout_info.pBindings = layout_bindings[i].cdata();

        VkDescriptorBindingFlagsEXT bind_flag = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT};
        extended_info.pNext = nullptr;
        extended_info.bindingCount = 1u;
        extended_info.pBindingFlags = &bind_flag;

        if (i == 1) {
            layout_info.pNext = &extended_info;
        }

        descr_set_layouts_.emplace_back();
        const VkResult res =
            vkCreateDescriptorSetLayout(ctx_->device(), &layout_info, nullptr, &descr_set_layouts_.back());

        if (res != VK_SUCCESS) {
            log->Error("Failed to create descriptor set layout!");
            return false;
        }
    }*/

    return true;
}

void Ray::Dx::Program::InitBindings(ILog *log) {
    attributes_.clear();
    uniforms_.clear();
    // pc_ranges_.clear();

    for (int i = 0; i < int(eShaderType::_Count); ++i) {
        const Shader *sh_ref = shaders_[i];
        if (!sh_ref) {
            continue;
        }

        const Shader &sh = (*sh_ref);
        for (const Descr &u : sh.unif_bindings) {
            auto it = std::find(std::begin(uniforms_), std::end(uniforms_), u);
            if (it == std::end(uniforms_)) {
                uniforms_.emplace_back(u);
            }
        }

        /*for (const Range r : sh.pc_ranges) {
            auto it = std::find_if(std::begin(pc_ranges_), std::end(pc_ranges_), [&](const VkPushConstantRange &rng)
        { return r.offset == rng.offset && r.size == rng.size;
            });

            if (it == std::end(pc_ranges_)) {
                VkPushConstantRange &new_rng = pc_ranges_.emplace_back();
                new_rng.stageFlags = g_shader_stages_vk[i];
                new_rng.offset = r.offset;
                new_rng.size = r.size;
            } else {
                it->stageFlags |= g_shader_stages_vk[i];
            }
        }*/
    }

    if (shaders_[int(eShaderType::Vert)]) {
        for (const Descr &a : shaders_[int(eShaderType::Vert)]->attr_bindings) {
            attributes_.emplace_back(a);
        }
    }

    /*log->Info("PROGRAM %s", name_.c_str());

    // Print all attributes
    log->Info("\tATTRIBUTES");
    for (int i = 0; i < int(attributes_.size()); i++) {
        if (attributes_[i].loc == -1) {
            continue;
        }
        log->Info("\t\t%s : %i", attributes_[i].name, attributes_[i].loc);
    }

    // Print all uniforms
    log->Info("\tUNIFORMS");
    for (int i = 0; i < int(uniforms_.size()); i++) {
        if (uniforms_[i].loc == -1) {
            continue;
        }
        log->Info("\t\t%s : %i", uniforms_[i].name, uniforms_[i].loc);
    }*/
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
