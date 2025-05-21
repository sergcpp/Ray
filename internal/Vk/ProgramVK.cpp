#include "ProgramVK.h"

#include <stdexcept>

#include "../../Log.h"
#include "ContextVK.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace Ray {
namespace Vk {
extern const VkShaderStageFlagBits g_shader_stages_vk[];
}
} // namespace Ray

Ray::Vk::Program::Program(std::string_view name, Context *ctx, Shader *vs_ref, Shader *fs_ref, Shader *tcs_ref,
                          Shader *tes_ref, ILog *log)
    : name_(name), ctx_(ctx) {
    if (!Init(vs_ref, fs_ref, tcs_ref, tes_ref, log)) {
        throw std::runtime_error("Program Init error!");
    }
}

Ray::Vk::Program::Program(std::string_view name, Context *ctx, Shader *cs_ref, ILog *log) : name_(name), ctx_(ctx) {
    if (!Init(cs_ref, log)) {
        throw std::runtime_error("Program Init error!");
    }
}

Ray::Vk::Program::Program(std::string_view name, Context *ctx, Shader *raygen_ref, Shader *closesthit_ref,
                          Shader *anyhit_ref, Shader *miss_ref, Shader *intersection_ref, ILog *log)
    : name_(name), ctx_(ctx) {
    if (!Init(raygen_ref, closesthit_ref, anyhit_ref, miss_ref, intersection_ref, log)) {
        throw std::runtime_error("Program Init error!");
    }
}

Ray::Vk::Program::~Program() { Destroy(); }

Ray::Vk::Program &Ray::Vk::Program::operator=(Program &&rhs) noexcept {
    Destroy();

    shaders_ = rhs.shaders_;
    attributes_ = std::move(rhs.attributes_);
    uniforms_ = std::move(rhs.uniforms_);
    pc_ranges_ = std::move(rhs.pc_ranges_);
    name_ = std::move(rhs.name_);

    ctx_ = std::exchange(rhs.ctx_, nullptr);
    descr_set_layouts_ = std::move(rhs.descr_set_layouts_);

    return *this;
}

void Ray::Vk::Program::Destroy() {
    for (VkDescriptorSetLayout &l : descr_set_layouts_) {
        if (l) {
            ctx_->api().vkDestroyDescriptorSetLayout(ctx_->device(), l, nullptr);
        }
    }
    descr_set_layouts_.clear();
}

bool Ray::Vk::Program::Init(Shader *vs_ref, Shader *fs_ref, Shader *tcs_ref, Shader *tes_ref, ILog *log) {
    if (!vs_ref || !fs_ref) {
        return false;
    }

    // store shaders
    shaders_[int(eShaderType::Vert)] = vs_ref;
    shaders_[int(eShaderType::Frag)] = fs_ref;
    shaders_[int(eShaderType::Tesc)] = tcs_ref;
    shaders_[int(eShaderType::Tese)] = tes_ref;

    if (!InitDescrSetLayouts(log)) {
        log->Error("Failed to initialize descriptor set layouts! (%s)", name_.c_str());
        return false;
    }
    InitBindings(log);

    return true;
}

bool Ray::Vk::Program::Init(Shader *cs_ref, ILog *log) {
    if (!cs_ref) {
        return false;
    }

    // store shader
    shaders_[int(eShaderType::Comp)] = cs_ref;

    if (!InitDescrSetLayouts(log)) {
        log->Error("Failed to initialize descriptor set layouts! (%s)", name_.c_str());
        return false;
    }
    InitBindings(log);

    return true;
}

bool Ray::Vk::Program::Init(Shader *raygen_ref, Shader *closesthit_ref, Shader *anyhit_ref, Shader *miss_ref,
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

    if (!InitDescrSetLayouts(log)) {
        log->Error("Failed to initialize descriptor set layouts! (%s)", name_.c_str());
        return false;
    }
    InitBindings(log);

    return true;
}

bool Ray::Vk::Program::InitDescrSetLayouts(ILog *log) {
    SmallVector<VkDescriptorSetLayoutBinding, 16> layout_bindings[4];

    for (int i = 0; i < int(eShaderType::_Count); ++i) {
        const Shader *sh_ref = shaders_[i];
        if (!sh_ref) {
            continue;
        }

        const Shader &sh = (*sh_ref);
        for (const Descr &u : sh.unif_bindings) {
            auto &bindings = layout_bindings[u.set];

            auto it = std::find_if(std::begin(bindings), std::end(bindings),
                                   [&u](const VkDescriptorSetLayoutBinding &b) { return u.loc == b.binding; });

            if (it == std::end(bindings)) {
                auto &new_binding = bindings.emplace_back();
                new_binding.binding = u.loc;
                new_binding.descriptorType = u.desc_type;

                if (u.count == 0 && (u.desc_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                                     u.desc_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)) {
                    new_binding.descriptorCount = ctx_->max_combined_image_samplers();
                } else {
                    new_binding.descriptorCount = u.count;
                }

                new_binding.stageFlags = g_shader_stages_vk[i];
                new_binding.pImmutableSamplers = nullptr;
            } else {
                it->stageFlags |= g_shader_stages_vk[i];
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        if (layout_bindings[i].empty()) {
            continue;
        }

        VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = layout_bindings[i].size();
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
            ctx_->api().vkCreateDescriptorSetLayout(ctx_->device(), &layout_info, nullptr, &descr_set_layouts_.back());

        if (res != VK_SUCCESS) {
            log->Error("Failed to create descriptor set layout!");
            return false;
        }
    }

    return true;
}

void Ray::Vk::Program::InitBindings(ILog *log) {
    attributes_.clear();
    uniforms_.clear();
    pc_ranges_.clear();

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

        for (const Range r : sh.pc_ranges) {
            auto it = std::find_if(std::begin(pc_ranges_), std::end(pc_ranges_), [&](const VkPushConstantRange &rng) {
                return r.offset == rng.offset && r.size == rng.size;
            });

            if (it == std::end(pc_ranges_)) {
                VkPushConstantRange &new_rng = pc_ranges_.emplace_back();
                new_rng.stageFlags = g_shader_stages_vk[i];
                new_rng.offset = r.offset;
                new_rng.size = r.size;
            } else {
                it->stageFlags |= g_shader_stages_vk[i];
            }
        }
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
        log->Info("\t\t%s : %i", attributes_[i].name.c_str(), attributes_[i].loc);
    }

    // Print all uniforms
    log->Info("\tUNIFORMS");
    for (int i = 0; i < int(uniforms_.size()); i++) {
        if (uniforms_[i].loc == -1) {
            continue;
        }
        log->Info("\t\t%s : %i", uniforms_[i].name.c_str(), uniforms_[i].loc);
    }*/
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
