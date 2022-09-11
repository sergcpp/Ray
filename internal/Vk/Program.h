#pragma once

#include <cstdint>
#include <cstring>

#include <array>
#include <string>

#include "../SmallVector.h"
#include "Shader.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace Ray {
class ILog;
namespace Vk {
class Program {
    uint32_t flags_ = 0;
    std::array<Shader *, int(eShaderType::_Count)> shaders_ = {};
    SmallVector<Attribute, 8> attributes_;
    SmallVector<Uniform, 16> uniforms_;
    std::string name_;

    Context *ctx_ = nullptr;
    SmallVector<VkDescriptorSetLayout, 4> descr_set_layouts_;
    SmallVector<VkPushConstantRange, 8> pc_ranges_;

    bool InitDescrSetLayouts(ILog *log);
    void InitBindings(ILog *log);

    void Destroy();

  public:
    Program() = default;
    Program(const char *name, Context *ctx, Shader *vs_ref, Shader *fs_ref, Shader *tcs_ref, Shader *tes_ref,
            ILog *log);
    Program(const char *name, Context *ctx, Shader *cs_ref, ILog *log);
    Program(const char *name, Context *ctx, Shader *raygen_ref, Shader *closesthit_ref, Shader *anyhit_ref,
            Shader *miss_ref, Shader *intersection_ref, ILog *log);

    Program(const Program &rhs) = delete;
    Program(Program &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Program();

    Program &operator=(const Program &rhs) = delete;
    Program &operator=(Program &&rhs) noexcept;

    uint32_t flags() const { return flags_; }
    bool ready() const {
        return (shaders_[int(eShaderType::Vert)] && shaders_[int(eShaderType::Frag)]) ||
               shaders_[int(eShaderType::Comp)] ||
               (shaders_[int(eShaderType::RayGen)] &&
                (shaders_[int(eShaderType::ClosestHit)] || shaders_[int(eShaderType::AnyHit)]) &&
                shaders_[int(eShaderType::Miss)]);
    }
    bool has_tessellation() const { return shaders_[int(eShaderType::Tesc)] && shaders_[int(eShaderType::Tese)]; }
    const std::string &name() const { return name_; }

    const Attribute &attribute(const int i) const { return attributes_[i]; }
    const Attribute &attribute(const char *name) const {
        for (int i = 0; i < int(attributes_.size()); i++) {
            if (attributes_[i].name == name) {
                return attributes_[i];
            }
        }
        return attributes_[0];
    }

    const Uniform &uniform(const int i) const { return uniforms_[i]; }
    const Uniform &uniform(const char *name) const {
        for (int i = 0; i < int(uniforms_.size()); i++) {
            if (uniforms_[i].name == name) {
                return uniforms_[i];
            }
        }
        return uniforms_[0];
    }

    const Shader *shader(eShaderType type) const { return shaders_[int(type)]; }

    uint32_t descr_set_layouts_count() const { return uint32_t(descr_set_layouts_.size()); }
    const VkDescriptorSetLayout *descr_set_layouts() const { return descr_set_layouts_.cdata(); }

    uint32_t pc_range_count() const { return uint32_t(pc_ranges_.size()); }
    const VkPushConstantRange *pc_ranges() const { return pc_ranges_.data(); }

    bool Init(Shader *vs_ref, Shader *fs_ref, Shader *tcs_ref, Shader *tes_ref, ILog *log);
    bool Init(Shader *cs_ref, ILog *log);
    bool Init(Shader *raygen_ref, Shader *closesthit_ref, Shader *anyhit_ref, Shader *miss_ref,
              Shader *intersection_ref, ILog *log);
};
} // namespace Vk
} // namespace Ray

#ifdef _MSC_VER
#pragma warning(pop)
#endif