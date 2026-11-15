#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../../Span.h"
#include "ApiMtl.h"

namespace Ray {
class ILog;
namespace Mtl {
class Context;

enum class eShaderType : uint8_t {
    Vert,
    Frag,
    Tesc,
    Tese,
    Comp,
    RayGen,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection,
    _Count
};

// Payload = [metallib | 'MLR1' refl | u32 size + 'MLR1' trailer], see scripts/compile_shaders.py
class Shader {
    Context *ctx_ = nullptr;
    MTL::Library *library_ = nullptr;
    MTL::Function *function_ = nullptr;
    eShaderType type_ = eShaderType::_Count;
    std::string name_;
    std::vector<MslBindingDesc> refl_;
    uint32_t workgroup_[3] = {};

    bool InitFromPayload(Span<const uint8_t> payload, eShaderType type, ILog *log);

  public:
    Shader() = default;
    Shader(std::string_view name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log);
    Shader(const Shader &rhs) = delete;
    Shader(Shader &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Shader();

    Shader &operator=(const Shader &rhs) = delete;
    Shader &operator=(Shader &&rhs) noexcept;

    bool ready() const { return function_ != nullptr; }
    MTL::Library *library() const { return library_; }
    MTL::Function *function() const { return function_; }
    eShaderType type() const { return type_; }
    std::string_view name() const { return name_; }

    const std::vector<MslBindingDesc> &refl() const { return refl_; }
    const uint32_t *workgroup_size() const { return workgroup_; }

    bool Init(std::string_view name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log);
};

} // namespace Mtl
} // namespace Ray
