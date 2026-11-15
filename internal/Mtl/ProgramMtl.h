#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "../SmallVector.h"
#include "ShaderMtl.h"

namespace Ray {
class ILog;
namespace Mtl {
class Program {
    std::array<Shader *, int(eShaderType::_Count)> shaders_ = {};
    std::string name_;

    Context *ctx_ = nullptr;

  public:
    Program() = default;
    Program(std::string_view name, Context *ctx, Shader *cs_ref, ILog *log);

    Program(const Program &rhs) = delete;
    Program(Program &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Program() = default;

    Program &operator=(const Program &rhs) = delete;
    Program &operator=(Program &&rhs) noexcept;

    bool ready() const { return shaders_[int(eShaderType::Comp)] && shaders_[int(eShaderType::Comp)]->ready(); }
    const std::string &name() const { return name_; }

    const Shader *shader(eShaderType type) const { return shaders_[int(type)]; }

    bool Init(Shader *cs_ref, ILog *log);
};
} // namespace Mtl
} // namespace Ray
