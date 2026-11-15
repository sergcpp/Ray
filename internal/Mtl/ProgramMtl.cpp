#include "ProgramMtl.h"

#include <utility>

#include "../../Log.h"
#include "ContextMtl.h"

namespace Ray::Mtl {
Program::Program(std::string_view name, Context *ctx, Shader *cs_ref, ILog *log) {
    if (Init(cs_ref, log)) {
        name_ = std::string(name);
    }
    ctx_ = ctx;
}

Program &Program::operator=(Program &&rhs) noexcept {
    shaders_ = std::move(rhs.shaders_);
    name_ = std::move(rhs.name_);
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    return (*this);
}

bool Program::Init(Shader *cs_ref, ILog *log) {
    if (cs_ref == nullptr || !cs_ref->ready()) {
        log->Error("Failed to init program %s: compute shader not ready!", name_.c_str());
        return false;
    }

    for (auto &s : shaders_) {
        s = nullptr;
    }
    shaders_[int(eShaderType::Comp)] = cs_ref;
    return true;
}
} // namespace Ray::Mtl
