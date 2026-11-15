#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "ShaderMtl.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "../../Log.h"
#include "ContextMtl.h"

namespace Ray::Mtl {
Shader::Shader(std::string_view name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log) {
    Init(name, ctx, shader_code, type, log);
}

Shader::~Shader() {
    if (function_) {
        function_->release();
    }
    if (library_) {
        library_->release();
    }
}

Shader &Shader::operator=(Shader &&rhs) noexcept {
    if (function_) {
        function_->release();
    }
    if (library_) {
        library_->release();
    }
    ctx_ = std::exchange(rhs.ctx_, nullptr);
    library_ = std::exchange(rhs.library_, nullptr);
    function_ = std::exchange(rhs.function_, nullptr);
    type_ = std::exchange(rhs.type_, eShaderType::_Count);
    name_ = std::move(rhs.name_);
    refl_ = std::move(rhs.refl_);
    memcpy(workgroup_, rhs.workgroup_, sizeof(workgroup_));
    return (*this);
}

bool Shader::Init(std::string_view name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log) {
    ctx_ = ctx;
    name_ = std::string(name);
    type_ = type;
    return InitFromPayload(shader_code, type, log);
}

bool Shader::InitFromPayload(Span<const uint8_t> payload, eShaderType, ILog *log) {
    if (payload.size() < 8 + 24) {
        log->Error("Empty Metal shader blob!");
        return false;
    }

    uint32_t refl_size = 0;
    char magic[4];
    memcpy(&refl_size, payload.data() + payload.size() - 8, 4);
    memcpy(magic, payload.data() + payload.size() - 4, 4);
    if (memcmp(magic, "MLR1", 4) != 0 || refl_size < 24 || refl_size + 8 > payload.size()) {
        log->Error("Bad Metal blob trailer!");
        return false;
    }

    const uint8_t *refl = payload.data() + payload.size() - 8 - refl_size;
    const size_t metallib_len = payload.size() - 8 - refl_size;

    memcpy(magic, refl, 4);
    uint32_t version = 0, count = 0;
    memcpy(&version, refl + 4, 4);
    memcpy(workgroup_, refl + 8, 12);
    memcpy(&count, refl + 20, 4);
    if (memcmp(magic, "MLR1", 4) != 0 || version != 1 || refl_size != 24 + count * 24) {
        log->Error("Bad Metal refl header!");
        return false;
    }

    refl_.resize(count);
    memcpy(refl_.data(), refl + 24, count * sizeof(MslBindingDesc));

    // dispatch_data with a copied buffer: the blob array may be transient, the data outlives the load call
    void *copy = malloc(metallib_len);
    memcpy(copy, payload.data(), metallib_len);
    dispatch_data_t dd = dispatch_data_create(copy, metallib_len, NULL, DISPATCH_DATA_DESTRUCTOR_FREE);

    NS::Error *err = nullptr;
    library_ = ctx_->device()->newLibrary(dd, &err);
    dispatch_release(dd);
    if (library_ == nullptr) {
        log->Error("Failed to load metallib: %s", err ? err->localizedDescription()->utf8String() : "?");
        return false; // out-param error is +0 autoreleased - do NOT release
    }

    // every blob is compiled from one kernel: exactly one entry point, named main0
    function_ = library_->newFunction(MTLSTR("main0"));
    if (function_ == nullptr) {
        log->Error("main0 entry point missing from metallib!");
        return false;
    }
    return true;
}
} // namespace Ray::Mtl
