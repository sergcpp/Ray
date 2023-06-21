#pragma once

#include <cstdint>

#include <string>
#include <vector>

#include "../SmallVector.h"

struct ID3D12Device;
typedef enum _D3D_SHADER_INPUT_TYPE D3D_SHADER_INPUT_TYPE;

namespace Ray {
class ILog;
namespace Dx {
class Context;

struct Range {
    uint32_t offset;
    uint32_t size;
};

struct Descr {
    char name[24];
    int loc = -1;
    D3D_SHADER_INPUT_TYPE input_type;
    int space = 0, count = 0;
    // VkFormat format = VK_FORMAT_UNDEFINED;
};
inline bool operator==(const Descr &lhs, const Descr &rhs) { return lhs.loc == rhs.loc && lhs.name == rhs.name; }
typedef Descr Attribute;
typedef Descr Uniform;
typedef Descr UniformBlock;

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

class Shader {
    ID3D12Device *device_ = {};
    std::vector<uint8_t> shader_code_;
    eShaderType type_ = eShaderType::_Count;

    std::string name_;

    bool InitFromCSO(const uint8_t *shader_code, int code_size, eShaderType type, ILog *log);

  public:
    SmallVector<Descr, 16> attr_bindings, unif_bindings;
    SmallVector<Range, 4> pc_ranges;

    Shader() = default;
    Shader(const char *name, Context *ctx, const uint8_t *shader_code, int code_size, eShaderType type, ILog *log);
    Shader(const char *name, Context *ctx, Span<const uint8_t> shader_code, eShaderType type, ILog *log)
        : Shader(name, ctx, shader_code.data(), int(shader_code.size()), type, log) {}

    Shader(const Shader &rhs) = delete;
    Shader(Shader &&rhs) noexcept { (*this) = std::move(rhs); }
    ~Shader();

    Shader &operator=(const Shader &rhs) = delete;
    Shader &operator=(Shader &&rhs) noexcept;

    bool ready() const { return !shader_code_.empty(); }
    ID3D12Device *device() const { return device_; }
    const std::vector<uint8_t> &shader_code() const { return shader_code_; }
    eShaderType type() const { return type_; }
    const std::string &name() const { return name_; }

    bool Init(const uint8_t *shader_code, int code_size, eShaderType type, ILog *log);
};

} // namespace Dx
} // namespace Ray