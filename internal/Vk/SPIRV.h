#pragma once

#include <cstdint>

#include "../../Span.h"
#include "../HashMap32.h"
#include "BufferVK.h"

namespace Ray {
enum class eEndianness { Invalid = -1, Little, Big };

uint32_t fix_endianness(uint32_t v, eEndianness target);

namespace Vk {
const uint32_t SPIRV_MAGIC = 0x07230203;
const uint32_t SPIRV_DATA_ALIGNMENT = 16;

struct spirv_header_t {
    uint32_t magic;
    uint32_t version;
    uint32_t generator;
    uint32_t bound;
    uint32_t schema;
    const uint32_t *instructions;
};

const int SPIRV_INDEX_MAGIC_NUMBER = 0;
const int SPIRV_INDEX_VERSION_NUMBER = 1;
const int SPIRV_INDEX_GENERATOR_NUMBER = 2;
const int SPIRV_INDEX_BOUND = 3;
const int SPIRV_INDEX_SCHEMA = 4;
const int SPIRV_INDEX_INSTRUCTION = 5;

eEndianness spirv_test_endianness(Span<const uint32_t> code);

enum class eSPIRVOp {
    Name = 5,
    TypeInt = 21,
    TypeFloat = 22,
    TypeVector = 23,
    TypeImage = 25,
    TypeSampledImage = 27,
    TypeArray = 28,
    TypeRuntimeArray = 29,
    TypeStruct = 30,
    TypePointer = 32,
    Constant = 43,
    Variable = 59,
    Decorate = 71,
    TypeAccelerationStructureKHR = 5341
};

enum class eSPIRVStorageClass {
    UniformConstant,
    Input,
    Uniform,
    Output,
    Workgroup,
    CrossWorkgroup,
    Private,
    Function,
    Generic,
    PushConstant,
    AtomicCounter,
    Image,
    StorageBuffer
};

enum class eSPIRVDim {
    _1D = 0,
    _2D = 1,
    _3D = 2,
    Cube = 3,
    Rect = 4,
    Buffer = 5,
    SubpassData = 6,
    TileImageDataEXT = 4173
};

enum class eSPIRVDecoration {
    RelaxedPrecision,
    SpecId,
    Block,
    BufferBlock,
    RowMajor,
    ColMajor,
    ArrayStride,
    MatrixStride,
    GLSLShared,
    GLSLPacked,
    CPacked,
    BuiltIn,
    NoPerspective = 13,
    Flat,
    Patch,
    Centroid,
    Sample,
    Invariant,
    Restrict,
    Aliased,
    Volatile,
    Constant,
    Coherent,
    NonWritable,
    NonReadable,
    Uniform,
    UniformId,
    SaturatedConversion,
    Stream,
    Location,
    Component,
    Index,
    Binding,
    DescriptorSet,
    Offset,
    XfbBuffer,
    XfbStride,
    FuncParamAttr,
    FPRoundingMode,
    FPFastMathMode,
    LinkageAttributes,
    NoContraction,
    InputAttachmentIndex,
    Alignment,
    MaxByteoffset,
    AlignmentId,
    MaxByteOffsetId,
    NoSignedWrap = 4469,
    NoUnsignedWrap
};

enum class eSPIRVImageFormat {
    Unknown = 0,
    Rgba32f,
    Rgba16f,
    R32f,
    Rgba8,
    Rgba8Snorm,
    Rg32f,
    Rg16f,
    R11fG11fB10f,
    R16f,
    Rgba16,
    Rgb10A2,
    Rg16,
    Rg8,
    R16,
    R8,
    Rgba16Snorm,
    Rg16Snorm,
    Rg8Snorm,
    R16Snorm,
    R8Snorm,
    Rgba32i,
    Rgba16i,
    Rgba8i,
    R32i,
    Rg32i,
    Rg16i,
    Rg8i,
    R16i,
    R8i,
    Rgba32ui,
    Rgba16ui,
    Rgba8ui,
    R32ui,
    Rgb10a2ui,
    Rg32ui,
    Rg16ui,
    Rg8ui,
    R16ui,
    R8ui,
    R64ui,
    R64i
};

union spirv_constant_t {
    uint32_t u32;
    float f32;
};

struct spirv_decoration_t {
    int descriptor_set = -1;
    int binding = -1;
};

struct spirv_buffer_props_t {
    int count = 1;
    bool runtime_array = false;
};

struct spirv_uniform_props_t {
    VkDescriptorType descr_type;
    eType type = eType::Undefined;
    eSPIRVDim dim;
    eSPIRVImageFormat format;
    int count = 1;
    bool runtime_array = false;
};

struct spirv_parser_state_t {
    eEndianness endianness = eEndianness::Invalid;
    spirv_header_t header = {};
    HashMap32<uint32_t, uint32_t> offsets;
    HashMap32<uint32_t, spirv_decoration_t> decorations;
};

const char *parse_debug_name(spirv_parser_state_t &ps, uint32_t id);
eType parse_numeric_type(spirv_parser_state_t &ps, uint32_t id);
spirv_constant_t parse_constant(spirv_parser_state_t &ps, uint32_t id);
uint32_t parse_type_size(spirv_parser_state_t &ps, uint32_t id);
spirv_buffer_props_t parse_buffer_props(spirv_parser_state_t &ps, uint32_t id);
spirv_uniform_props_t parse_uniform_props(spirv_parser_state_t &ps, uint32_t id);

} // namespace Vk
} // namespace Ray