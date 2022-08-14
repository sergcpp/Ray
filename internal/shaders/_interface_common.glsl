
#ifndef INTERFACE_COMMON_GLSL
#define INTERFACE_COMMON_GLSL

#ifdef __cplusplus
#define VEC2_TYPE std::array<float, 2>
#define VEC3_TYPE std::array<float, 3>
#define VEC4_TYPE std::array<float, 4>

#define IVEC2_TYPE std::array<int32_t, 2>
#define IVEC3_TYPE std::array<int32_t, 3>
#define IVEC4_TYPE std::array<int32_t, 4>

#define UINT_TYPE uint32_t
#define UVEC2_TYPE std::array<uint32_t, 2>
#define UVEC3_TYPE std::array<uint32_t, 3>
#define UVEC4_TYPE std::array<uint32_t, 4>

#define MAT2_TYPE std::array<std::array<float, 2>, 2>
#define MAT3_TYPE std::array<std::array<float, 3>, 3>
#define MAT4_TYPE std::array<std::array<float, 4>, 4>

#define INTERFACE_START(name) namespace name {
#define INTERFACE_END }

#define DEF_CONST_INT(name, index) const int name = index;
#else // __cplusplus
#define VEC2_TYPE vec2
#define VEC3_TYPE vec3
#define VEC4_TYPE vec4

#define IVEC2_TYPE ivec2
#define IVEC3_TYPE ivec3
#define IVEC4_TYPE ivec4

#define UINT_TYPE uint
#define UVEC2_TYPE uvec2
#define UVEC3_TYPE uvec3
#define UVEC4_TYPE uvec4

#define MAT2_TYPE mat2
#define MAT3_TYPE mat3
#define MAT4_TYPE mat4

#define INTERFACE_START(name)
#define INTERFACE_END

#if defined(VULKAN)
#define LAYOUT_PARAMS layout(push_constant)
#elif defined(GL_SPIRV)
#define LAYOUT_PARAMS layout(binding = REN_UB_UNIF_PARAM_LOC, std140)
#else
#define LAYOUT_PARAMS layout(std140)
#endif
#endif // __cplusplus

#endif // INTERFACE_COMMON_GLSL
