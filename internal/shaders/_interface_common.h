
#ifndef INTERFACE_COMMON_H
#define INTERFACE_COMMON_H

#ifdef __cplusplus
using vec2 = float[2];
using vec3 = float[3];
using vec4 = float[4];

using ivec2 = int[2];
using ivec3 = int[3];
using ivec4 = int[4];

using uint = uint32_t;
using uvec2 = uint[2];
using uvec3 = uint[3];
using uvec4 = uint[4];

using mat2 = float[2][2];
using mat3 = float[3][3];
using mat4 = float[4][4];

#define INTERFACE_START(name) namespace name {
#define INTERFACE_END }

#else // __cplusplus
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

#endif // INTERFACE_COMMON_H
