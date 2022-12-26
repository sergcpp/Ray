#pragma once

#include <cstdint>
#if !defined(DISABLE_OCL)
#include <string>
#include <vector>
#endif

/**
  @file Types.h
*/

namespace Ray {
/// RGBA single precision f32 color
struct pixel_color_t {
    float r, g, b, a;
};
static_assert(sizeof(pixel_color_t) == 16, "!");

template <typename T, int N> struct color_t { T v[N]; };

using color_rgba8_t = color_t<uint8_t, 4>;
using color_rgb8_t = color_t<uint8_t, 3>;
using color_rg8_t = color_t<uint8_t, 2>;
using color_r8_t = color_t<uint8_t, 1>;

using color_rgba_t = color_t<float, 4>;
using color_rgb_t = color_t<float, 3>;
using color_rg_t = color_t<float, 2>;
using color_r_t = color_t<float, 1>;

struct shl1_data_t {
    float coeff_r[4], coeff_g[4], coeff_b[4];
};
static_assert(sizeof(shl1_data_t) == 48, "!");

/// Rectangle struct
struct rect_t {
    int x, y, w, h;
};

enum eCamType { Persp, Ortho, Geo };

enum eFilterType { Box, Tent };

enum eDeviceType { None, SRGB };

enum eLensUnits { FOV, FLength };

enum ePassFlags {
    SkipDirectLight = (1 << 0),
    SkipIndirectLight = (1 << 1),
    LightingOnly = (1 << 2),
    NoBackground = (1 << 3),
    Clamp = (1 << 4),
    OutputSH = (1 << 5)
};

struct pass_settings_t {
    uint8_t max_diff_depth, max_spec_depth, max_refr_depth, max_transp_depth, max_total_depth;
    uint8_t termination_start_depth;
    uint8_t pad[2];
    uint32_t flags;
};

struct camera_t {
    eCamType type;
    eFilterType filter;
    eDeviceType dtype;
    eLensUnits ltype;
    float fov, gamma, sensor_height;
    float focus_distance, focal_length, fstop, lens_rotation, lens_ratio;
    int lens_blades;
    float clip_start, clip_end;
    float origin[3], fwd[3], side[3], up[3], shift[2];
    uint32_t mi_index, uv_index;
    pass_settings_t pass_settings;
};

#if !defined(DISABLE_OCL)
namespace Ocl {
struct Device {
    std::string name;
};
struct Platform {
    std::string vendor, name;
    std::vector<Device> devices;
};
} // namespace Ocl
#endif
} // namespace Ray
