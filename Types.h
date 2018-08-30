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

/// RGBA u8 color
struct pixel_color8_t {
    uint8_t r, g, b, a;
};
static_assert(sizeof(pixel_color8_t) == 4, "!");

/// Rectangle struct
struct rect_t { int x, y, w, h; };

enum eCamType { Persp, Ortho, Geo };

enum eFilterType { Box, Tent };

enum ePassFlags { SkipDirectLight = 1, SkipIndirectLight = 2, LightingOnly = 4, NoBackground = 8 };

struct camera_t {
    eCamType type;
    eFilterType filter;
    float fov, gamma;
    float focus_distance, focus_factor;
    float origin[3],
          fwd[3],
          side[3],
          up[3];
    uint32_t mi_index, uv_index;
    uint32_t pass_flags;
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
}
#endif
}