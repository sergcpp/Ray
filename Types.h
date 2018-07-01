#pragma once

#include <cstdint>
#if !defined(DISABLE_OCL)
#include <string>
#include <vector>
#endif

/**
  @file Types.h
*/

namespace ray {
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

enum eCamType { Persp, Ortho };

/** Camera struct.
    Should not be used directly, usually returned from SceneBase::GetCamera
*/
struct camera_t {
    eCamType type;      ///< Projection type
    int pad[2];
    float origin[3],    ///< Origin point
          fwd[3],       ///< Forward unit vector
          side[3],      ///< Right side unit vector
          up[3];        ///< Up vector
};
static_assert(sizeof(camera_t) == 60, "!");

#if !defined(DISABLE_OCL)
namespace ocl {
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