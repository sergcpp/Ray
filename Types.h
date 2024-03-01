#pragma once

#include <cstdint>
#if !defined(DISABLE_OCL)
#include <string>
#include <vector>
#endif

#include "Bitmask.h"

/**
  @file Types.h
*/

namespace Ray {
/// Templated color struct
template <typename T, int N> struct color_t {
    T v[N];
};

/// 8-bit color types
using color_rgba8_t = color_t<uint8_t, 4>;
using color_rgb8_t = color_t<uint8_t, 3>;
using color_rg8_t = color_t<uint8_t, 2>;
using color_r8_t = color_t<uint8_t, 1>;

/// Floating-point color types
using color_rgba_t = color_t<float, 4>;
using color_rgb_t = color_t<float, 3>;
using color_rg_t = color_t<float, 2>;
using color_r_t = color_t<float, 1>;

/// Color data struct, references array of pixels
template <typename T, int N> struct color_data_t {
    const color_t<T, N> *ptr;   ///< Pixels array
    int pitch;                  ///< Data pitch (distance between lines) expressed in pixels
};

using color_data_rgba8_t = color_data_t<uint8_t, 4>;
using color_data_rgb8_t = color_data_t<uint8_t, 3>;
using color_data_rg8_t = color_data_t<uint8_t, 2>;
using color_data_r8_t = color_data_t<uint8_t, 1>;

using color_data_rgba_t = color_data_t<float, 4>;
using color_data_rgb_t = color_data_t<float, 3>;
using color_data_rg_t = color_data_t<float, 2>;
using color_data_r_t = color_data_t<float, 1>;

enum class eAUXBuffer : uint32_t { SHL1 = 0, BaseColor = 1, DepthNormals = 2 };

struct shl1_data_t {
    float coeff_r[4], coeff_g[4], coeff_b[4];
};
static_assert(sizeof(shl1_data_t) == 48, "!");

/// Rectangle struct
struct rect_t {
    int x, y, w, h;
};

/// Camera type
enum class eCamType : uint8_t { Persp, Ortho, Geo };

/// Type of reconstruction filter
enum class ePixelFilter : uint8_t { Box, Gaussian, BlackmanHarris, _Count };

enum class eLensUnits : uint8_t { FOV, FLength };

/// View transform (affects tonemapping)
enum class eViewTransform : uint8_t {
    Standard,
    AgX,
    AgX_Punchy,
    Filmic_VeryLowContrast,
    Filmic_LowContrast,
    Filmic_MediumLowContrast,
    Filmic_MediumContrast,
    Filmic_MediumHighContrast,
    Filmic_HighContrast,
    Filmic_VeryHighContrast,
    _Count
};

enum class ePassFlags : uint8_t {
    SkipDirectLight,
    SkipIndirectLight,
    LightingOnly,
    NoBackground,
    OutputSH
};

struct pass_settings_t {
    uint8_t max_diff_depth, max_spec_depth, max_refr_depth, max_transp_depth, max_total_depth;
    uint8_t min_total_depth, min_transp_depth;
    Bitmask<ePassFlags> flags;
    float clamp_direct = 0.0f, clamp_indirect = 0.0f;
    int min_samples = 128;
    float variance_threshold = 0.0f;
    float regularize_alpha = 0.03f;
};
static_assert(sizeof(pass_settings_t) == 28, "!");

struct camera_t {
    eCamType type;
    ePixelFilter filter;
    eViewTransform view_transform;
    eLensUnits ltype;
    float filter_width;
    float fov, exposure, gamma, sensor_height;
    float focus_distance, focal_length, fstop, lens_rotation, lens_ratio;
    int lens_blades;
    float clip_start, clip_end;
    float origin[3], fwd[3], side[3], up[3], shift[2];
    uint32_t mi_index, uv_index;
    pass_settings_t pass_settings;
};

struct gpu_device_t {
    char name[256];
};

struct unet_filter_properties_t {
    int pass_count = 0;
    int alias_dependencies[16][4] = {};
};
} // namespace Ray
