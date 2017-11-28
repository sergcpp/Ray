#pragma once

namespace ray {
struct pixel_color_t {
    float r, g, b, a;
};
static_assert(sizeof(pixel_color_t) == 16, "!");

struct pixel_color8_t {
    uint8_t r, g, b, a;
};
static_assert(sizeof(pixel_color8_t) == 4, "!");

enum eCamType { Persp, Ortho };

struct camera_t {
    eCamType type;
    int pad[2];
    float origin[3], fwd[3], side[3], up[3];
};
static_assert(sizeof(camera_t) == 60, "!");
}