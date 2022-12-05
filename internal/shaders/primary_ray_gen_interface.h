#ifndef PRIMARY_RAY_GEN_INTERFACE_H
#define PRIMARY_RAY_GEN_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(PrimaryRayGen)

struct Params {
    UVEC2_TYPE img_size;
    int hi;
    float spread_angle;
    VEC4_TYPE cam_origin;
    VEC4_TYPE cam_fwd;
    VEC4_TYPE cam_side;
    VEC4_TYPE cam_up;
    float cam_fstop;
    float cam_focal_length;
    float cam_lens_rotation;
    float cam_lens_ratio;
    int cam_lens_blades;
    float cam_clip_start;
    float _pad[3];
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int HALTON_SEQ_BUF_SLOT = 1;

const int OUT_RAYS_BUF_SLOT = 0;

INTERFACE_END

#endif // PRIMARY_RAY_GEN_INTERFACE_H