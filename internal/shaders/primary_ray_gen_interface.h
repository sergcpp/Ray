#ifndef PRIMARY_RAY_GEN_INTERFACE_H
#define PRIMARY_RAY_GEN_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(PrimaryRayGen)

struct Params {
    uvec4 rect;
    vec4 cam_origin;   // w is fov factor
    vec4 cam_fwd;      // w is clip start
    vec4 cam_side;     // w is focus distance
    vec4 cam_up;       // w is sensor height
    uvec2 img_size;
    uint rand_seed;
    float spread_angle;
    //
    float cam_fstop;
    float cam_focal_length;
    float cam_lens_rotation;
    float cam_lens_ratio;
    //
    int cam_filter_and_lens_blades;
    float shift_x;
    float shift_y;
    int iteration;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int RANDOM_SEQ_BUF_SLOT = 2;
const int FILTER_TABLE_BUF_SLOT = 3;
const int REQUIRED_SAMPLES_IMG_SLOT = 4;

const int OUT_RAYS_BUF_SLOT = 0;
const int INOUT_COUNTERS_BUF_SLOT = 1;

INTERFACE_END

#endif // PRIMARY_RAY_GEN_INTERFACE_H