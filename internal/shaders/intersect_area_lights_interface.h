#ifndef INTERSECT_AREA_LIGHTS_INTERFACE_H
#define INTERSECT_AREA_LIGHTS_INTERFACE_H

#include "_interface_common.h"

INTERFACE_START(IntersectAreaLights)

struct Params {
    UVEC2_TYPE img_size;
    UINT_TYPE visible_lights_count;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int LIGHTS_BUF_SLOT = 1;
const int VISIBLE_LIGHTS_BUF_SLOT = 2;
const int TRANSFORMS_BUF_SLOT = 3;
const int RAYS_BUF_SLOT = 4;
const int COUNTERS_BUF_SLOT = 5;

const int INOUT_HITS_BUF_SLOT = 0;

INTERFACE_END

#endif // INTERSECT_AREA_LIGHTS_INTERFACE_H