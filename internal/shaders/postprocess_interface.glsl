#ifndef POSTPROCESS_INTERFACE_GLSL
#define POSTPROCESS_INTERFACE_GLSL

#include "_interface_common.glsl"

INTERFACE_START(Postprocess)

struct Params {
    UVEC2_TYPE img_size;
    int srgb, _clamp;
};

const int LOCAL_GROUP_SIZE_X = 8;
const int LOCAL_GROUP_SIZE_Y = 8;

const int IN_IMG_SLOT = 1;

const int OUT_IMG_SLOT = 0;

INTERFACE_END

#endif // POSTPROCESS_INTERFACE_GLSL