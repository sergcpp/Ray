#pragma once

#include "CoreRef.h"

namespace Ray {
namespace Ref {
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
force_inline fvec4 vectorcall reversible_tonemap(const fvec4 c) {
    return c / (fmaxf(c.get<0>(), fmaxf(c.get<1>(), c.get<2>())) + 1.0f);
}

force_inline fvec4 vectorcall reversible_tonemap_invert(const fvec4 c) {
    return c / (1.0f - fmaxf(c.get<0>(), fmaxf(c.get<1>(), c.get<2>())));
}

struct tonemap_params_t {
    eViewTransform view_transform;
    float inv_gamma;
};

force_inline fvec4 vectorcall TonemapStandard(fvec4 c) {
    UNROLLED_FOR(i, 3, {
        if (c.get<i>() < 0.0031308f) {
            c.set<i>(12.92f * c.get<i>());
        } else {
            c.set<i>(1.055f * powf(c.get<i>(), (1.0f / 2.4f)) - 0.055f);
        }
    })
    return c;
}

fvec4 vectorcall TonemapFilmic(eViewTransform view_transform, fvec4 color);

force_inline fvec4 vectorcall Tonemap(const tonemap_params_t &params, fvec4 c) {
    if (params.view_transform == eViewTransform::Standard) {
        c = TonemapStandard(c);
    } else {
        c = TonemapFilmic(params.view_transform, c);
    }

    if (params.inv_gamma != 1.0f) {
        c = pow(c, fvec4{params.inv_gamma, params.inv_gamma, params.inv_gamma, 1.0f});
    }

    return saturate(c);
}
}
}