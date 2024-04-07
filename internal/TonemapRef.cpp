#include "TonemapRef.h"

namespace Ray {
extern const int LUT_DIMS = 48;
#include "precomputed/__agx.inl"
#include "precomputed/__agx_punchy.inl"
#include "precomputed/__filmic_high_contrast.inl"
#include "precomputed/__filmic_low_contrast.inl"
#include "precomputed/__filmic_med_contrast.inl"
#include "precomputed/__filmic_med_high_contrast.inl"
#include "precomputed/__filmic_med_low_contrast.inl"
#include "precomputed/__filmic_very_high_contrast.inl"
#include "precomputed/__filmic_very_low_contrast.inl"

const uint32_t *transform_luts[] = {
    nullptr,                    // Standard
    __agx,                      // AgX
    __agx_punchy,               // AgX_Punchy
    __filmic_very_low_contrast, // Filmic_VeryLowContrast
    __filmic_low_contrast,      // Filmic_LowContrast
    __filmic_med_low_contrast,  // Filmic_MediumLowContrast
    __filmic_med_contrast,      // Filmic_MediumContrast
    __filmic_med_high_contrast, // Filmic_MediumHighContrast
    __filmic_high_contrast,     // Filmic_HighContrast
    __filmic_very_high_contrast // Filmic_VeryHighContrast
};
static_assert(sizeof(transform_luts) / sizeof(transform_luts[0]) == int(eViewTransform::_Count), "!");

namespace Ref {
force_inline fvec4 FetchLUT(const eViewTransform view_transform, const int ix, const int iy, const int iz) {
    const uint32_t packed_val = transform_luts[int(view_transform)][iz * LUT_DIMS * LUT_DIMS + iy * LUT_DIMS + ix];
    const ivec4 ret = ivec4{int((packed_val >> 0) & 0x3ff), int((packed_val >> 10) & 0x3ff),
                            int((packed_val >> 20) & 0x3ff), int((packed_val >> 30) & 0x3)};
    return fvec4(ret) * fvec4{1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 1023.0f, 1.0f / 3.0f};
}
} // namespace Ref
} // namespace Ray

Ray::Ref::fvec4 vectorcall Ray::Ref::TonemapFilmic(const eViewTransform view_transform, fvec4 color) {
    const fvec4 encoded = color / (color + 1.0f);
    const fvec4 uv = encoded * float(LUT_DIMS - 1);
    const ivec4 xyz = ivec4(uv);
    const fvec4 f = fract(uv);
    const ivec4 xyz_next = min(xyz + 1, ivec4{LUT_DIMS - 1});

    const int ix = xyz.get<0>(), iy = xyz.get<1>(), iz = xyz.get<2>();
    const int jx = xyz_next.get<0>(), jy = xyz_next.get<1>(), jz = xyz_next.get<2>();
    const float fx = f.get<0>(), fy = f.get<1>(), fz = f.get<2>();

    const fvec4 c000 = FetchLUT(view_transform, ix, iy, iz), c001 = FetchLUT(view_transform, jx, iy, iz),
                c010 = FetchLUT(view_transform, ix, jy, iz), c011 = FetchLUT(view_transform, jx, jy, iz),
                c100 = FetchLUT(view_transform, ix, iy, jz), c101 = FetchLUT(view_transform, jx, iy, jz),
                c110 = FetchLUT(view_transform, ix, jy, jz), c111 = FetchLUT(view_transform, jx, jy, jz);

    const fvec4 c00x = (1.0f - fx) * c000 + fx * c001, c01x = (1.0f - fx) * c010 + fx * c011,
                c10x = (1.0f - fx) * c100 + fx * c101, c11x = (1.0f - fx) * c110 + fx * c111;

    const fvec4 c0xx = (1.0f - fy) * c00x + fy * c01x, c1xx = (1.0f - fy) * c10x + fy * c11x;

    fvec4 cxxx = (1.0f - fz) * c0xx + fz * c1xx;
    cxxx.set<3>(color.get<3>());

    return cxxx;
}