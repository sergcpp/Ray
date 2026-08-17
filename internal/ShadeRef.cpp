#include "ShadeRef.h"

#include <utility>

#include "RadCacheRef.h"
#include "TextureStorageCPU.h"

#pragma warning(push)
#pragma warning(disable : 6326) // potential comparison of a constant with another constant

namespace Ray {
extern const float __sheen_ltc_lut[32][32][4];
extern const float ggx_E[][32];
extern const float ggx_Eavg[32];
extern const float ggx_E_glass[][16][16];
extern const float ggx_E_glass_inv[][16][16];
extern const float ggx_Eavg_glass[][16];
extern const float ggx_Eavg_glass_inv[][16];
} // namespace Ray

namespace Ray::Ref {
force_inline fvec2 calc_alpha(const float roughness, const float anisotropy, const float regularize_alpha) {
    const float roughness2 = sqr(roughness);
    const float aspect = sqrtf(1.0f - 0.9f * anisotropy);

    fvec2 alpha = {roughness2 / aspect, roughness2 * aspect};
    where(alpha < regularize_alpha, alpha) = clamp(2 * alpha, 0.25f * regularize_alpha, regularize_alpha);
    return alpha;
}

force_inline float pow5(const float v) { return (v * v) * (v * v) * v; }

force_inline float F0_from_IOR(const float ior) { return sqr((ior - 1.0f) / (ior + 1.0f)); }

force_inline float schlick_weight(const float u) {
    const float m = Ray::saturate(1.0f - u);
    return pow5(m);
}

force_inline fvec4 reflect(const fvec4 &I, const fvec4 &N, const float dot_N_I) { return I - 2 * dot_N_I * N; }

force_inline float mix(const float v1, const float v2, const float k) { return (1.0f - k) * v1 + k * v2; }

// __sheen_ltc_lut[roughness_index][cosNI_index] = {transformA, transformB, albedo, 0}
force_inline fvec4 sheen_ltc_lookup(const float cosNI, const float roughness) {
    const int res = 32;
    const float x = Ray::saturate(cosNI) * float(res - 1);
    const float y = Ray::saturate(roughness) * float(res - 1);

    const int x0 = std::min(int(x), res - 1), x1 = std::min(x0 + 1, res - 1);
    const int y0 = std::min(int(y), res - 1), y1 = std::min(y0 + 1, res - 1);
    const float tx = x - float(x0), ty = y - float(y0);

    const fvec4 c00 = fvec4{__sheen_ltc_lut[y0][x0], vector_aligned},
                c10 = fvec4{__sheen_ltc_lut[y0][x1], vector_aligned};
    const fvec4 c01 = fvec4{__sheen_ltc_lut[y1][x0], vector_aligned},
                c11 = fvec4{__sheen_ltc_lut[y1][x1], vector_aligned};

    const float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty, w11 = tx * ty;

    return w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11;
}

force_inline void sheen_local_frame(const fvec4 &N, const fvec4 &V, fvec4 &T, fvec4 &B) {
    const fvec4 proj = V - N * dot(N, V);
    const float len2 = dot(proj, proj);
    if (len2 > 1e-8f) {
        T = proj / sqrtf(len2);
    } else {
        T = fabsf(N.get<0>()) > fabsf(N.get<2>()) ? normalize(fvec4{-N.get<1>(), N.get<0>(), 0.0f, 0.0f})
                                                  : normalize(fvec4{0.0f, -N.get<2>(), N.get<1>(), 0.0f});
    }
    B = cross(N, T);
}

lobe_weights_t get_lobe_weights(const float base_color_lum, const float sheen_lum, const float spec_color_lum,
                                const float specular, const float metallic, const float transmission,
                                const float clearcoat) {
    lobe_weights_t weights;

    // taken from Cycles
    weights.diffuse = base_color_lum * (1.0f - metallic) * (1.0f - transmission);
    const float final_transmission = transmission * (1.0f - metallic);
    weights.specular = (specular != 0.0f || metallic != 0.0f) ? spec_color_lum * (1.0f - final_transmission) : 0.0f;
    weights.clearcoat = clearcoat * (1.0f - metallic);
    weights.refraction = final_transmission * base_color_lum;
    // sheen sits on top of the stack (like in Cycles), not attenuated by metallic/transmission
    weights.sheen = sheen_lum;

    const float total_weight =
        weights.diffuse + weights.sheen + weights.specular + weights.clearcoat + weights.refraction;
    if (total_weight != 0.0f) {
        weights.diffuse /= total_weight;
        weights.sheen /= total_weight;
        weights.specular /= total_weight;
        weights.clearcoat /= total_weight;
        weights.refraction /= total_weight;
    }

    return weights;
}

float fresnel_dielectric_cos(const float cosi, const float eta) {
    // compute fresnel reflectance without explicitly computing the refracted direction
    const float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;

    float result = 1.0f; // default to TIR (no refracted component)
    if (g > 0) {
        g = sqrtf(g);
        const float A = (g - c) / (g + c);
        const float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        result = 0.5f * A * A * (1 + B * B);
    }

    return result;
}

// F82-tint Fresnel model used by Blender/Cycles for metallic Principled BSDF
force_inline fvec4 fresnel_f82tint_B(const fvec4 &f0, const fvec4 &tint) {
    const float f = 6.0f / 7.0f;
    const float f5 = sqr(sqr(f)) * f;
    const fvec4 f_schlick = mix(f0, fvec4{1.0f}, f5);
    return f_schlick * (7.0f / (f5 * f)) * (fvec4{1.0f} - tint);
}

force_inline fvec4 fresnel_f82(const float cosi, const fvec4 &f0, const fvec4 &b) {
    const float s = saturate(1.0f - cosi);
    const float s5 = sqr(sqr(s)) * s;
    const fvec4 f_schlick = mix(f0, fvec4{1.0f}, s5);
    return saturate(f_schlick - b * (cosi * s5 * s));
}

float oren_nayar_G(const float cos_theta) {
    if (cos_theta < 1e-6f) {
        return (0.5f * PI - 2.0f / 3.0f) - cos_theta;
    }
    const float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
    const float theta = portable_acosf(clamp(cos_theta, -1.0f, 1.0f));
    return sin_theta * (theta - 2.0f / 3.0f - sin_theta * cos_theta) +
           2.0f / 3.0f * (sin_theta / cos_theta) * (1.0f - sqr(sin_theta) * sin_theta);
}

fvec3 SampleGGX_NDF(const float rgh, const float r1, const float r2) {
    const float a = fmaxf(0.001f, rgh);

    const float phi = r1 * (2.0f * PI);

    const float cos_theta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
    const float sin_theta = Ray::saturate(sqrtf(1.0f - (cos_theta * cos_theta)));
    const fvec2 sincos_phi = portable_sincos(phi);
    const float sin_phi = sincos_phi.get<0>(), cos_phi = sincos_phi.get<1>();

    return fvec3{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
}

// http://jcgt.org/published/0007/04/01/paper.pdf
fvec4 SampleVNDF_Hemisphere_CrossSect(const fvec4 &Vh, float U1, float U2) {
    // orthonormal basis (with special case if cross product is zero)
    const float lensq = sqr(Vh.get<0>()) + sqr(Vh.get<1>());
    const fvec4 T1 =
        lensq > 0.0f ? fvec4(-Vh.get<1>(), Vh.get<0>(), 0.0f, 0.0f) / sqrtf(lensq) : fvec4(1.0f, 0.0f, 0.0f, 0.0f);
    const fvec4 T2 = cross(Vh, T1);
    // parameterization of the projected area
    const float r = sqrtf(U1);
    const float phi = 2.0f * PI * U2;
    const fvec2 sincos_phi = portable_sincos(phi);
    const float t1 = r * sincos_phi.get<1>();
    float t2 = r * sincos_phi.get<0>();
    const float s = 0.5f * (1.0f + Vh.get<2>());
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    // reprojection onto hemisphere
    const fvec4 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    // normalization will be done later
    return Nh;
}

// https://arxiv.org/pdf/2306.05044.pdf
fvec4 SampleVNDF_Hemisphere_SphCap(const fvec4 &Vh, const fvec2 alpha, const fvec2 rand) {
    // sample a spherical cap in (-Vh.z, 1]
    const float phi = 2.0f * PI * rand.get<0>();
    const float z = fma(1.0f - rand.get<1>(), 1.0f + Vh.get<2>(), -Vh.get<2>());
    const float sin_theta = sqrtf(Ray::saturate(1.0f - z * z));
    const fvec2 sincos_phi = portable_sincos(phi);
    const float x = sin_theta * sincos_phi.get<1>();
    const float y = sin_theta * sincos_phi.get<0>();
    const fvec4 c = fvec4{x, y, z, 0.0f};
    // normalization will be done later
    return c + Vh;
}

// https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
fvec4 SampleVNDF_Hemisphere_SphCap_Bounded(const fvec4 &Ve, const fvec4 &Vh, const fvec2 alpha, const fvec2 rand) {
    // sample a spherical cap in (-Vh.z, 1]
    const float phi = 2.0f * PI * rand.get<0>();
    const float a = Ray::saturate(fminf(alpha.get<0>(), alpha.get<1>()));
    const float s = 1.0f + length(fvec2{Ve.get<0>(), Ve.get<1>()});
    const float a2 = a * a, s2 = s * s;
    const float k = (1.0f - a2) * s2 / (s2 + a2 * Ve.get<2>() * Ve.get<2>());
    const float b = (Ve.get<2>() > 0.0f) ? k * Vh.get<2>() : Vh.get<2>();
    const float z = fma(1.0f - rand.get<1>(), 1.0f + b, -b);
    const float sin_theta = sqrtf(Ray::saturate(1.0f - z * z));
    const fvec2 sincos_phi = portable_sincos(phi);
    const float x = sin_theta * sincos_phi.get<1>();
    const float y = sin_theta * sincos_phi.get<0>();
    const fvec4 c = fvec4{x, y, z, 0.0f};
    // normalization will be done later
    return c + Vh;
}

// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
fvec4 SampleGGX_VNDF(const fvec4 &Ve, fvec2 alpha, fvec2 rand) {
    // transforming the view direction to the hemisphere configuration
    const fvec4 Vh = normalize(fvec4(alpha.get<0>() * Ve.get<0>(), alpha.get<1>() * Ve.get<1>(), Ve.get<2>(), 0.0f));
    // sample the hemisphere
    const fvec4 Nh = SampleVNDF_Hemisphere_SphCap(Vh, alpha, rand);
    // transforming the normal back to the ellipsoid configuration
    const fvec4 Ne =
        normalize(fvec4(alpha.get<0>() * Nh.get<0>(), alpha.get<1>() * Nh.get<1>(), fmaxf(0.0f, Nh.get<2>()), 0.0f));
    return Ne;
}

fvec4 SampleGGX_VNDF_Bounded(const fvec4 &Ve, fvec2 alpha, fvec2 rand) {
    // transforming the view direction to the hemisphere configuration
    const fvec4 Vh = normalize(fvec4(alpha.get<0>() * Ve.get<0>(), alpha.get<1>() * Ve.get<1>(), Ve.get<2>(), 0.0f));
    // sample the hemisphere
    const fvec4 Nh = SampleVNDF_Hemisphere_SphCap_Bounded(Ve, Vh, alpha, rand);
    // transforming the normal back to the ellipsoid configuration
    const fvec4 Ne =
        normalize(fvec4(alpha.get<0>() * Nh.get<0>(), alpha.get<1>() * Nh.get<1>(), fmaxf(0.0f, Nh.get<2>()), 0.0f));
    return Ne;
}

float GGX_VNDF_Reflection_Bounded_PDF(const float D, const fvec4 &view_dir_ts, const fvec2 alpha) {
    const fvec2 ai = alpha * fvec2{view_dir_ts.get<0>(), view_dir_ts.get<1>()};
    const float len2 = dot(ai, ai);
    const float t = sqrtf(len2 + view_dir_ts.get<2>() * view_dir_ts.get<2>());
    if (view_dir_ts.get<2>() >= 0.0f) {
        const float a = Ray::saturate(fminf(alpha.get<0>(), alpha.get<1>()));
        const float s = 1.0f + length(fvec2{view_dir_ts.get<0>(), view_dir_ts.get<1>()});
        const float a2 = a * a, s2 = s * s;
        const float k = (1.0f - a2) * s2 / (s2 + a2 * view_dir_ts.get<2>() * view_dir_ts.get<2>());
        return D / (2.0f * (k * view_dir_ts.get<2>() + t));
    }
    return D * (t - view_dir_ts.get<2>()) / (2.0f * len2);
}

// Smith shadowing function
force_inline float G1_lambda(const fvec4 &Ve, fvec2 alpha2) {
    return (-1.0f + sqrtf(1.0f + safe_div_pos(alpha2.get<0>() * sqr(Ve.get<0>()) + alpha2.get<1>() * sqr(Ve.get<1>()),
                                              sqr(Ve.get<2>())))) /
           2.0f;
}

force_inline float G1(const fvec4 &Ve, fvec2 alpha) {
    const float lambda = G1_lambda(Ve, alpha * alpha);
    return 1.0f / (1.0f + lambda);
}

float D_GTR2(const float N_dot_H, const float a) {
    const float a2 = sqr(a);
    const float t = 1.0f + (a2 - 1.0f) * N_dot_H * N_dot_H;
    return a2 / (PI * t * t);
}

float D_GGX(const fvec4 &H, const fvec2 alpha) {
    if (H.get<2>() == 0.0f) {
        return 0.0f;
    }
    const float sx = -H.get<0>() / (H.get<2>() * alpha.get<0>());
    const float sy = -H.get<1>() / (H.get<2>() * alpha.get<1>());
    const float s1 = 1.0f + sx * sx + sy * sy;
    const float cos_theta_h4 = sqr(sqr(H.get<2>()));
    return 1.0f / (sqr(s1) * PI * alpha.get<0>() * alpha.get<1>() * cos_theta_h4);
}

force_inline float safe_sqrtf(float f) { return sqrtf(fmaxf(f, 0.0f)); }

// Taken from Cycles
fvec4 ensure_valid_reflection(const fvec4 &Ng, const fvec4 &I, const fvec4 &N) {
    const fvec4 R = 2 * dot(N, I) * N - I;

    // Reflection rays may always be at least as shallow as the incoming ray.
    const float threshold = fminf(0.9f * dot(Ng, I), 0.01f);
    if (dot(Ng, R) >= threshold) {
        return N;
    }

    // Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
    // The X axis is found by normalizing the component of N that's orthogonal to Ng.
    // The Y axis isn't actually needed.
    const float NdotNg = dot(N, Ng);
    const fvec4 X = normalize(N - NdotNg * Ng);

    // Calculate N.z and N.x in the local coordinate system.
    //
    // The goal of this computation is to find a N' that is rotated towards Ng just enough
    // to lift R' above the threshold (here called t), therefore dot(R', Ng) = t.
    //
    // According to the standard reflection equation,
    // this means that we want dot(2*dot(N', I)*N' - I, Ng) = t.
    //
    // Since the Z axis of our local coordinate system is Ng, dot(x, Ng) is just x.z, so we get
    // 2*dot(N', I)*N'.z - I.z = t.
    //
    // The rotation is simple to express in the coordinate system we formed -
    // since N lies in the X-Z-plane, we know that N' will also lie in the X-Z-plane,
    // so N'.y = 0 and therefore dot(N', I) = N'.x*I.x + N'.z*I.z .
    //
    // Furthermore, we want N' to be normalized, so N'.x = sqrt(1 - N'.z^2).
    //
    // With these simplifications,
    // we get the final equation 2*(sqrt(1 - N'.z^2)*I.x + N'.z*I.z)*N'.z - I.z = t.
    //
    // The only unknown here is N'.z, so we can solve for that.
    //
    // The equation has four solutions in general:
    //
    // N'.z = +-sqrt(0.5*(+-sqrt(I.x^2*(I.x^2 + I.z^2 - t^2)) + t*I.z + I.x^2 + I.z^2)/(I.x^2 + I.z^2))
    // We can simplify this expression a bit by grouping terms:
    //
    // a = I.x^2 + I.z^2
    // b = sqrt(I.x^2 * (a - t^2))
    // c = I.z*t + a
    // N'.z = +-sqrt(0.5*(+-b + c)/a)
    //
    // Two solutions can immediately be discarded because they're negative so N' would lie in the
    // lower hemisphere.

    const float Ix = dot(I, X), Iz = dot(I, Ng);
    const float Ix2 = (Ix * Ix), Iz2 = (Iz * Iz);
    const float a = Ix2 + Iz2;

    const float b = safe_sqrtf(Ix2 * (a - (threshold * threshold)));
    const float c = Iz * threshold + a;

    // Evaluate both solutions.
    // In many cases one can be immediately discarded (if N'.z would be imaginary or larger than
    // one), so check for that first. If no option is viable (might happen in extreme cases like N
    // being in the wrong hemisphere), give up and return Ng.
    const float fac = 0.5f / a;
    const float N1_z2 = fac * (b + c), N2_z2 = fac * (-b + c);
    bool valid1 = (N1_z2 > 1e-5f) && (N1_z2 <= (1.0f + 1e-5f));
    bool valid2 = (N2_z2 > 1e-5f) && (N2_z2 <= (1.0f + 1e-5f));

    fvec2 N_new;
    if (valid1 && valid2) {
        // If both are possible, do the expensive reflection-based check.
        const fvec2 N1 = fvec2(safe_sqrtf(1.0f - N1_z2), safe_sqrtf(N1_z2));
        const fvec2 N2 = fvec2(safe_sqrtf(1.0f - N2_z2), safe_sqrtf(N2_z2));

        const float R1 = 2 * (N1.get<0>() * Ix + N1.get<1>() * Iz) * N1.get<1>() - Iz;
        const float R2 = 2 * (N2.get<0>() * Ix + N2.get<1>() * Iz) * N2.get<1>() - Iz;

        valid1 = (R1 >= 1e-5f);
        valid2 = (R2 >= 1e-5f);
        if (valid1 && valid2) {
            // If both solutions are valid, return the one with the shallower reflection since it will be
            // closer to the input (if the original reflection wasn't shallow, we would not be in this
            // part of the function).
            N_new = (R1 < R2) ? N1 : N2;
        } else {
            // If only one reflection is valid (= positive), pick that one.
            N_new = (R1 > R2) ? N1 : N2;
        }
    } else if (valid1 || valid2) {
        // Only one solution passes the N'.z criterium, so pick that one.
        const float Nz2 = valid1 ? N1_z2 : N2_z2;
        N_new = fvec2(safe_sqrtf(1.0f - Nz2), safe_sqrtf(Nz2));
    } else {
        return Ng;
    }

    return N_new.get<0>() * X + N_new.get<1>() * Ng;
}

fvec4 rotate_around_axis(const fvec4 &p, const fvec4 &axis, const float angle) {
    const fvec2 sincos_theta = portable_sincos(angle);
    const float costheta = sincos_theta.get<1>();
    const float sintheta = sincos_theta.get<0>();
    fvec4 r;

    r.set<0>(((costheta + (1.0f - costheta) * axis.get<0>() * axis.get<0>()) * p.get<0>()) +
             (((1.0f - costheta) * axis.get<0>() * axis.get<1>() - axis.get<2>() * sintheta) * p.get<1>()) +
             (((1.0f - costheta) * axis.get<0>() * axis.get<2>() + axis.get<1>() * sintheta) * p.get<2>()));
    r.set<1>((((1.0f - costheta) * axis.get<0>() * axis.get<1>() + axis.get<2>() * sintheta) * p.get<0>()) +
             ((costheta + (1.0f - costheta) * axis.get<1>() * axis.get<1>()) * p.get<1>()) +
             (((1.0f - costheta) * axis.get<1>() * axis.get<2>() - axis.get<0>() * sintheta) * p.get<2>()));
    r.set<2>((((1.0f - costheta) * axis.get<0>() * axis.get<2>() - axis.get<1>() * sintheta) * p.get<0>()) +
             (((1.0f - costheta) * axis.get<1>() * axis.get<2>() + axis.get<0>() * sintheta) * p.get<1>()) +
             ((costheta + (1.0f - costheta) * axis.get<2>() * axis.get<2>()) * p.get<2>()));
    r.set<3>(0.0f);

    return r;
}

void push_ior_stack(float stack[4], const float val) {
    UNROLLED_FOR(i, 3, {
        if (stack[i] < 0.0f) {
            stack[i] = val;
            return;
        }
    })
    // replace the last value regardless of sign
    stack[3] = val;
}

float pop_ior_stack(float stack[4], const float default_value = 1.0f) {
    UNROLLED_FOR_R(i, 4, {
        if (stack[i] > 0.0f) {
            return std::exchange(stack[i], -1.0f);
        }
    })
    return default_value;
}

float peek_ior_stack(const float stack[4], bool skip_first, const float default_value = 1.0f) {
    UNROLLED_FOR_R(i, 4, {
        if (stack[i] > 0.0f && !std::exchange(skip_first, false)) {
            return stack[i];
        }
    })
    return default_value;
}

float spread_attenuation(const fvec4 &D, const fvec4 &light_fwd, float tan_half_spread, float spread_normalization);
} // namespace Ray::Ref

float Ray::Ref::BRDF_PrincipledDiffuse(const fvec4 &V, const fvec4 &N, const fvec4 &L, const fvec4 &H,
                                       const float roughness) {
    const float N_dot_L = dot(N, L);
    const float N_dot_V = dot(N, V);
    if (N_dot_L <= 0.0f /*|| N_dot_V <= 0.0f*/) {
        return 0.0f;
    }

    const float FL = schlick_weight(N_dot_L);
    const float FV = schlick_weight(N_dot_V);

    const float L_dot_H = dot(L, H);
    const float Fd90 = 0.5f + 2.0f * L_dot_H * L_dot_H * roughness;
    const float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

    return Fd;
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_OrenDiffuse_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L,
                                                    const float roughness, const fvec4 &base_color) {

    const float sigma = roughness;
    const float a = 1.0f / (PI + sigma * (0.5f * PI - 2.0f / 3.0f));
    const float b = sigma * a;
    const float nl = fmaxf(dot(N, L), 0.0f);
    const float nv = fmaxf(dot(N, V), 0.0f);
    if (b <= 0.0f) {
        fvec4 ret = base_color * (nl / PI);
        ret.set<3>(nl / PI);
        return ret;
    }

    const float Eavg = a * PI + ((2.0f * PI - 5.6f) / 3.0f) * b;
    const fvec4 albedo = base_color;
    const fvec4 Ems = (1.0f / PI) * sqr(albedo) * (Eavg / (1.0f - Eavg)) / (1.0f - albedo * (1.0f - Eavg));
    const float Ev = a * PI + b * oren_nayar_G(nv);
    const fvec4 ms_term = Ems * (1.0f - Ev);

    ////

    float t = dot(L, V) - nl * nv;
    if (t > 0.0f) {
        t /= fmaxf(nl, nv) + FLT_MIN;
    }

    const float single_scatter = a + b * t;

    const float E1 = a * PI + b * oren_nayar_G(nl);
    const fvec4 multi_scatter = ms_term * (1.0f - E1);

    fvec4 diff_col = nl * (single_scatter + multi_scatter);

    fvec4 ret = base_color * diff_col;
    ret.set<3>(nl / PI);
    return ret;
}

Ray::Ref::fvec4 Ray::Ref::Sample_OrenDiffuse_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I,
                                                  const float roughness, const fvec4 &base_color, const fvec2 rand,
                                                  fvec4 &out_V) {
    const float phi = 2 * PI * rand.get<1>();
    const fvec2 sincos_phi = portable_sincos(phi);
    const float cos_phi = sincos_phi.get<1>(), sin_phi = sincos_phi.get<0>();

    const float z = sqrtf(1.0f - rand.get<0>());
    const float r = sqrtf(rand.get<0>());
    auto V = fvec4{r * cos_phi, r * sin_phi, z, 0.0f}; // in tangent-space

    out_V = world_from_tangent(T, B, N, V);
    return Evaluate_OrenDiffuse_BSDF(-I, N, out_V, roughness, base_color);
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_PrincipledDiffuse_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L,
                                                          const float roughness, const fvec4 &base_color,
                                                          const fvec4 &sheen_color, const bool uniform_sampling) {
    float weight, pdf;
    if (uniform_sampling) {
        weight = 2 * dot(N, L);
        pdf = 0.5f / PI;
    } else {
        weight = 1.0f;
        pdf = dot(N, L) / PI;
    }

    fvec4 H = normalize(L + V);
    if (dot(V, H) < 0.0f) {
        H = -H;
    }

    fvec4 diff_col = base_color * (weight * BRDF_PrincipledDiffuse(V, N, L, H, roughness));

    const float FH = PI * schlick_weight(dot(L, H));
    diff_col += FH * sheen_color;
    diff_col.set<3>(pdf);

    return diff_col;
}

Ray::Ref::fvec4 Ray::Ref::Sample_PrincipledDiffuse_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I,
                                                        const float roughness, const fvec4 &base_color,
                                                        const fvec4 &sheen_color, const bool uniform_sampling,
                                                        const fvec2 rand, fvec4 &out_V) {
    const float phi = 2 * PI * rand.get<1>();
    const fvec2 sincos_phi = portable_sincos(phi);
    const float cos_phi = sincos_phi.get<1>(), sin_phi = sincos_phi.get<0>();

    fvec4 V;
    if (uniform_sampling) {
        const float dir = sqrtf(1.0f - rand.get<0>() * rand.get<0>());
        V = fvec4{dir * cos_phi, dir * sin_phi, rand.get<0>(), 0.0f}; // in tangent-space
    } else {
        const float dir = sqrtf(rand.get<0>());
        const float k = sqrtf(1.0f - rand.get<0>());
        V = fvec4{dir * cos_phi, dir * sin_phi, k, 0.0f}; // in tangent-space
    }

    out_V = world_from_tangent(T, B, N, V);
    return Evaluate_PrincipledDiffuse_BSDF(-I, N, out_V, roughness, base_color, sheen_color, uniform_sampling);
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_PrincipledSheen_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L,
                                                        const float sheen_roughness) {
    fvec4 T, B;
    sheen_local_frame(N, V, T, B);

    const float cosNI = fmaxf(dot(N, V), 0.0f);
    const fvec4 ab_albedo = sheen_ltc_lookup(cosNI, sheen_roughness);
    if (fabsf(ab_albedo.get<0>()) < 1e-5f || ab_albedo.get<2>() < 1e-5f) {
        return fvec4{0.0f};
    }

    const fvec4 local_L = tangent_from_world(T, B, N, L);
    const float lenSqr = sqr(ab_albedo.get<0>() * local_L.get<0>() + ab_albedo.get<1>() * local_L.get<2>()) +
                         sqr(ab_albedo.get<0>() * local_L.get<1>()) + sqr(local_L.get<2>());
    const float val = (1.0f / PI) * fmaxf(local_L.get<2>(), 0.0f) * sqr(ab_albedo.get<0>() / lenSqr);

    return fvec4{val * ab_albedo.get<2>(), val * ab_albedo.get<2>(), val * ab_albedo.get<2>(), val};
}

Ray::Ref::fvec4 Ray::Ref::Sample_PrincipledSheen_BSDF(const fvec4 &N, const fvec4 &I, const float sheen_roughness,
                                                      const fvec2 rand, fvec4 &out_V) {
    const fvec4 V = -I;
    fvec4 T, B;
    sheen_local_frame(N, V, T, B);

    const float cosNI = fmaxf(dot(N, V), 0.0f);
    const fvec4 ab_albedo = sheen_ltc_lookup(cosNI, sheen_roughness);
    if (fabsf(ab_albedo.get<0>()) < 1e-5f || ab_albedo.get<2>() < 1e-5f) {
        out_V = fvec4{0.0f};
        return fvec4{0.0f};
    }

    const float phi = 2 * PI * rand.get<1>();
    const fvec2 sincos_phi = portable_sincos(phi);
    const float cos_phi = sincos_phi.get<1>(), sin_phi = sincos_phi.get<0>();
    const float r = sqrtf(rand.get<0>());
    const float diskZ = sqrtf(1.0f - rand.get<0>()); // cosine-weighted hemisphere sample, warped below

    const fvec4 local_L =
        normalize(fvec4{r * cos_phi - diskZ * ab_albedo.get<1>(), r * sin_phi, diskZ * ab_albedo.get<0>(), 0.0f});
    out_V = world_from_tangent(T, B, N, local_L);

    const float lenSqr = sqr(ab_albedo.get<0>() * local_L.get<0>() + ab_albedo.get<1>() * local_L.get<2>()) +
                         sqr(ab_albedo.get<0>() * local_L.get<1>()) + sqr(local_L.get<2>());
    const float val = (1.0f / PI) * local_L.get<2>() * sqr(ab_albedo.get<0>() / lenSqr);

    return fvec4{val * ab_albedo.get<2>(), val * ab_albedo.get<2>(), val * ab_albedo.get<2>(), val};
}

// Bilinear lookup into ggx_E[ndotv_row][rough_col] (see the LUT generator in RendererCPU.h)
force_inline float ggx_E_lookup(const float ndotv, const float rough) {
    const int res = 32;
    const float x = Ray::saturate(rough) * float(res - 1);
    const float y = Ray::saturate(ndotv) * float(res - 1);

    const int x0 = std::min(int(x), res - 1), x1 = std::min(x0 + 1, res - 1);
    const int y0 = std::min(int(y), res - 1), y1 = std::min(y0 + 1, res - 1);
    const float tx = x - float(x0), ty = y - float(y0);

    const float c00 = Ray::ggx_E[y0][x0], c10 = Ray::ggx_E[y0][x1];
    const float c01 = Ray::ggx_E[y1][x0], c11 = Ray::ggx_E[y1][x1];

    return (1.0f - tx) * (1.0f - ty) * c00 + tx * (1.0f - ty) * c10 + (1.0f - tx) * ty * c01 + tx * ty * c11;
}

// Linear lookup into ggx_Eavg[rough]
force_inline float ggx_Eavg_lookup(const float rough) {
    const int res = 32;
    const float x = Ray::saturate(rough) * float(res - 1);

    const int x0 = int(x), x1 = std::min(x0 + 1, res - 1);
    const float tx = x - float(x0);

    return (1.0f - tx) * Ray::ggx_Eavg[x0] + tx * Ray::ggx_Eavg[x1];
}

force_inline float ggx_E_glass_lookup(const float ndotv, const float rough, float ior) {
    const int res = 16;
    const auto *table = (ior > 1.0f) ? Ray::ggx_E_glass : Ray::ggx_E_glass_inv;
    if (ior < 1.0f) {
        ior = 1.0f / ior;
    }

    const float x = Ray::saturate(rough) * float(res - 1);
    const float y = Ray::saturate(ndotv) * float(res - 1);
    const float z = sqrtf(fabsf((ior - 1.0f) / (ior + 1.0f))) * float(res - 1);

    const int x0 = int(x), x1 = std::min(x0 + 1, res - 1);
    const int y0 = int(y), y1 = std::min(y0 + 1, res - 1);
    const int z0 = int(z), z1 = std::min(z0 + 1, res - 1);
    const float tx = x - float(x0), ty = y - float(y0), tz = z - float(z0);

    const float c000 = table[z0][y0][x0], c100 = table[z0][y0][x1];
    const float c010 = table[z0][y1][x0], c110 = table[z0][y1][x1];
    const float c001 = table[z1][y0][x0], c101 = table[z1][y0][x1];
    const float c011 = table[z1][y1][x0], c111 = table[z1][y1][x1];

    const float c00 = c000 + tx * (c100 - c000), c10 = c010 + tx * (c110 - c010);
    const float c01 = c001 + tx * (c101 - c001), c11 = c011 + tx * (c111 - c011);

    const float c0 = c00 + ty * (c10 - c00);
    const float c1 = c01 + ty * (c11 - c01);

    return c0 + tz * (c1 - c0);
}

force_inline float ggx_Eavg_glass_lookup(const float rough, float ior) {
    const int res = 16;
    const auto *table = (ior > 1.0f) ? Ray::ggx_Eavg_glass : Ray::ggx_Eavg_glass_inv;
    if (ior < 1.0f) {
        ior = 1.0f / ior;
    }

    const float x = Ray::saturate(rough) * float(res - 1);
    const float z = sqrtf(fabsf((ior - 1.0f) / (ior + 1.0f))) * float(res - 1);

    const int x0 = int(x), x1 = std::min(x0 + 1, res - 1);
    const int z0 = int(z), z1 = std::min(z0 + 1, res - 1);
    const float tx = x - float(x0), tz = z - float(z0);

    const float c0 = table[z0][x0] + tx * (table[z0][x1] - table[z0][x0]);
    const float c1 = table[z1][x0] + tx * (table[z1][x1] - table[z1][x0]);

    return c0 + tz * (c1 - c0);
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_GGXSpecular_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts,
                                                    const fvec4 &reflected_dir_ts, const fvec2 alpha,
                                                    const float spec_ior, const float spec_F0, const fvec4 &spec_col,
                                                    const fvec4 &spec_col_90, const float metallic,
                                                    const fvec4 &metallic_f0, const fvec4 &metallic_f82_b,
                                                    const bool preserve_energy) {
    const float D = D_GGX(sampled_normal_ts, alpha);

    const fvec2 alpha2 = alpha * alpha;
    const float G = 1.0f / (1.0f + G1_lambda(view_dir_ts, alpha2) + G1_lambda(reflected_dir_ts, alpha2));

    const float cos_theta_i = dot(view_dir_ts, sampled_normal_ts);

    const float FH =
        (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec_ior) - spec_F0) / (1.0f - spec_F0);
    const fvec4 dielectric_F = mix(spec_col, spec_col_90, FH);

    fvec4 F = dielectric_F;
    if (metallic > 0.0f) {
        const fvec4 metallic_F = fresnel_f82(cos_theta_i, metallic_f0, metallic_f82_b);
        F = mix(dielectric_F, metallic_F, metallic);
    }

    const float denom = 4.0f * fabsf(view_dir_ts.get<2>() * reflected_dir_ts.get<2>());
    F *= (denom != 0.0f) ? (D * G / denom) : 0.0f;
    F *= fmaxf(reflected_dir_ts.get<2>(), 0.0f);

    if (preserve_energy) {
        const float rough = sqrtf(sqrtf(alpha.get<0>() * alpha.get<1>()));
        const float E = ggx_E_lookup(fmaxf(view_dir_ts.get<2>(), 0.0f), rough);
        const float Eavg = ggx_Eavg_lookup(rough);
        const float missing_factor = (1.0f - E) / E;

        const fvec4 Fss = clamp(mix(spec_col, metallic_f0, metallic), fvec4{0.0f}, fvec4{0.999f});
        const fvec4 Fms = Fss * Eavg / (fvec4{1.0f} - Fss * (1.0f - Eavg));

        F *= 1.0f + Fms * missing_factor;
    }

    const float pdf = GGX_VNDF_Reflection_Bounded_PDF(D, view_dir_ts, alpha);
    F.set<3>(pdf);

    return F;
}

Ray::Ref::fvec4 Ray::Ref::Sample_GGXSpecular_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I,
                                                  const fvec2 alpha, const float spec_ior, const float spec_F0,
                                                  const fvec4 &spec_col, const fvec4 &spec_col_90, const fvec2 rand,
                                                  const float metallic, const fvec4 &metallic_f0,
                                                  const fvec4 &metallic_f82_b, const bool preserve_energy,
                                                  fvec4 &out_V) {
    if (alpha.get<0>() * alpha.get<1>() < 1e-7f) {
        const fvec4 V = reflect(I, N, dot(N, I));
        const float cos_theta_i = dot(V, N);
        const float FH = (fresnel_dielectric_cos(cos_theta_i, spec_ior) - spec_F0) / (1.0f - spec_F0);
        fvec4 F = mix(spec_col, spec_col_90, FH);
        if (metallic > 0.0f) {
            const fvec4 metallic_F = fresnel_f82(cos_theta_i, metallic_f0, metallic_f82_b);
            F = mix(F, metallic_F, metallic);
        }
        out_V = V;
        return fvec4{F.get<0>() * 1e6f, F.get<1>() * 1e6f, F.get<2>() * 1e6f, 1e6f};
    }

    const fvec4 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    const fvec4 sampled_normal_ts = SampleGGX_VNDF_Bounded(view_dir_ts, alpha, rand);

    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const fvec4 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts, dot_N_V));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);
    return Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha, spec_ior, spec_F0,
                                     spec_col, spec_col_90, metallic, metallic_f0, metallic_f82_b, preserve_energy);
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_GGXRefractionSpecular_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts,
                                                              const fvec4 &reflected_dir_ts, const fvec2 alpha,
                                                              const float spec_ior, const float spec_F0,
                                                              const fvec4 &spec_col, const fvec4 &spec_col_90,
                                                              const bool preserve_energy) {
    const float D = D_GGX(sampled_normal_ts, alpha);

    const fvec2 alpha2 = alpha * alpha;
    const float G = 1.0f / (1.0f + G1_lambda(view_dir_ts, alpha2) + G1_lambda(reflected_dir_ts, alpha2));

    const float cos_theta_i = dot(view_dir_ts, sampled_normal_ts);

    const float FH = (fresnel_dielectric_cos(cos_theta_i, spec_ior) - spec_F0) / (1.0f - spec_F0);
    fvec4 F = mix(spec_col, spec_col_90, FH);

    F *= fresnel_dielectric_cos(cos_theta_i, spec_ior);

    const float denom = 4.0f * fabsf(view_dir_ts.get<2>() * reflected_dir_ts.get<2>());
    F *= (denom != 0.0f) ? (D * G / denom) : 0.0f;
    F *= fmaxf(reflected_dir_ts.get<2>(), 0.0f);

    if (preserve_energy) {
        const float rough = sqrtf(sqrtf(alpha.get<0>() * alpha.get<1>()));
        const float E = ggx_E_glass_lookup(fmaxf(view_dir_ts.get<2>(), 0.0f), rough, spec_ior);
        const float Eavg = ggx_Eavg_glass_lookup(rough, spec_ior);
        const float missing_factor = (1.0f - E) / E;

        const fvec4 Fss = clamp(spec_col, fvec4{0.0f}, fvec4{0.999f});
        const fvec4 Fms = Fss * Eavg / (fvec4{1.0f} - Fss * (1.0f - Eavg));

        F *= 1.0f + Fms * missing_factor;
    }

    float pdf = safe_div_pos(D * fmaxf(cos_theta_i, 0.0f),
                             fabsf(view_dir_ts.get<2>()) * (1.0f + G1_lambda(view_dir_ts, alpha2)));
    const float div = 4.0f * cos_theta_i;
    if (div != 0.0f) {
        pdf /= div;
    }

    F.set<3>(pdf);

    return F;
}

Ray::Ref::fvec4 Ray::Ref::Sample_GGXRefractionSpecular_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N,
                                                            const fvec4 &I, const fvec2 alpha, const float spec_ior,
                                                            const float spec_F0, const fvec4 &spec_col,
                                                            const fvec4 &spec_col_90, const fvec2 rand,
                                                            const bool preserve_energy, fvec4 &out_V) {
    if (alpha.get<0>() * alpha.get<1>() < 1e-7f) {
        const fvec4 V = reflect(I, N, dot(N, I));
        const float cos_theta_i = dot(V, N);
        const float FH = (fresnel_dielectric_cos(cos_theta_i, spec_ior) - spec_F0) / (1.0f - spec_F0);
        fvec4 F = mix(spec_col, spec_col_90, FH);
        F *= fresnel_dielectric_cos(cos_theta_i, spec_ior);
        out_V = V;
        return fvec4{F.get<0>() * 1e6f, F.get<1>() * 1e6f, F.get<2>() * 1e6f, 1e6f};
    }

    const fvec4 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    const fvec4 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, alpha, rand);

    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const fvec4 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts, dot_N_V));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);
    return Evaluate_GGXRefractionSpecular_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, alpha, spec_ior,
                                               spec_F0, spec_col, spec_col_90, preserve_energy);
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_GGXRefraction_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts,
                                                      const fvec4 &refr_dir_ts, const fvec2 alpha, float eta,
                                                      const fvec4 &refr_col, const bool fresnel,
                                                      const bool preserve_energy) {
    const float Vz = view_dir_ts.get<2>(), Lz = refr_dir_ts.get<2>();
    if (Lz >= 0.0f || Vz <= 0.0f || alpha.get<0>() * alpha.get<1>() < 1e-7f) {
        return fvec4{0.0f};
    }

    const float VoH = dot(view_dir_ts, sampled_normal_ts);
    const float LoH = dot(refr_dir_ts, sampled_normal_ts);

    if (VoH <= 0.0f || LoH >= 0.0f) {
        return fvec4{0.0f};
    }

    const float D = D_GGX(sampled_normal_ts, alpha);

    const fvec2 alpha2 = alpha * alpha;
    const float G1o_lambda = G1_lambda(refr_dir_ts, alpha2), G1i_lambda = G1_lambda(view_dir_ts, alpha2);

    const float denom = LoH + eta * VoH;
    const float jacobian = safe_div_pos(-LoH, sqr(denom));

    const float F = fresnel ? fresnel_dielectric_cos(VoH, 1.0f / eta) : 0.0f;

    const float common = D * VoH * jacobian;
    const float bsdf = (1.0f - F) * common / (Vz * (1.0f + G1o_lambda + G1i_lambda));
    const float pdf = common / (Vz * (1.0f + G1i_lambda));

    fvec4 ret = bsdf * refr_col;
    if (preserve_energy) {
        const float rough = sqrtf(sqrtf(alpha.get<0>() * alpha.get<1>()));
        const float ior = 1.0f / eta;
        const float E = ggx_E_glass_lookup(fmaxf(Vz, 0.0f), rough, ior);
        const float Eavg = ggx_Eavg_glass_lookup(rough, ior);
        const float missing_factor = (1.0f - E) / E;

        const fvec4 Fss = clamp(refr_col, fvec4{0.0f}, fvec4{0.999f});
        const fvec4 Fms = Fss * Eavg / (fvec4{1.0f} - Fss * (1.0f - Eavg));

        ret *= 1.0f + Fms * missing_factor;
    }

    ret.set<3>(pdf);

    return ret;
}

Ray::Ref::fvec4 Ray::Ref::Sample_GGXRefraction_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I,
                                                    const fvec2 alpha, const float eta, const fvec4 &refr_col,
                                                    const fvec2 rand, const bool fresnel, const bool preserve_energy,
                                                    fvec4 &out_V) {
    if (alpha.get<0>() * alpha.get<1>() < 1e-7f) {
        const float cosi = -dot(I, N);
        const float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
        if (cost2 < 0) {
            return fvec4{0.0f};
        }
        const float m = eta * cosi - sqrtf(cost2);
        const fvec4 V = normalize(eta * I + m * N);
        const float F = fresnel ? fresnel_dielectric_cos(cosi, 1.0f / eta) : 0.0f;

        out_V = fvec4{V.get<0>(), V.get<1>(), V.get<2>(), m};
        return fvec4{refr_col.get<0>() * (1.0f - F) * 1e6f, refr_col.get<1>() * (1.0f - F) * 1e6f,
                     refr_col.get<2>() * (1.0f - F) * 1e6f, 1e6f};
    }

    const fvec4 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    const fvec4 sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, alpha, rand);

    const float cosi = dot(view_dir_ts, sampled_normal_ts);
    const float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (cost2 < 0) {
        return fvec4{0.0f};
    }
    const float m = eta * cosi - sqrtf(cost2);
    const fvec4 refr_dir_ts = normalize(-eta * view_dir_ts + m * sampled_normal_ts);

    const fvec4 F = Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, refr_dir_ts, alpha, eta, refr_col,
                                                fresnel, preserve_energy);

    const fvec4 V = world_from_tangent(T, B, N, refr_dir_ts);
    out_V = fvec4{V.get<0>(), V.get<1>(), V.get<2>(), m};
    return F;
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_PrincipledClearcoat_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts,
                                                            const fvec4 &reflected_dir_ts, const float coat_roughness2,
                                                            const float coat_ior, const float coat_F0) {
    const float D = D_GGX(sampled_normal_ts, coat_roughness2);
    const float alpha2 = sqr(coat_roughness2);
    const float G = 1.0f / (1.0f + G1_lambda(view_dir_ts, alpha2) + G1_lambda(reflected_dir_ts, alpha2));

    const float FH =
        (fresnel_dielectric_cos(dot(reflected_dir_ts, sampled_normal_ts), coat_ior) - coat_F0) / (1.0f - coat_F0);
    float F = mix(coat_F0, 1.0f, FH);

    const float denom = 4.0f * fabsf(view_dir_ts.get<2>()) * fabsf(reflected_dir_ts.get<2>());
    F *= (denom != 0.0f) ? D * G / denom : 0.0f;
    F *= fmaxf(reflected_dir_ts.get<2>(), 0.0f);

    const float pdf = GGX_VNDF_Reflection_Bounded_PDF(D, view_dir_ts, coat_roughness2);
    return fvec4{F, F, F, pdf};
}

Ray::Ref::fvec4 Ray::Ref::Sample_PrincipledClearcoat_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N,
                                                          const fvec4 &I, const float coat_roughness2,
                                                          const float coat_ior, const float coat_F0, const fvec2 rand,
                                                          fvec4 &out_V) {
    if (sqr(coat_roughness2) < 1e-7f) {
        const fvec4 V = reflect(I, N, dot(N, I));

        const float FH = (fresnel_dielectric_cos(dot(V, N), coat_ior) - coat_F0) / (1.0f - coat_F0);
        const float F = mix(coat_F0, 1.0f, FH);

        out_V = V;
        return fvec4{F * 1e6f, F * 1e6f, F * 1e6f, 1e6f};
    }

    const fvec4 view_dir_ts = normalize(tangent_from_world(T, B, N, -I));
    const fvec4 sampled_normal_ts = SampleGGX_VNDF_Bounded(view_dir_ts, coat_roughness2, rand);

    const float dot_N_V = -dot(sampled_normal_ts, view_dir_ts);
    const fvec4 reflected_dir_ts = normalize(reflect(-view_dir_ts, sampled_normal_ts, dot_N_V));

    out_V = world_from_tangent(T, B, N, reflected_dir_ts);

    return Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, reflected_dir_ts, coat_roughness2,
                                             coat_ior, coat_F0);
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_DiffuseNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                                               const fvec4 &base_color, const float roughness, const float mix_weight,
                                               const bool use_mis, shadow_ray_t &sh_r) {
    const fvec4 I = make_fvec3(ray.d);

    const fvec4 diff_col = Evaluate_OrenDiffuse_BSDF(-I, surf.N, ls.L, roughness, base_color);
    const float bsdf_pdf = diff_col.get<3>();

    float mis_weight = 1.0f;
    if (use_mis && ls.area > 0.0f) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }

    const fvec4 lcol = ls.col * diff_col * (mix_weight * mis_weight / ls.pdf);

    if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    memcpy(&sh_r.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
    UNROLLED_FOR(i, 3, { sh_r.c[i] = lcol[i]; })
    return fvec4{0.0f};
}

void Ray::Ref::Sample_DiffuseNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color,
                                  const float roughness, const fvec2 rand, const float mix_weight,
                                  ray_data_t &new_ray) {
    const fvec4 I = make_fvec3(ray.d);

    fvec4 V;
    const fvec4 F = Sample_OrenDiffuse_BSDF(surf.T, surf.B, surf.N, I, roughness, base_color, rand, V);

    new_ray.depth = pack_ray_type(RAY_TYPE_DIFFUSE);
    new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(1, 0, 0, 0);

    memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
    memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));
    UNROLLED_FOR(i, 3, { new_ray.c[i] = safe_div_pos(F.get<i>() * mix_weight, F.get<3>()); })
    new_ray.pdf = F.get<3>();
    new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT;
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_GlossyNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                                              const fvec4 &base_color, const float roughness,
                                              const float regularize_alpha, const float mix_weight, const bool use_mis,
                                              shadow_ray_t &sh_r) {
    const fvec4 I = make_fvec3(ray.d);
    const fvec4 H = normalize(ls.L - I);

    const fvec4 view_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, -I);
    const fvec4 light_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, ls.L);
    const fvec4 sampled_normal_ts = tangent_from_world(surf.T, surf.B, surf.N, H);

    const fvec2 alpha = calc_alpha(roughness, 0.0f, regularize_alpha);
    if (alpha.get<0>() * alpha.get<1>() < 1e-7f) {
        return fvec4{0.0f};
    }

    const fvec4 spec_col = Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, alpha, 1.0f, 0.0f,
                                                     base_color, base_color, 0.0f, base_color, base_color, true);
    const float bsdf_pdf = spec_col.get<3>();

    float mis_weight = 1.0f;
    if (use_mis && ls.area > 0.0f) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }
    const fvec4 lcol = ls.col * spec_col * (mix_weight * mis_weight / ls.pdf);

    if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    memcpy(&sh_r.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
    memcpy(&sh_r.c[0], value_ptr(lcol), 3 * sizeof(float));

    return fvec4{0.0f};
}

void Ray::Ref::Sample_GlossyNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color,
                                 const float roughness, const float regularize_alpha, const fvec2 rand,
                                 const float mix_weight, ray_data_t &new_ray) {
    const fvec4 I = make_fvec3(ray.d);
    const fvec2 alpha = calc_alpha(roughness, 0.0f, regularize_alpha);

    fvec4 V;
    const fvec4 F = Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, I, alpha, 1.0f, 0.0f, base_color, base_color, rand,
                                            0.0f, base_color, base_color, true, V);

    new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
    new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

    memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
    memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));

    UNROLLED_FOR(i, 3, { new_ray.c[i] = F.get<i>() * safe_div_pos(mix_weight, F.get<3>()); })
    new_ray.pdf = F.get<3>();
    new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * fminf(alpha.get<0>(), alpha.get<1>());
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_RefractiveNode(const light_sample_t &ls, const ray_data_t &ray,
                                                  const surface_t &surf, const fvec4 &base_color, const float roughness,
                                                  const float regularize_alpha, const float eta, const float mix_weight,
                                                  const bool use_mis, shadow_ray_t &sh_r) {
    const fvec4 I = make_fvec3(ray.d);

    const fvec4 H = normalize(ls.L - I * eta);
    const fvec4 view_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, -I);
    const fvec4 light_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, ls.L);
    const fvec4 sampled_normal_ts = tangent_from_world(surf.T, surf.B, surf.N, H);

    const fvec4 refr_col =
        Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts,
                                    calc_alpha(roughness, 0.0f, regularize_alpha), eta, base_color, false, false);
    const float bsdf_pdf = refr_col[3];

    float mis_weight = 1.0f;
    if (use_mis && ls.area > 0.0f) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }
    const fvec4 lcol = ls.col * refr_col * (mix_weight * mis_weight / ls.pdf);

    if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    memcpy(&sh_r.o[0], value_ptr(offset_ray(surf.P, -surf.plane_N)), 3 * sizeof(float));
    memcpy(&sh_r.c[0], value_ptr(lcol), 3 * sizeof(float));

    return fvec4{0.0f};
}

void Ray::Ref::Sample_RefractiveNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color,
                                     const float roughness, const float regularize_alpha, const bool is_backfacing,
                                     const float int_ior, const float ext_ior, const fvec2 rand, const float mix_weight,
                                     ray_data_t &new_ray) {
    const fvec4 I = make_fvec3(ray.d);
    const fvec2 alpha = calc_alpha(roughness, 0.0f, regularize_alpha);
    const float eta = is_backfacing ? (int_ior / ext_ior) : (ext_ior / int_ior);

    fvec4 V;
    const fvec4 F = Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, I, alpha, eta, base_color, rand, false, false, V);

    new_ray.depth = pack_ray_type(RAY_TYPE_REFR);
    new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 0, 1, 0);

    UNROLLED_FOR(i, 3, { new_ray.c[i] = F.get<i>() * safe_div_pos(mix_weight, F.get<3>()); })
    new_ray.pdf = F.get<3>();

    if (!is_backfacing) {
        // Entering the surface, push new value
        push_ior_stack(new_ray.ior, int_ior);
    } else {
        // Exiting the surface, pop the last ior value
        pop_ior_stack(new_ray.ior);
    }

    memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, -surf.plane_N)), 3 * sizeof(float));
    memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));
    new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * fminf(alpha.get<0>(), alpha.get<1>());
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_PrincipledNode(const light_sample_t &ls, const ray_data_t &ray,
                                                  const surface_t &surf, const lobe_weights_t &lobe_weights,
                                                  const diff_params_t &diff, const spec_params_t &spec,
                                                  const coat_params_t &coat, const transmission_params_t &trans,
                                                  const float metallic, const float transmission, const float N_dot_L,
                                                  const float mix_weight, const bool use_mis,
                                                  const float regularize_alpha, shadow_ray_t &sh_r) {
    const fvec4 I = make_fvec3(ray.d);

    fvec4 lcol = 0.0f;
    float bsdf_pdf = 0.0f;

    fvec4 H;
    if (N_dot_L > 0.0f) {
        H = normalize(ls.L - I);
    } else {
        H = normalize(ls.L - I * trans.eta);
    }

    const fvec4 view_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, -I);
    const fvec4 light_dir_ts = tangent_from_world(surf.T, surf.B, surf.N, ls.L);
    const fvec4 sampled_normal_ts = tangent_from_world(surf.T, surf.B, surf.N, H);

    if (lobe_weights.diffuse > 0.0f && N_dot_L > 0.0f && (ls.ray_flags & RAY_TYPE_DIFFUSE_BIT) != 0) {
        fvec4 diff_col = Evaluate_OrenDiffuse_BSDF(-I, surf.N, ls.L, diff.roughness, diff.base_color);
        bsdf_pdf += lobe_weights.diffuse * diff_col.get<3>();

        const float FH_sampled =
            (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec.fresnel_ior) - spec.F0) /
            (1.0f - spec.F0);
        const fvec4 spec_col_sampled = mix(spec.tmp_col, fvec4(1.0f), FH_sampled);
        const fvec4 diff_spec_atten_sampled = clamp(1.0f - spec_col_sampled, fvec4{0.0f}, fvec4{1.0f});

        diff_col *= (1.0f - metallic) * (1.0f - transmission) * diff_spec_atten_sampled;

        lcol += ls.col * diff_col / ls.pdf;
    }

    if (lobe_weights.sheen > 0.0f && N_dot_L > 0.0f && (ls.ray_flags & RAY_TYPE_DIFFUSE_BIT) != 0) {
        fvec4 sheen_col = Evaluate_PrincipledSheen_BSDF(-I, surf.N, ls.L, diff.sheen_roughness);
        bsdf_pdf += lobe_weights.sheen * sheen_col.get<3>();
        sheen_col *= diff.sheen_color;

        lcol += ls.col * sheen_col / ls.pdf;
    }

    const fvec2 spec_alpha = calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha);
    if (lobe_weights.specular > 0.0f && spec_alpha.get<0>() * spec_alpha.get<1>() >= 1e-7f && N_dot_L > 0.0f &&
        (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0) {
        const fvec4 spec_col = Evaluate_GGXSpecular_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, spec_alpha,
                                                         spec.fresnel_ior, spec.F0, spec.tmp_col, fvec4{1.0f},
                                                         spec.metallic, spec.metal_f0, spec.metal_f82_b, true);
        bsdf_pdf += lobe_weights.specular * spec_col.get<3>();

        lcol += ls.col * spec_col / ls.pdf;
    }

    const fvec2 coat_alpha = calc_alpha(coat.roughness, 0.0f, regularize_alpha);
    if (lobe_weights.clearcoat > 0.0f && coat_alpha.get<0>() * coat_alpha.get<1>() >= 1e-7f && N_dot_L > 0.0f &&
        (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0) {
        const fvec4 coat_col = Evaluate_PrincipledClearcoat_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts,
                                                                 coat_alpha.get<0>(), coat.ior, coat.F0);
        bsdf_pdf += lobe_weights.clearcoat * coat_col.get<3>();

        lcol += ls.col * coat_col * coat.weight / ls.pdf;
    }

    if (lobe_weights.refraction > 0.0f) {
        const float refr_spec_alpha = calc_alpha(spec.roughness, 0.0f, regularize_alpha).get<0>();
        const float trans_fresnel = fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), 1.0f / trans.eta);
        if (sqr(refr_spec_alpha) >= 1e-7f && N_dot_L > 0.0f && (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0) {
            const fvec4 spec_col = Evaluate_GGXRefractionSpecular_BSDF(
                view_dir_ts, sampled_normal_ts, light_dir_ts, refr_spec_alpha, (1.0f / trans.eta),
                F0_from_IOR(1.0f / trans.eta), fvec4{1.0f}, fvec4{1.0f}, true);
            bsdf_pdf += lobe_weights.refraction * trans_fresnel * spec_col.get<3>();

            lcol += ls.col * spec_col / ls.pdf;
        }
        if (sqr(refr_spec_alpha) >= 1e-7f && N_dot_L < 0.0f && (ls.ray_flags & RAY_TYPE_REFR_BIT) != 0) {
            const fvec4 refr_col =
                Evaluate_GGXRefraction_BSDF(view_dir_ts, sampled_normal_ts, light_dir_ts, refr_spec_alpha, trans.eta,
                                            sqrt(diff.base_color), true, true);
            bsdf_pdf += lobe_weights.refraction * (1.0f - trans_fresnel) * refr_col.get<3>();

            lcol += ls.col * refr_col / ls.pdf;
        }
    }

    float mis_weight = 1.0f;
    if (use_mis && ls.area > 0.0f) {
        mis_weight = power_heuristic(ls.pdf, bsdf_pdf);
    }
    lcol *= mix_weight * mis_weight;

    if (!ls.cast_shadow) {
        // apply light immediately
        return lcol;
    }

    // schedule shadow ray
    memcpy(&sh_r.o[0], value_ptr(offset_ray(surf.P, N_dot_L < 0.0f ? -surf.plane_N : surf.plane_N)), 3 * sizeof(float));
    memcpy(&sh_r.c[0], value_ptr(lcol), 3 * sizeof(float));

    return fvec4{0.0f};
}

void Ray::Ref::Sample_PrincipledNode(const pass_settings_t &ps, const ray_data_t &ray, const surface_t &surf,
                                     const lobe_weights_t &lobe_weights, const diff_params_t &diff,
                                     const spec_params_t &spec, const coat_params_t &coat,
                                     const transmission_params_t &trans, const float metallic, const float transmission,
                                     const fvec2 rand, float mix_rand, const float mix_weight,
                                     const float regularize_alpha, ray_data_t &new_ray) {
    const fvec4 I = make_fvec3(ray.d);

    const int diff_depth = get_diff_depth(ray.depth), spec_depth = get_spec_depth(ray.depth),
              refr_depth = get_refr_depth(ray.depth);
    // NOTE: transparency depth is not accounted here
    const int total_depth = diff_depth + spec_depth + refr_depth;

    if (mix_rand < lobe_weights.diffuse) {
        //
        // Diffuse lobe
        //
        if (diff_depth < ps.max_diff_depth && total_depth < ps.max_total_depth) {
            fvec4 V;
            fvec4 F = Sample_OrenDiffuse_BSDF(surf.T, surf.B, surf.N, I, diff.roughness, diff.base_color, rand, V);
            const float pdf = F.get<3>() * lobe_weights.diffuse;

            const fvec4 view_dir_ts = normalize(tangent_from_world(surf.T, surf.B, surf.N, -I));
            const fvec2 spec_alpha = calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha);
            fvec4 sampled_normal_ts = fvec4(0.0f, 0.0f, 1.0f, 0.0f);
            if (spec_alpha.get<0>() * spec_alpha.get<1>() >= 1e-7f) {
                sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, spec_alpha, rand);
            }
            const float FH_sampled =
                (fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), spec.fresnel_ior) - spec.F0) /
                (1.0f - spec.F0);
            const fvec4 spec_col_sampled = mix(spec.tmp_col, fvec4(1.0f), FH_sampled);
            const fvec4 diff_spec_atten_sampled = clamp(1.0f - spec_col_sampled, fvec4{0.0f}, fvec4{1.0f});

            F *= (1.0f - metallic) * (1.0f - transmission) * diff_spec_atten_sampled;

            new_ray.depth = pack_ray_type(RAY_TYPE_DIFFUSE);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(1, 0, 0, 0);

            memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
            memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));

            UNROLLED_FOR(i, 3, { new_ray.c[i] = F.get<i>() * safe_div_pos(mix_weight, pdf); })
            new_ray.pdf = pdf;
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT;
        }
    } else if (mix_rand < lobe_weights.diffuse + lobe_weights.sheen) {
        //
        // Sheen lobe
        //
        if (diff_depth < ps.max_diff_depth && total_depth < ps.max_total_depth) {
            fvec4 V;
            fvec4 F = Sample_PrincipledSheen_BSDF(surf.N, I, diff.sheen_roughness, rand, V);
            const float pdf = F.get<3>() * lobe_weights.sheen;

            F *= diff.sheen_color;

            new_ray.depth = pack_ray_type(RAY_TYPE_DIFFUSE);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(1, 0, 0, 0);

            memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
            memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));

            UNROLLED_FOR(i, 3, { new_ray.c[i] = F.get<i>() * safe_div_pos(mix_weight, pdf); })
            new_ray.pdf = pdf;
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT;
        }
    } else if (mix_rand < lobe_weights.diffuse + lobe_weights.sheen + lobe_weights.specular) {
        //
        // Main specular lobe
        //
        if (spec_depth < ps.max_spec_depth && total_depth < ps.max_total_depth) {
            const fvec2 alpha = calc_alpha(spec.roughness, spec.anisotropy, regularize_alpha);
            fvec4 V;
            fvec4 F =
                Sample_GGXSpecular_BSDF(surf.T, surf.B, surf.N, I, alpha, spec.fresnel_ior, spec.F0, spec.tmp_col,
                                        fvec4{1.0f}, rand, spec.metallic, spec.metal_f0, spec.metal_f82_b, true, V);
            const float pdf = F.get<3>() * lobe_weights.specular;

            new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

            UNROLLED_FOR(i, 3, { new_ray.c[i] = F.get<i>() * safe_div_pos(mix_weight, pdf); })
            new_ray.pdf = pdf;

            memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
            memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * fminf(alpha.get<0>(), alpha.get<1>());
        }
    } else if (mix_rand < lobe_weights.diffuse + lobe_weights.sheen + lobe_weights.specular + lobe_weights.clearcoat) {
        //
        // Clearcoat lobe (secondary specular)
        //
        if (spec_depth < ps.max_spec_depth && total_depth < ps.max_total_depth) {
            const float alpha = calc_alpha(coat.roughness, 0.0f, regularize_alpha).get<0>();
            fvec4 V;
            fvec4 F = Sample_PrincipledClearcoat_BSDF(surf.T, surf.B, surf.N, I, alpha, coat.ior, coat.F0, rand, V);
            const float pdf = F.get<3>() * lobe_weights.clearcoat;

            new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
            new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);

            UNROLLED_FOR(i, 3, { new_ray.c[i] = coat.weight * F.get<i>() * safe_div_pos(mix_weight, pdf); })
            new_ray.pdf = pdf;

            memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
            memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));
            new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * alpha;
        }
    } else /*if (mix_rand < lobe_weights.diffuse + lobe_weights.sheen + lobe_weights.specular + lobe_weights.clearcoat +
              lobe_weights.refraction)*/
    {
        //
        // Refraction/reflection lobes
        //
        const float alpha = calc_alpha(spec.roughness, 0.0f, regularize_alpha).get<0>();

        const fvec4 view_dir_ts = normalize(tangent_from_world(surf.T, surf.B, surf.N, -I));
        fvec4 sampled_normal_ts = fvec4(0.0f, 0.0f, 1.0f, 0.0f);
        if (sqr(alpha) >= 1e-7f) {
            sampled_normal_ts = SampleGGX_VNDF(view_dir_ts, alpha, rand);
        }
        const float trans_fresnel = fresnel_dielectric_cos(dot(view_dir_ts, sampled_normal_ts), 1.0f / trans.eta);

        mix_rand -= lobe_weights.diffuse + lobe_weights.sheen + lobe_weights.specular + lobe_weights.clearcoat;
        mix_rand = safe_div_pos(mix_rand, lobe_weights.refraction);
        if (((mix_rand >= trans_fresnel && refr_depth < ps.max_refr_depth) ||
             (mix_rand < trans_fresnel && spec_depth < ps.max_spec_depth)) &&
            total_depth < ps.max_total_depth) {

            fvec4 F, V;
            if (mix_rand < trans_fresnel) {
                F = Sample_GGXRefractionSpecular_BSDF(surf.T, surf.B, surf.N, I, alpha, (1.0f / trans.eta),
                                                      F0_from_IOR(1.0f / trans.eta), fvec4{1.0f}, fvec4{1.0f}, rand,
                                                      true, V);
                F.set<3>(F.get<3>() * trans_fresnel);

                new_ray.depth = pack_ray_type(RAY_TYPE_SPECULAR);
                new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 1, 0, 0);
                memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, surf.plane_N)), 3 * sizeof(float));
                new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * alpha;
            } else {
                F = Sample_GGXRefraction_BSDF(surf.T, surf.B, surf.N, I, alpha, trans.eta, sqrt(diff.base_color), rand,
                                              true, true, V);
                F.set<3>(F.get<3>() * (1.0 - trans_fresnel));

                new_ray.depth = pack_ray_type(RAY_TYPE_REFR);
                new_ray.depth |= mask_ray_depth(ray.depth) + pack_ray_depth(0, 0, 1, 0);
                memcpy(&new_ray.o[0], value_ptr(offset_ray(surf.P, -surf.plane_N)), 3 * sizeof(float));
                new_ray.cone_spread += MAX_CONE_SPREAD_INCREMENT * alpha;

                if (!trans.backfacing) {
                    // Entering the surface, push new value
                    push_ior_stack(new_ray.ior, spec.ior);
                } else {
                    // Exiting the surface, pop the last ior value
                    pop_ior_stack(new_ray.ior);
                }
            }

            const float pdf = F.get<3>() * lobe_weights.refraction;

            UNROLLED_FOR(i, 3, { new_ray.c[i] = F.get<i>() * safe_div_pos(mix_weight, pdf); })
            new_ray.pdf = pdf;

            memcpy(&new_ray.d[0], value_ptr(V), 3 * sizeof(float));
        }
    }
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_EnvColor(const ray_data_t &ray, const environment_t &env,
                                            const Cpu::TexStorageRGBA &tex_storage, const float pdf_factor,
                                            const fvec2 &rand) {
    const fvec4 I = make_fvec3(ray.d);
    fvec4 env_col = 1.0f;

    const uint32_t env_map = is_indirect(ray.depth) ? env.env_map : env.back_map;
    const float env_map_rotation = is_indirect(ray.depth) ? env.env_map_rotation : env.back_map_rotation;
    if (env_map != 0xffffffff) {
        env_col = SampleLatlong_RGBE(tex_storage, env_map, I, env_map_rotation, rand);
    }

    if (USE_NEE && env.light_index != 0xffffffff && pdf_factor >= 0.0f && is_indirect(ray.depth)) {
        if (env.qtree_levels) {
            const auto *qtree_mips = reinterpret_cast<const fvec4 *const *>(env.qtree_mips);

            const float light_pdf =
                safe_div_pos(Evaluate_EnvQTree(env_map_rotation, qtree_mips, env.qtree_levels, I), pdf_factor);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            env_col *= mis_weight;
        } else {
            const float light_pdf = safe_div_pos(0.5f, PI * pdf_factor);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            env_col *= mis_weight;
        }
    }

    env_col *= is_indirect(ray.depth) ? fvec4{env.env_col[0], env.env_col[1], env.env_col[2], 1.0f}
                                      : fvec4{env.back_col[0], env.back_col[1], env.back_col[2], 1.0f};
    env_col.set<3>(1.0f);

    return env_col;
}

Ray::Ref::fvec4 Ray::Ref::Evaluate_LightColor(const ray_data_t &ray, const hit_data_t &inter, const environment_t &env,
                                              const Cpu::TexStorageRGBA &tex_storage, Span<const light_t> lights,
                                              const uint32_t lights_count, const fvec2 &rand) {
    const fvec4 ro = make_fvec3(ray.o), I = make_fvec3(ray.d);

    const light_t &l = lights[-inter.obj_index - 1];
    const float pdf_factor = USE_HIERARCHICAL_NEE ? (1.0f / inter.u) : float(lights_count);

    fvec4 lcol = make_fvec3(l.col);
    if (l.sky_portal != 0) {
        fvec4 env_col = make_fvec3(env.env_col);
        if (env.env_map != 0xffffffff) {
            env_col *= SampleLatlong_RGBE(tex_storage, env.env_map, I, env.env_map_rotation, rand);
        }
        lcol *= env_col;
    }
    if (USE_NEE) {
        if (l.type == LIGHT_TYPE_SPHERE) {
            const fvec4 light_pos = make_fvec3(l.sph.pos);

            float d;
            const fvec4 disk_normal = normalize_len(light_pos - ro, d);

            if (d > l.sph.radius) {
                const float temp = sqrtf(d * d - l.sph.radius * l.sph.radius);
                const float disk_radius = (temp * l.sph.radius) / d;
                const float disk_dist = l.sph.radius > 0.0f ? ((temp * disk_radius) / l.sph.radius) : d;

                const float sampled_area = PI * disk_radius * disk_radius;
                const float cos_theta = dot(I, disk_normal);

                const float light_pdf = (disk_dist * disk_dist) / (sampled_area * cos_theta * pdf_factor);
                const float bsdf_pdf = ray.pdf;

                const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                lcol *= mis_weight;

                if (l.sph.spot > 0.0f && l.sph.blend > 0.0f) {
                    const float _dot = -dot(I, fvec4{l.sph.dir});
                    assert(_dot > 0.0f);
                    const float _angle = acosf(Ray::saturate(_dot));
                    assert(_angle <= l.sph.spot);
                    if (l.sph.blend > 0.0f) {
                        lcol *= Ray::saturate((l.sph.spot - _angle) / l.sph.blend);
                    }
                }
            }
        } else if (l.type == LIGHT_TYPE_DIR) {
            const float radius = l.dir.tan_angle;
            const float light_area = PI * radius * radius;

            const float cos_theta = dot(I, make_fvec3(l.dir.dir));

            const float light_pdf = 1.0f / (light_area * cos_theta * pdf_factor);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        } else if (l.type == LIGHT_TYPE_RECT) {
            const fvec4 light_pos = make_fvec3(l.rect.pos);
            const fvec4 light_u = make_fvec3(l.rect.u), light_v = make_fvec3(l.rect.v);
            const fvec4 light_forward = normalize(cross(light_u, light_v));

            if (l.rect.spread_normalization > 0.0f) {
                lcol *= spread_attenuation(I, light_forward, l.rect.tan_half_spread, l.rect.spread_normalization);
            }

            float light_pdf = 0.0f;
            if (USE_SPHERICAL_AREA_LIGHT_SAMPLING) {
                light_pdf = SampleSphericalRectangle(ro, light_pos, light_u, light_v, {}, nullptr) / pdf_factor;
            }
            if (light_pdf == 0.0f) {
                const float light_area = l.rect.area;
                const float cos_theta = dot(I, light_forward);
                light_pdf = (inter.t * inter.t) / (light_area * cos_theta * pdf_factor);
            }

            const float bsdf_pdf = ray.pdf;
            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        } else if (l.type == LIGHT_TYPE_DISK) {
            const fvec4 light_u = make_fvec3(l.disk.u), light_v = make_fvec3(l.disk.v);
            const fvec4 light_forward = normalize(cross(light_u, light_v));

            if (l.disk.spread_normalization > 0.0f) {
                lcol *= spread_attenuation(I, light_forward, l.disk.tan_half_spread, l.disk.spread_normalization);
            }

            const float light_area = l.disk.area;
            const float cos_theta = dot(I, light_forward);

            const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta * pdf_factor);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        } else if (l.type == LIGHT_TYPE_LINE) {
            const fvec4 light_pos = make_fvec3(l.line.pos);
            const fvec4 light_dir = make_fvec3(l.line.v);
            const float light_area = 0.5f * l.line.area;

            const fvec4 hit_pos = ro + inter.t * I;
            const fvec4 to_hit = hit_pos - light_pos;
            const fvec4 normal = normalize(to_hit - dot(to_hit, light_dir) * light_dir);

            const float cos_theta = -dot(I, normal);

            const float light_pdf = (inter.t * inter.t) / (light_area * cos_theta * pdf_factor);
            const float bsdf_pdf = ray.pdf;

            const float mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            lcol *= mis_weight;
        }
    }
    return lcol;
}

Ray::color_rgba_t Ray::Ref::ShadeSurface(const pass_settings_t &ps, const float limits[2],
                                         const eSpatialCacheMode cache_mode, const hit_data_t &inter,
                                         const ray_data_t &ray, const uint32_t ray_index, const uint32_t rand_seq[],
                                         const uint32_t rand_seed, const int iteration, const scene_data_t &sc,
                                         const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                                         int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                                         int *out_shadow_rays_count, uint32_t *out_def_sky, int *out_def_sky_count,
                                         color_rgba_t *out_base_color, color_rgba_t *out_depth_normal) {
    const fvec4 I = make_fvec3(ray.d);
    const fvec4 ro = make_fvec3(ray.o);

    // used to randomize random sequence among pixels
    const uint32_t px_hash = hash(ray.xy);
    const uint32_t rand_hash = hash_combine(px_hash, rand_seed);
    const uint32_t rand_dim = RAND_DIM_BASE_COUNT + get_total_depth(ray.depth) * RAND_DIM_BOUNCE_COUNT;

    const fvec2 tex_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_TEX, rand_hash, iteration - 1, rand_seq);

    if (inter.v < 0.0f) {
        if (out_def_sky && ray.cone_spread < sc.env.sky_map_spread_angle) {
            out_def_sky[(*out_def_sky_count)++] = ray_index;
            return color_rgba_t{};
        }

        float pdf_factor;
        if (USE_HIERARCHICAL_NEE) {
            pdf_factor = (get_total_depth(ray.depth) < ps.max_total_depth) ? safe_div_pos(1.0f, inter.u) : -1.0f;
        } else {
            pdf_factor = (get_total_depth(ray.depth) < ps.max_total_depth) ? float(sc.li_indices.size()) : -1.0f;
        }

        fvec4 env_col = Evaluate_EnvColor(ray, sc.env, *static_cast<const Cpu::TexStorageRGBA *>(textures[0]),
                                          pdf_factor, tex_rand);
        if (cache_mode != eSpatialCacheMode::Update) {
            env_col *= fvec4{ray.c[0], ray.c[1], ray.c[2], 0.0f};
        }
        const float sum = hsum(env_col);
        if (sum > limits[0]) {
            env_col *= (limits[0] / sum);
        }

        return color_rgba_t{env_col.get<0>(), env_col.get<1>(), env_col.get<2>(), env_col.get<3>()};
    }

    surface_t surf = {};
    surf.P = ro + inter.t * I;

    if (inter.obj_index < 0) { // Area light intersection
        fvec4 lcol = Evaluate_LightColor(ray, inter, sc.env, *static_cast<const Cpu::TexStorageRGBA *>(textures[0]),
                                         sc.lights, uint32_t(sc.li_indices.size()), tex_rand);
        if (cache_mode != eSpatialCacheMode::Update) {
            lcol *= fvec4{ray.c[0], ray.c[1], ray.c[2], 0.0f};
        }
        const float sum = hsum(lcol);
        if (sum > limits[0]) {
            lcol *= (limits[0] / sum);
        }

        return color_rgba_t{lcol.get<0>(), lcol.get<1>(), lcol.get<2>(), 1.0f};
    }

    const bool is_backfacing = (inter.prim_index < 0);
    const uint32_t tri_index = is_backfacing ? -inter.prim_index - 1 : inter.prim_index;

    const material_t *mat = &sc.materials[sc.tri_materials[tri_index].front_mi & MATERIAL_INDEX_BITS];
    const mesh_instance_t *mi = &sc.mesh_instances[inter.obj_index];

    const vertex_t &v1 = sc.vertices[sc.vtx_indices[tri_index * 3 + 0]];
    const vertex_t &v2 = sc.vertices[sc.vtx_indices[tri_index * 3 + 1]];
    const vertex_t &v3 = sc.vertices[sc.vtx_indices[tri_index * 3 + 2]];

    const float w = 1.0f - inter.u - inter.v;
    surf.N = normalize(make_fvec3(v1.n) * w + make_fvec3(v2.n) * inter.u + make_fvec3(v3.n) * inter.v);
    surf.uvs = fvec2(v1.t) * w + fvec2(v2.t) * inter.u + fvec2(v3.t) * inter.v;

    float pa;
    surf.plane_N = normalize_len(cross(fvec4{v2.p} - fvec4{v1.p}, fvec4{v3.p} - fvec4{v1.p}), pa);

    surf.B = make_fvec3(v1.b) * w + make_fvec3(v2.b) * inter.u + make_fvec3(v3.b) * inter.v;
    surf.T = cross(surf.B, surf.N);

    if (is_backfacing) {
        if (sc.tri_materials[tri_index].back_mi == 0xffff) {
            return color_rgba_t{0.0f, 0.0f, 0.0f, 0.0f};
        } else {
            mat = &sc.materials[sc.tri_materials[tri_index].back_mi & MATERIAL_INDEX_BITS];
            surf.plane_N = -surf.plane_N;
            surf.N = -surf.N;
            surf.B = -surf.B;
            surf.T = -surf.T;
        }
    }

    surf.plane_N = TransformNormal(surf.plane_N, mi->inv_xform);
    surf.N = TransformNormal(surf.N, mi->inv_xform);
    surf.B = TransformNormal(surf.B, mi->inv_xform);
    surf.T = TransformNormal(surf.T, mi->inv_xform);

    // normalize vectors (scaling might have been applied)
    surf.plane_N = safe_normalize(surf.plane_N);
    surf.N = safe_normalize(surf.N);
    surf.B = safe_normalize(surf.B);
    surf.T = safe_normalize(surf.T);

    const float ta = fabsf((v2.t[0] - v1.t[0]) * (v3.t[1] - v1.t[1]) - (v3.t[0] - v1.t[0]) * (v2.t[1] - v1.t[1]));

    const float cone_width = ray.cone_width + ray.cone_spread * inter.t;

    float lambda = 0.5f * fast_log2(ta / pa);
    lambda += fast_log2(cone_width);
    // lambda += 0.5 * fast_log2(tex_res.x * tex_res.y);
    // lambda -= fast_log2(fabsf(dot(I, plane_N)));

    const float ext_ior = peek_ior_stack(ray.ior, is_backfacing);

    fvec4 col = {0.0f};

    const int diff_depth = get_diff_depth(ray.depth), spec_depth = get_spec_depth(ray.depth),
              refr_depth = get_refr_depth(ray.depth);
    // NOTE: transparency depth is not accounted here
    const int total_depth = diff_depth + spec_depth + refr_depth;

    const fvec2 mix_term_rand =
        get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF_PICK, rand_hash, iteration - 1, rand_seq);

    float mix_rand = mix_term_rand.get<0>();
    float mix_weight = 1.0f;

    // resolve mix material
    while (mat->type == eShadingNode::Mix) {
        float mix_val = mat->strength;
        const uint32_t base_texture = mat->textures[BASE_TEXTURE];
        if (base_texture != 0xffffffff) {
            fvec4 tex_color = SampleBilinear(textures, base_texture, surf.uvs, 0, tex_rand);
            if (base_texture & TEX_YCOCG_BIT) {
                tex_color = YCoCg_to_RGB(tex_color);
            }
            if (base_texture & TEX_SRGB_BIT) {
                tex_color = srgb_to_linear(tex_color);
            }
            mix_val *= tex_color.get<0>();
        }

        const float eta = is_backfacing ? safe_div_pos(ext_ior, mat->ior) : safe_div_pos(mat->ior, ext_ior);
        const float RR = mat->ior != 0.0f ? fresnel_dielectric_cos(dot(I, surf.N), eta) : 1.0f;

        mix_val *= Ray::saturate(RR);

        if (mix_rand > mix_val) {
            mix_weight *= (mat->flags & MAT_FLAG_MIX_ADD) ? 1.0f / (1.0f - mix_val) : 1.0f;

            mat = &sc.materials[mat->textures[MIX_MAT1]];
            mix_rand = safe_div_pos(mix_rand - mix_val, 1.0f - mix_val);
        } else {
            mix_weight *= (mat->flags & MAT_FLAG_MIX_ADD) ? 1.0f / mix_val : 1.0f;

            mat = &sc.materials[mat->textures[MIX_MAT2]];
            mix_rand = safe_div_pos(mix_rand, mix_val);
        }
    }

    // apply normal map
    if (mat->textures[NORMALS_TEXTURE] != 0xffffffff) {
        fvec4 normals = SampleBilinear(textures, mat->textures[NORMALS_TEXTURE], surf.uvs, 0, tex_rand);
        normals = normals * 2.0f - 1.0f;
        normals.set<2>(1.0f);
        if (mat->textures[NORMALS_TEXTURE] & TEX_RECONSTRUCT_Z_BIT) {
            normals.set<2>(safe_sqrt(1.0f - normals.get<0>() * normals.get<0>() - normals.get<1>() * normals.get<1>()));
        }
        fvec4 in_normal = surf.N;
        surf.N = normalize(normals.get<0>() * surf.T + normals.get<2>() * surf.N + normals.get<1>() * surf.B);
        if (mat->normal_map_strength_unorm != 0xffff) {
            surf.N = normalize(in_normal + (surf.N - in_normal) * unpack_unorm_16(mat->normal_map_strength_unorm));
        }
        surf.N = ensure_valid_reflection(surf.plane_N, -I, surf.N);
    }

#if 0
    create_tbn_matrix(N, _tangent_from_world);
#else
    // Find radial tangent in local space
    const fvec4 P_ls = make_fvec3(v1.p) * w + make_fvec3(v2.p) * inter.u + make_fvec3(v3.p) * inter.v;
    // rotate around Y axis by 90 degrees in 2d
    fvec4 tangent = {-P_ls.get<2>(), 0.0f, P_ls.get<0>(), 0.0f};
    tangent = TransformNormal(tangent, mi->inv_xform);
    if (length2(cross(tangent, surf.N)) == 0.0f) {
        tangent = TransformNormal(P_ls, mi->inv_xform);
    }
    if (mat->tangent_rotation != 0.0f) {
        tangent = rotate_around_axis(tangent, surf.N, mat->tangent_rotation);
    }

    surf.B = safe_normalize(cross(tangent, surf.N));
    surf.T = cross(surf.N, surf.B);
#endif

    if (cache_mode == eSpatialCacheMode::Query && mat->type != eShadingNode::Emissive) {
        const fvec2 cache_rand = get_scrambled_2d_rand(rand_dim + RAND_DIM_CACHE, rand_hash, iteration - 1, rand_seq);

        const uint32_t grid_level = calc_grid_level(surf.P, sc.spatial_cache_grid);
        const float voxel_size = calc_voxel_size(grid_level, sc.spatial_cache_grid);

        bool use_cache = cone_width > mix(1.0f, 1.5f, cache_rand.get<0>()) * voxel_size;
        use_cache &= inter.t > mix(1.0f, 2.0f, cache_rand.get<1>()) * voxel_size;
        if (use_cache) {
            const uint32_t cache_entry =
                find_entry(sc.spatial_cache_entries, surf.P, surf.plane_N, sc.spatial_cache_grid);
            if (cache_entry != HASH_GRID_INVALID_CACHE_ENTRY) {
                const packed_cache_voxel_t &voxel = sc.spatial_cache_voxels[cache_entry];
                const cache_voxel_t unpacked = unpack_voxel_data(voxel);
                if (unpacked.sample_count > RAD_CACHE_SAMPLE_COUNT_THRESHOLD) {
                    fvec4 color = make_fvec3(unpacked.radiance);
                    color /= sc.spatial_cache_grid.exposure;
                    color *= fvec4{ray.c[0], ray.c[1], ray.c[2], 0.0f};
                    return color_rgba_t{color.get<0>(), color.get<1>(), color.get<2>(), color.get<3>()};
                }
            }
        }
    }

    light_sample_t ls;
    if (USE_NEE && (!sc.light_cwnodes.empty() || !sc.light_nodes.empty()) && mat->type != eShadingNode::Emissive) {
        const float rand_pick_light =
            get_scrambled_2d_rand(rand_dim + RAND_DIM_LIGHT_PICK, rand_hash, iteration - 1, rand_seq).get<0>();
        const fvec2 rand_light_uv =
            get_scrambled_2d_rand(rand_dim + RAND_DIM_LIGHT, rand_hash, iteration - 1, rand_seq);

        SampleLightSource(surf.P, surf.T, surf.B, surf.N, sc, textures, rand_pick_light, rand_light_uv, tex_rand, ls);
    }
    const float N_dot_L = dot(surf.N, ls.L);

    // sample base texture
    fvec4 base_color = fvec4{mat->base_color[0], mat->base_color[1], mat->base_color[2], 1.0f};
    if (mat->textures[BASE_TEXTURE] != 0xffffffff) {
        const uint32_t base_texture = mat->textures[BASE_TEXTURE];
        const float base_lod = get_texture_lod(textures, base_texture, lambda);
        fvec4 tex_color = SampleBilinear(textures, base_texture, surf.uvs, int(base_lod), tex_rand);
        if (base_texture & TEX_YCOCG_BIT) {
            tex_color = YCoCg_to_RGB(tex_color);
        }
        if (base_texture & TEX_SRGB_BIT) {
            tex_color = srgb_to_linear(tex_color);
        }
        base_color *= tex_color;
    }

    if (out_base_color) {
        memcpy(out_base_color->v, value_ptr(base_color), 3 * sizeof(float));
    }
    if (out_depth_normal) {
        if (cache_mode != eSpatialCacheMode::Update) {
            memcpy(out_depth_normal->v, value_ptr(surf.N), 3 * sizeof(float));
        } else {
            memcpy(out_depth_normal->v, value_ptr(surf.plane_N), 3 * sizeof(float));
        }
        out_depth_normal->v[3] = inter.t;
    }

    float roughness = unpack_unorm_16(mat->roughness_unorm);
    if (mat->textures[ROUGH_TEXTURE] != 0xffffffff) {
        const uint32_t roughness_tex = mat->textures[ROUGH_TEXTURE];
        const float roughness_lod = get_texture_lod(textures, roughness_tex, lambda);
        fvec4 roughness_color =
            SampleBilinear(textures, roughness_tex, surf.uvs, int(roughness_lod), tex_rand).get<0>();
        if (roughness_tex & TEX_SRGB_BIT) {
            roughness_color = srgb_to_linear(roughness_color);
        }
        roughness *= roughness_color.get<0>();
    }
    if (cache_mode == eSpatialCacheMode::Update && mat->type != eShadingNode::Diffuse) {
        roughness = fmaxf(roughness, RAD_CACHE_MIN_ROUGHNESS);
    }

    const fvec2 rand_bsdf = get_scrambled_2d_rand(rand_dim + RAND_DIM_BSDF, rand_hash, iteration - 1, rand_seq);

    ray_data_t &new_ray = out_secondary_rays[*out_secondary_rays_count];
    memcpy(new_ray.ior, ray.ior, 4 * sizeof(float));
    new_ray.cone_width = cone_width;
    new_ray.cone_spread = ray.cone_spread;
    new_ray.xy = ray.xy;
    new_ray.pdf = 0.0f;

    shadow_ray_t &sh_r = out_shadow_rays[*out_shadow_rays_count];
    sh_r.c[0] = sh_r.c[1] = sh_r.c[2] = 0.0f;
    sh_r.depth = ray.depth;
    sh_r.xy = ray.xy;

    const float regularize_alpha = (get_diff_depth(ray.depth) > 0) ? ps.regularize_alpha : 0.0f;

    // Sample materials
    if (mat->type == eShadingNode::Diffuse) {
        if (USE_NEE && ls.pdf > 0.0f && (ls.ray_flags & RAY_TYPE_DIFFUSE_BIT) != 0 && N_dot_L > 0.0f) {
            col += Evaluate_DiffuseNode(ls, ray, surf, base_color, roughness, mix_weight,
                                        (total_depth < ps.max_total_depth), sh_r);
        }
        if (diff_depth < ps.max_diff_depth && total_depth < ps.max_total_depth) {
            Sample_DiffuseNode(ray, surf, base_color, roughness, rand_bsdf, mix_weight, new_ray);
        }
    } else if (mat->type == eShadingNode::Glossy) {
        if (USE_NEE && ls.pdf > 0.0f && (ls.ray_flags & RAY_TYPE_SPECULAR_BIT) != 0 && N_dot_L > 0.0f) {
            col += Evaluate_GlossyNode(ls, ray, surf, base_color, roughness, regularize_alpha, mix_weight,
                                       (total_depth < ps.max_total_depth), sh_r);
        }
        if (spec_depth < ps.max_spec_depth && total_depth < ps.max_total_depth) {
            Sample_GlossyNode(ray, surf, base_color, roughness, regularize_alpha, rand_bsdf, mix_weight, new_ray);
        }
    } else if (mat->type == eShadingNode::Refractive) {
        if (USE_NEE && ls.pdf > 0.0f && (ls.ray_flags & RAY_TYPE_REFR_BIT) != 0 && N_dot_L < 0.0f) {
            const float eta = is_backfacing ? (mat->ior / ext_ior) : (ext_ior / mat->ior);
            col += Evaluate_RefractiveNode(ls, ray, surf, base_color, roughness, regularize_alpha, eta, mix_weight,
                                           (total_depth < ps.max_total_depth), sh_r);
        }
        if (refr_depth < ps.max_refr_depth && total_depth < ps.max_total_depth) {
            Sample_RefractiveNode(ray, surf, base_color, roughness, regularize_alpha, is_backfacing, mat->ior, ext_ior,
                                  rand_bsdf, mix_weight, new_ray);
        }
    } else if (mat->type == eShadingNode::Emissive) {
        float mis_weight = 1.0f;
        if (USE_NEE && (ray.depth & 0x00ffffff) != 0 && (mat->flags & MAT_FLAG_IMP_SAMPLE)) {
            float pdf_factor;
            if (USE_HIERARCHICAL_NEE) {
                // TODO: maybe this can be done more efficiently
                if (!sc.light_cwnodes.empty()) {
                    pdf_factor = EvalTriLightFactor(surf.P, ro, tri_index, sc.lights, sc.light_cwnodes);
                } else {
                    pdf_factor = EvalTriLightFactor(surf.P, ro, tri_index, sc.lights, sc.light_nodes);
                }
            } else {
                pdf_factor = float(sc.li_indices.size());
            }

            const auto p1 = make_fvec3(v1.p), p2 = make_fvec3(v2.p), p3 = make_fvec3(v3.p);

            float light_forward_len;
            fvec4 light_forward =
                normalize_len(TransformDirection(cross(p2 - p1, p3 - p1), mi->xform), light_forward_len);
            const float tri_area = 0.5f * light_forward_len;

            const float cos_theta = fabsf(dot(I, light_forward)); // abs for doublesided light
            if (cos_theta > 0.0f) {
                float light_pdf = 0.0f;
                if (USE_SPHERICAL_AREA_LIGHT_SAMPLING) {
                    const fvec4 P = TransformPoint(ro, mi->inv_xform);
                    light_pdf = SampleSphericalTriangle(P, p1, p2, p3, {}, nullptr) / pdf_factor;
                }
                if (light_pdf == 0.0f) {
                    light_pdf = (inter.t * inter.t) / (tri_area * cos_theta * pdf_factor);
                }

                const float bsdf_pdf = ray.pdf;
                mis_weight = power_heuristic(bsdf_pdf, light_pdf);
            }
        }
        col += mix_weight * mis_weight * mat->strength * base_color;
    } else if (mat->type == eShadingNode::Principled) {
        float metallic = unpack_unorm_16(mat->metallic_unorm);
        if (mat->textures[METALLIC_TEXTURE] != 0xffffffff) {
            const uint32_t metallic_tex = mat->textures[METALLIC_TEXTURE];
            const float metallic_lod = get_texture_lod(textures, metallic_tex, lambda);
            metallic *= SampleBilinear(textures, metallic_tex, surf.uvs, int(metallic_lod), tex_rand).get<0>();
        }

        float ior_level = unpack_unorm_16(mat->ior_level_unorm);
        if (mat->textures[IOR_LEVEL_TEXTURE] != 0xffffffff) {
            const uint32_t ior_level_tex = mat->textures[IOR_LEVEL_TEXTURE];
            const float ior_level_lod = get_texture_lod(textures, ior_level_tex, lambda);
            fvec4 ior_level_color = SampleBilinear(textures, ior_level_tex, surf.uvs, int(ior_level_lod), tex_rand);
            if (ior_level_tex & TEX_SRGB_BIT) {
                ior_level_color = srgb_to_linear(ior_level_color);
            }
            ior_level *= ior_level_color.get<0>();
        }

        const float diffuse_roughness = unpack_unorm_16(mat->diffuse_roughness_unorm);
        const float sheen_roughness = unpack_unorm_16(mat->sheen_roughness_unorm);
        const fvec4 specular_tint = {unpack_unorm_16(mat->specular_tint_unorm[0]),
                                     unpack_unorm_16(mat->specular_tint_unorm[1]),
                                     unpack_unorm_16(mat->specular_tint_unorm[2]), 0.0f};
        const float transmission = unpack_unorm_16(mat->transmission_unorm);
        const float coat_weight = unpack_unorm_16(mat->coat_weight_unorm);
        const float coat_roughness = unpack_unorm_16(mat->coat_roughness_unorm);
        const float sheen = unpack_unorm_16(mat->sheen_unorm);
        const fvec4 sheen_tint = {unpack_unorm_16(mat->sheen_tint_unorm[0]), unpack_unorm_16(mat->sheen_tint_unorm[1]),
                                  unpack_unorm_16(mat->sheen_tint_unorm[2]), 0.0f};

        diff_params_t diff = {};
        diff.base_color = base_color;
        diff.sheen_color = sheen * sheen_tint;
        diff.roughness = diffuse_roughness;
        diff.sheen_roughness = sheen_roughness;

        spec_params_t spec = {};
        const float base_F0 = fresnel_dielectric_cos(1.0f, mat->ior);
        spec.F0 = clamp(2.0f * ior_level * base_F0, 0.0f, 0.9999f);
        const float sqrt_F0 = sqrtf(spec.F0);
        spec.fresnel_ior = (1.0f + sqrt_F0) / (1.0f - sqrt_F0);
        spec.tmp_col = mix(spec.F0 * specular_tint, base_color, metallic);
        spec.roughness = roughness;
        spec.ior = mat->ior;
        spec.anisotropy = unpack_unorm_16(mat->anisotropic_unorm);
        spec.metallic = metallic;
        spec.metal_f0 = base_color;
        spec.metal_f82_b = fresnel_f82tint_B(spec.metal_f0, specular_tint);

        coat_params_t coat = {};
        coat.weight = coat_weight;
        coat.roughness = coat_roughness;
        coat.ior = mat->coat_ior;
        coat.F0 = fresnel_dielectric_cos(1.0f, mat->coat_ior);

        transmission_params_t trans = {};
        trans.eta = is_backfacing ? (mat->ior / ext_ior) : (ext_ior / mat->ior);
        trans.backfacing = is_backfacing;

        const float base_color_lum = lum(base_color);

        // Approximation of FH (using shading normal)
        const float FN = (fresnel_dielectric_cos(dot(I, surf.N), spec.fresnel_ior) - spec.F0) / (1.0f - spec.F0);

        const fvec4 approx_spec_col = mix(spec.tmp_col, fvec4(1.0f), FN);
        const float spec_color_lum = lum(approx_spec_col);
        // matches Cycles' sheen closure sample_weight = average(sheen_weight * sheen_tint)
        const float sheen_lum = lum(diff.sheen_color);

        const auto lobe_weights =
            get_lobe_weights(base_color_lum, sheen_lum, spec_color_lum, ior_level, metallic, transmission, coat_weight);

        if (USE_NEE && ls.pdf > 0.0f) {
            col += Evaluate_PrincipledNode(ls, ray, surf, lobe_weights, diff, spec, coat, trans, metallic, transmission,
                                           N_dot_L, mix_weight, (total_depth < ps.max_total_depth), regularize_alpha,
                                           sh_r);
        }
        Sample_PrincipledNode(ps, ray, surf, lobe_weights, diff, spec, coat, trans, metallic, transmission, rand_bsdf,
                              mix_rand, mix_weight, regularize_alpha, new_ray);
    } /*else if (mat->type == TransparentNode) {
        assert(false);
    }*/

    const bool can_terminate_path = USE_PATH_TERMINATION && total_depth > ps.min_total_depth;

    if (cache_mode != eSpatialCacheMode::Update) {
        UNROLLED_FOR(i, 3, { new_ray.c[i] *= ray.c[i]; })
    }
    const float lum = fmaxf(new_ray.c[0], fmaxf(new_ray.c[1], new_ray.c[2]));
    const float p = mix_term_rand.get<1>();
    const float q = can_terminate_path ? fmaxf(0.05f, 1.0f - lum) : 0.0f;
    if (p >= q && lum > 0.0f && new_ray.pdf > 0.0f) {
        new_ray.pdf = fminf(new_ray.pdf, 1e6f);
        new_ray.c[0] /= (1.0f - q);
        new_ray.c[1] /= (1.0f - q);
        new_ray.c[2] /= (1.0f - q);
        ++(*out_secondary_rays_count);
    }

    if (USE_NEE) {
        if (cache_mode != eSpatialCacheMode::Update) {
            UNROLLED_FOR(i, 3, { sh_r.c[i] *= ray.c[i]; })
        }
        const float sh_lum = fmaxf(sh_r.c[0], fmaxf(sh_r.c[1], sh_r.c[2]));
        if (sh_lum > 0.0f) {
            // actual ray direction accouning for bias from both ends
            const fvec4 to_light = normalize_len(ls.lp - fvec4{sh_r.o[0], sh_r.o[1], sh_r.o[2], 0.0f}, sh_r.dist);
            memcpy(&sh_r.d[0], value_ptr(to_light), 3 * sizeof(float));
            sh_r.dist *= ls.dist_mul;
            if (ls.from_env) {
                // NOTE: hacky way to identify env ray
                sh_r.dist = -sh_r.dist;
            }
            ++(*out_shadow_rays_count);
        }
    }
    if (cache_mode != eSpatialCacheMode::Update) {
        col *= fvec4{ray.c[0], ray.c[1], ray.c[2], 0.0f};
    }
    const float sum = hsum(col);
    if (sum > limits[1]) {
        col *= (limits[1] / sum);
    }

    return color_rgba_t{col.get<0>(), col.get<1>(), col.get<2>(), 1.0f};
}

void Ray::Ref::ShadePrimary(const pass_settings_t &ps, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
                            const uint32_t rand_seq[], const uint32_t rand_seed, const int iteration,
                            const eSpatialCacheMode cache_mode, const scene_data_t &sc,
                            const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                            int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays, int *out_shadow_rays_count,
                            uint32_t *out_def_sky, int *out_def_sky_count, int img_w, float mix_factor,
                            color_rgba_t *out_color, color_rgba_t *out_base_color, color_rgba_t *out_depth_normal) {
    const float limits[2] = {(ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX,
                             (ps.clamp_direct != 0.0f) ? 3.0f * ps.clamp_direct : FLT_MAX};
    for (int i = 0; i < int(inters.size()); ++i) {
        const ray_data_t &r = rays[i];
        const hit_data_t &inter = inters[i];

        const int x = (r.xy >> 16) & 0x0000ffff;
        const int y = r.xy & 0x0000ffff;

        color_rgba_t base_color = {}, depth_normal = {};
        const color_rgba_t col =
            ShadeSurface(ps, limits, cache_mode, inter, r, i, rand_seq, rand_seed, iteration, sc, textures,
                         out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count,
                         out_def_sky, out_def_sky_count, &base_color, &depth_normal);
        out_color[y * img_w + x] = col;

        if (out_base_color) {
            if (cache_mode != eSpatialCacheMode::Update) {
                auto old_val = Ref::fvec4{out_base_color[y * img_w + x].v, Ref::vector_aligned};
                auto new_val = Ref::fvec4{base_color.v, Ref::vector_aligned};
                const float norm_factor =
                    fmaxf(fmaxf(new_val.get<0>(), new_val.get<1>()), fmaxf(new_val.get<2>(), 1.0f));
                new_val /= norm_factor;
                old_val += (new_val - old_val) * mix_factor;
                old_val.store_to(out_base_color[y * img_w + x].v, Ref::vector_aligned);
            } else {
                out_base_color[y * img_w + x] = base_color;
            }
        }
        if (out_depth_normal) {
            if (cache_mode != eSpatialCacheMode::Update) {
                auto old_val = Ref::fvec4{out_depth_normal[y * img_w + x].v, Ref::vector_aligned};
                old_val += (Ref::fvec4{depth_normal.v, Ref::vector_aligned} - old_val) * mix_factor;
                old_val.store_to(out_depth_normal[y * img_w + x].v, Ref::vector_aligned);
            } else {
                out_depth_normal[y * img_w + x] = depth_normal;
            }
        }
    }
}

void Ray::Ref::ShadeSecondary(const pass_settings_t &ps, const float clamp_direct, Span<const hit_data_t> inters,
                              Span<const ray_data_t> rays, const uint32_t rand_seq[], uint32_t rand_seed, int iteration,
                              const eSpatialCacheMode cache_mode, const scene_data_t &sc,
                              const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                              int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays, int *out_shadow_rays_count,
                              uint32_t *out_def_sky, int *out_def_sky_count, int img_w, color_rgba_t *out_color,
                              color_rgba_t *out_base_color, color_rgba_t *out_depth_normal) {
    const float limits[2] = {(clamp_direct != 0.0f) ? 3.0f * clamp_direct : FLT_MAX,
                             (ps.clamp_indirect != 0.0f) ? 3.0f * ps.clamp_indirect : FLT_MAX};
    for (int i = 0; i < int(inters.size()); ++i) {
        const ray_data_t &r = rays[i];
        const hit_data_t &inter = inters[i];

        const int x = (r.xy >> 16) & 0x0000ffff;
        const int y = r.xy & 0x0000ffff;

        color_rgba_t base_color = {}, depth_normal = {};
        color_rgba_t col =
            ShadeSurface(ps, limits, cache_mode, inter, r, i, rand_seq, rand_seed, iteration, sc, textures,
                         out_secondary_rays, out_secondary_rays_count, out_shadow_rays, out_shadow_rays_count,
                         out_def_sky, out_def_sky_count, &base_color, &depth_normal);
        if (cache_mode != eSpatialCacheMode::Update) {
            auto old_val = Ref::fvec4{out_color[y * img_w + x].v, Ref::vector_aligned};
            old_val += make_fvec3(col.v);
            old_val.store_to(out_color[y * img_w + x].v, Ref::vector_aligned);
        } else {
            out_color[y * img_w + x] = col;
        }
        if (out_base_color) {
            out_base_color[y * img_w + x] = base_color;
        }
        if (out_depth_normal) {
            out_depth_normal[y * img_w + x] = depth_normal;
        }
    }
}

#pragma warning(pop)