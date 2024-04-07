#pragma once

#include "CoreRef.h"

namespace Ray {
namespace Ref {
// BRDFs
float BRDF_PrincipledDiffuse(const fvec4 &V, const fvec4 &N, const fvec4 &L, const fvec4 &H, float roughness);

fvec4 Evaluate_OrenDiffuse_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L, float roughness,
                                const fvec4 &base_color);
fvec4 Sample_OrenDiffuse_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, float roughness,
                              const fvec4 &base_color, fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_PrincipledDiffuse_BSDF(const fvec4 &V, const fvec4 &N, const fvec4 &L, float roughness,
                                      const fvec4 &base_color, const fvec4 &sheen_color, bool uniform_sampling);
fvec4 Sample_PrincipledDiffuse_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, float roughness,
                                    const fvec4 &base_color, const fvec4 &sheen_color, bool uniform_sampling,
                                    fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_GGXSpecular_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts, const fvec4 &reflected_dir_ts,
                                fvec2 alpha, float spec_ior, float spec_F0, const fvec4 &spec_col,
                                const fvec4 &spec_col_90);
fvec4 Sample_GGXSpecular_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, fvec2 alpha,
                              float spec_ior, float spec_F0, const fvec4 &spec_col, const fvec4 &spec_col_90,
                              fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_GGXRefraction_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts, const fvec4 &refr_dir_ts,
                                  fvec2 slpha, float eta, const fvec4 &refr_col);
fvec4 Sample_GGXRefraction_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I, fvec2 alpha, float eta,
                                const fvec4 &refr_col, fvec2 rand, fvec4 &out_V);

fvec4 Evaluate_PrincipledClearcoat_BSDF(const fvec4 &view_dir_ts, const fvec4 &sampled_normal_ts,
                                        const fvec4 &reflected_dir_ts, float clearcoat_roughness2, float clearcoat_ior,
                                        float clearcoat_F0);
fvec4 Sample_PrincipledClearcoat_BSDF(const fvec4 &T, const fvec4 &B, const fvec4 &N, const fvec4 &I,
                                      float clearcoat_roughness2, float clearcoat_ior, float clearcoat_F0, fvec2 rand,
                                      fvec4 &out_V);

// Evaluate individual nodes
fvec4 Evaluate_DiffuseNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                           const fvec4 &base_color, float roughness, float mix_weight, bool use_mis,
                           shadow_ray_t &sh_r);
void Sample_DiffuseNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color, float roughness,
                        fvec2 rand, float mix_weight, ray_data_t &new_ray);

fvec4 Evaluate_GlossyNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                          const fvec4 &base_color, float roughness, float regularize_alpha, float spec_ior,
                          float spec_F0, float mix_weight, bool use_mis, shadow_ray_t &sh_r);
void Sample_GlossyNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color, float roughness,
                       float regularize_alpha, float spec_ior, float spec_F0, fvec2 rand, float mix_weight,
                       ray_data_t &new_ray);

fvec4 Evaluate_RefractiveNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                              const fvec4 &base_color, float roughness, float regularize_alpha, float eta,
                              float mix_weight, bool use_mis, shadow_ray_t &sh_r);
void Sample_RefractiveNode(const ray_data_t &ray, const surface_t &surf, const fvec4 &base_color, float roughness,
                           float regularize_alpha, bool is_backfacing, float int_ior, float ext_ior, fvec2 rand,
                           float mix_weight, ray_data_t &new_ray);

struct diff_params_t {
    fvec4 base_color;
    fvec4 sheen_color;
    float roughness;
};

struct spec_params_t {
    fvec4 tmp_col;
    float roughness;
    float ior;
    float F0;
    float anisotropy;
};

struct clearcoat_params_t {
    float roughness;
    float ior;
    float F0;
};

struct transmission_params_t {
    float roughness;
    float int_ior;
    float eta;
    float fresnel;
    bool backfacing;
};

struct lobe_weights_t {
    float diffuse, specular, clearcoat, refraction;
};

fvec4 Evaluate_PrincipledNode(const light_sample_t &ls, const ray_data_t &ray, const surface_t &surf,
                              const lobe_weights_t &lobe_weights, const diff_params_t &diff, const spec_params_t &spec,
                              const clearcoat_params_t &coat, const transmission_params_t &trans, float metallic,
                              float transmission, float N_dot_L, float mix_weight, bool use_mis, float regularize_alpha,
                              shadow_ray_t &sh_r);
void Sample_PrincipledNode(const pass_settings_t &ps, const ray_data_t &ray, const surface_t &surf,
                           const lobe_weights_t &lobe_weights, const diff_params_t &diff, const spec_params_t &spec,
                           const clearcoat_params_t &coat, const transmission_params_t &trans, float metallic,
                           float transmission, fvec2 rand, float mix_rand, float mix_weight, float regularize_alpha,
                           ray_data_t &new_ray);

// Shade
color_rgba_t ShadeSurface(const pass_settings_t &ps, const float limits[2], eSpatialCacheMode cache_mode,
                          const hit_data_t &inter, const ray_data_t &ray, const uint32_t rand_seq[], uint32_t rand_seed,
                          int iteration, const scene_data_t &sc, const Cpu::TexStorageBase *const textures[],
                          ray_data_t *out_secondary_rays, int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                          int *out_shadow_rays_count, color_rgba_t *out_base_color, color_rgba_t *out_depth_normal);
void ShadePrimary(const pass_settings_t &ps, Span<const hit_data_t> inters, Span<const ray_data_t> rays,
                  const uint32_t rand_seq[], uint32_t rand_seed, int iteration, eSpatialCacheMode cache_mode,
                  const scene_data_t &sc, const Cpu::TexStorageBase *const textures[], ray_data_t *out_secondary_rays,
                  int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays, int *out_shadow_rays_count, int img_w,
                  float mix_factor, color_rgba_t *out_color, color_rgba_t *out_base_color,
                  color_rgba_t *out_depth_normal);
void ShadeSecondary(const pass_settings_t &ps, float clamp_direct, Span<const hit_data_t> inters,
                    Span<const ray_data_t> rays, const uint32_t rand_seq[], uint32_t rand_seed, int iteration,
                    eSpatialCacheMode cache_mode, const scene_data_t &sc, const Cpu::TexStorageBase *const textures[],
                    ray_data_t *out_secondary_rays, int *out_secondary_rays_count, shadow_ray_t *out_shadow_rays,
                    int *out_shadow_rays_count, int img_w, color_rgba_t *out_color, color_rgba_t *out_base_color,
                    color_rgba_t *out_depth_normal);
} // namespace Ref
} // namespace Ray