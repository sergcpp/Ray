#include "RendererNEON.h"

namespace Ray {
namespace Neon {
template void GeneratePrimaryRays<RayPacketDimX, RayPacketDimY>(const int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<RayPacketSize>> &out_rays);
template void SampleMeshInTextureSpace<RayPacketDimX, RayPacketDimY>(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                                                                     const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<RayPacketSize>> &out_rays, aligned_vector<hit_data_t<RayPacketSize>> &out_inters);

template void SortRays<RayPacketSize>(ray_packet_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks, int &secondary_rays_count, const float root_min[3], const float cell_size[3],
                                      simd_ivec<RayPacketSize> *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

template bool IntersectTris<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<RayPacketSize> &out_inter);
template bool IntersectTris<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<RayPacketSize> &out_inter);

template bool Traverse_MacroTree_Stackless_CPU<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                              const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                                              const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_Stackless_CPU<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                              const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MacroTree_WithStack<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                          const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                                          const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_WithStack<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                          const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);

template ray_packet_t<RayPacketSize> TransformRay<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const float *xform);
template void TransformNormal<RayPacketSize>(const simd_fvec<RayPacketSize> n[3], const float *inv_xform, simd_fvec<RayPacketSize> out_n[3]);
template void TransformUVs<RayPacketSize>(const simd_fvec<RayPacketSize> _uvs[2], float sx, float sy, const texture_t &t, const simd_ivec<RayPacketSize> &mip_level, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_res[2]);

template void SampleNearest<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleBilinear<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_ivec<RayPacketSize> &lod, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleBilinear<RayPacketSize>(const Ref::TextureAtlas &atlas, const simd_fvec<RayPacketSize> uvs[2], const simd_ivec<RayPacketSize> &page, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleTrilinear<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleAnisotropic<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> duv_dx[2], const simd_fvec<RayPacketSize> duv_dy[2], const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleLatlong_RGBE<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> dir[3], const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgb[3]);

template void ComputeDirectLighting<RayPacketSize>(const simd_fvec<RayPacketSize> P[3], const simd_fvec<RayPacketSize> N[3], const simd_fvec<RayPacketSize> B[3], const simd_fvec<RayPacketSize> plane_N[3],
                                                   const float *halton, const int hi, const simd_fvec<RayPacketSize> &rand_offset, const simd_fvec<RayPacketSize> &rand_offset2,
                                                   const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                                   const uint32_t *vtx_indices, const vertex_t *vertices, const bvh_node_t *nodes, uint32_t node_index,
                                                   const tri_accel_t *tris, const uint32_t *tri_indices, const light_t *lights,
                                                   const uint32_t *li_indices, uint32_t light_node_index, const simd_ivec<RayPacketSize> &ray_mask, simd_fvec<RayPacketSize> *out_col);

template void ShadeSurface<RayPacketSize>(const simd_ivec<RayPacketSize> &index, const pass_info_t &pi, const float *halton, const hit_data_t<RayPacketSize> &inter, const ray_packet_t<RayPacketSize> &ray,
                                          const environment_t &env, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
                                          const mesh_t *meshes, const transform_t *transforms, const uint32_t *vtx_indices, const vertex_t *vertices,
                                          const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const uint32_t *tri_indices,
                                          const material_t *materials, const texture_t *textures, const Ref::TextureAtlas &tex_atlas,
                                          const light_t *lights, const uint32_t *li_indices, uint32_t light_node_index, simd_fvec<RayPacketSize> out_rgba[4], simd_ivec<RayPacketSize> *out_secondary_masks, ray_packet_t<RayPacketSize> *out_secondary_rays, int *out_secondary_rays_count);

template class RendererSIMD<RayPacketDimX, RayPacketDimY>;
}
}
