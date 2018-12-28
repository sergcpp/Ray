#include "RendererNEON.h"

namespace Ray {
namespace Neon {
template void GeneratePrimaryRays<RayPacketDimX, RayPacketDimY>(const int iteration, const camera_t &cam, const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<RayPacketSize>> &out_rays);
template void SampleMeshInTextureSpace<RayPacketDimX, RayPacketDimY>(int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices, const vertex_t *vertices,
                                                                     const rect_t &r, int w, int h, const float *halton, aligned_vector<ray_packet_t<RayPacketSize>> &out_rays, aligned_vector<hit_data_t<RayPacketSize>> &out_inters);

template void SortRays_CPU<RayPacketSize>(ray_packet_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks, int &secondary_rays_count, const float root_min[3], const float cell_size[3],
                                          simd_ivec<RayPacketSize> *hash_values, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
template void SortRays_GPU<RayPacketSize>(ray_packet_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks, int &secondary_rays_count, const float root_min[3], const float cell_size[3],
                                          simd_ivec<RayPacketSize> *hash_values, int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

template bool IntersectTris_ClosestHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<RayPacketSize> &out_inter);
template bool IntersectTris_ClosestHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<RayPacketSize> &out_inter);
template bool IntersectTris_AnyHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const tri_accel_t *tris, uint32_t num_tris, uint32_t obj_index, hit_data_t<RayPacketSize> &out_inter, simd_ivec<RayPacketSize> &out_is_solid_hit);
template bool IntersectTris_AnyHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const tri_accel_t *tris, const uint32_t *indices, uint32_t num_tris, uint32_t obj_index, hit_data_t<RayPacketSize> &out_inter, simd_ivec<RayPacketSize> &out_is_solid_hit);

#ifdef USE_STACKLESS_BVH_TRAVERSAL
template bool Traverse_MacroTree_Stackless_CPU<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                              const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                                              const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_Stackless_CPU<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                              const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
#endif
                                                              
template bool Traverse_MacroTree_WithStack_ClosestHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                                     const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                                                     const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MacroTree_WithStack_AnyHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                                 const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms,
                                                                 const tri_accel_t *tris, const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter, simd_ivec<RayPacketSize> &is_solid_hit);
template bool Traverse_MicroTree_WithStack_ClosestHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                                     const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_WithStack_AnyHit<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
                                                                 const tri_accel_t *tris, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter, simd_ivec<RayPacketSize> &is_solid_hit);

template ray_packet_t<RayPacketSize> TransformRay<RayPacketSize>(const ray_packet_t<RayPacketSize> &r, const float *xform);
template void TransformNormal<RayPacketSize>(const simd_fvec<RayPacketSize> n[3], const float *inv_xform, simd_fvec<RayPacketSize> out_n[3]);
template void TransformUVs<RayPacketSize>(const simd_fvec<RayPacketSize> _uvs[2], float sx, float sy, const texture_t &t, const simd_ivec<RayPacketSize> &mip_level, simd_fvec<RayPacketSize> out_res[2]);

template void SampleNearest<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleBilinear<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_ivec<RayPacketSize> &lod, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleBilinear<RayPacketSize>(const Ref::TextureAtlas &atlas, const simd_fvec<RayPacketSize> uvs[2], const simd_ivec<RayPacketSize> &page, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleTrilinear<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod, const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleAnisotropic<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> duv_dx[2], const simd_fvec<RayPacketSize> duv_dy[2], const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleLatlong_RGBE<RayPacketSize>(const Ref::TextureAtlas &atlas, const texture_t &t, const simd_fvec<RayPacketSize> dir[3], const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgb[3]);

template simd_fvec<RayPacketSize> ComputeVisibility<RayPacketSize>(const simd_fvec<RayPacketSize> p1[3], const simd_fvec<RayPacketSize> p2[3], const simd_ivec<RayPacketSize> &mask,
                                                                   const float *halton, const int hi, const simd_ivec<RayPacketSize> &rand_hash2,
                                                                   const scene_data_t &sc, uint32_t node_index, const Ref::TextureAtlas &tex_atlas);

template void ComputeDirectLighting<RayPacketSize>(const simd_fvec<RayPacketSize> P[3], const simd_fvec<RayPacketSize> N[3], const simd_fvec<RayPacketSize> B[3], const simd_fvec<RayPacketSize> plane_N[3],
                                                   const float *halton, const int hi, const simd_ivec<RayPacketSize> &rand_hash, const simd_ivec<RayPacketSize> &rand_hash2,
                                                   const simd_fvec<RayPacketSize> &rand_offset, const simd_fvec<RayPacketSize> &rand_offset2, const scene_data_t &sc, uint32_t node_index,
                                                   uint32_t light_node_index, const Ref::TextureAtlas &tex_atlas, const simd_ivec<RayPacketSize> &ray_mask, simd_fvec<RayPacketSize> *out_col);

template void ComputeDerivatives<RayPacketSize>(const simd_fvec<RayPacketSize> I[3], const simd_fvec<RayPacketSize> &t, const simd_fvec<RayPacketSize> do_dx[3], const simd_fvec<RayPacketSize> do_dy[3], const simd_fvec<RayPacketSize> dd_dx[3], const simd_fvec<RayPacketSize> dd_dy[3],
                                                const simd_fvec<RayPacketSize> p1[3], const simd_fvec<RayPacketSize> p2[3], const simd_fvec<RayPacketSize> p3[3], const simd_fvec<RayPacketSize> n1[3], const simd_fvec<RayPacketSize> n2[3], const simd_fvec<RayPacketSize> n3[3],
                                                const simd_fvec<RayPacketSize> u1[2], const simd_fvec<RayPacketSize> u2[2], const simd_fvec<RayPacketSize> u3[2], const simd_fvec<RayPacketSize> plane_N[3], derivatives_t<RayPacketSize> &out_der);

template void ShadeSurface<RayPacketSize>(const simd_ivec<RayPacketSize> &index, const pass_info_t &pi, const float *halton, const hit_data_t<RayPacketSize> &inter, const ray_packet_t<RayPacketSize> &ray,
                                          const scene_data_t &sc, uint32_t node_index, uint32_t light_node_index, const Ref::TextureAtlas &tex_atlas,
                                          simd_fvec<RayPacketSize> out_rgba[4], simd_ivec<RayPacketSize> *out_secondary_masks, ray_packet_t<RayPacketSize> *out_secondary_rays, int *out_secondary_rays_count);

template class RendererSIMD<RayPacketDimX, RayPacketDimY>;
}
}
