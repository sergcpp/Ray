#include "RendererSSE41.h"

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("sse2")
#endif

namespace Ray {
namespace Sse41 {
template void SortRays_CPU<RayPacketSize>(ray_data_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks,
                                          int &secondary_rays_count, const float root_min[3], const float cell_size[3],
                                          simd_ivec<RayPacketSize> *hash_values, uint32_t *scan_values,
                                          ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
template void SortRays_GPU<RayPacketSize>(ray_data_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks,
                                          int &secondary_rays_count, const float root_min[3], const float cell_size[3],
                                          simd_ivec<RayPacketSize> *hash_values, int *head_flags, uint32_t *scan_values,
                                          ray_chunk_t *chunks, ray_chunk_t *chunks_temp, uint32_t *skeleton);

template bool Traverse_MacroTree_WithStack_ClosestHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const tri_accel_t *tris, const uint32_t *tri_indices,
    hit_data_t<RayPacketSize> &inter);
template bool Traverse_MacroTree_WithStack_ClosestHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const mbvh_node_t *oct_nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const mtri_accel_t *mtris, const uint32_t *tri_indices,
    hit_data_t<RayPacketSize> &inter);
template simd_ivec<RayPacketSize> Traverse_MacroTree_WithStack_AnyHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const tri_accel_t *tris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
template simd_ivec<RayPacketSize> Traverse_MacroTree_WithStack_AnyHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const mbvh_node_t *oct_nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const mtri_accel_t *mtris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_WithStack_ClosestHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris,
    const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_WithStack_ClosestHit<RayPacketSize>(const float ro[3], const float rd[3], int i,
                                                                     const mbvh_node_t *oct_nodes, uint32_t node_index,
                                                                     const mtri_accel_t *mtris,
                                                                     const uint32_t *tri_indices, int obj_index,
                                                                     hit_data_t<RayPacketSize> &inter);
template simd_ivec<RayPacketSize> Traverse_MicroTree_WithStack_AnyHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris,
    const tri_mat_data_t *materials, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
template bool Traverse_MicroTree_WithStack_AnyHit(const float ro[3], const float rd[3], int i,
                                                  const mbvh_node_t *oct_nodes, uint32_t node_index,
                                                  const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                                  const uint32_t *tri_indices, int obj_index,
                                                  hit_data_t<RayPacketSize> &inter);

template void SampleNearest<RayPacketSize>(const Ref::TexStorageBase *const textures[], const uint32_t index,
                                           const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod,
                                           const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleBilinear<RayPacketSize>(const Ref::TexStorageBase *const textures[], const uint32_t index,
                                            const simd_fvec<RayPacketSize> uvs[2], const simd_ivec<RayPacketSize> &lod,
                                            const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleTrilinear<RayPacketSize>(const Ref::TexStorageBase *const textures[], const uint32_t index,
                                             const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod,
                                             const simd_ivec<RayPacketSize> &mask,
                                             simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleLatlong_RGBE<RayPacketSize>(const Ref::TexStorageRGBA &atlas, uint32_t index,
                                                const simd_fvec<RayPacketSize> dir[3], float y_rotation,
                                                const simd_ivec<RayPacketSize> &mask,
                                                simd_fvec<RayPacketSize> out_rgb[3]);

template void ComputeDerivatives<RayPacketSize>(
    const simd_fvec<RayPacketSize> I[3], const simd_fvec<RayPacketSize> &t, const simd_fvec<RayPacketSize> do_dx[3],
    const simd_fvec<RayPacketSize> do_dy[3], const simd_fvec<RayPacketSize> dd_dx[3],
    const simd_fvec<RayPacketSize> dd_dy[3], const simd_fvec<RayPacketSize> p1[3], const simd_fvec<RayPacketSize> p2[3],
    const simd_fvec<RayPacketSize> p3[3], const simd_fvec<RayPacketSize> n1[3], const simd_fvec<RayPacketSize> n2[3],
    const simd_fvec<RayPacketSize> n3[3], const simd_fvec<RayPacketSize> u1[2], const simd_fvec<RayPacketSize> u2[2],
    const simd_fvec<RayPacketSize> u3[2], const simd_fvec<RayPacketSize> plane_N[3],
    const simd_fvec<RayPacketSize> xform[16], derivatives_t<RayPacketSize> &out_der);

template class RendererSIMD<RayPacketDimX, RayPacketDimY>;
} // namespace Sse41
} // namespace Ray

#ifdef __GNUC__
#pragma GCC pop_options
#endif
