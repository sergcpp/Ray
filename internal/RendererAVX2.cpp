#include "RendererAVX2.h"

namespace Ray {
namespace Avx2 {
template void GeneratePrimaryRays<RayPacketDimX, RayPacketDimY>(const int iteration, const camera_t &cam,
                                                                const rect_t &r, int w, int h, const float *halton,
                                                                aligned_vector<ray_data_t<RayPacketSize>> &out_rays,
                                                                aligned_vector<simd_ivec<RayPacketSize>> &out_masks);
template void SampleMeshInTextureSpace<RayPacketDimX, RayPacketDimY>(
    int iteration, int obj_index, int uv_layer, const mesh_t &mesh, const transform_t &tr, const uint32_t *vtx_indices,
    const vertex_t *vertices, const rect_t &r, int w, int h, const float *halton,
    aligned_vector<ray_data_t<RayPacketSize>> &out_rays, aligned_vector<hit_data_t<RayPacketSize>> &out_inters);

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

template void SampleNearest<RayPacketSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                           const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod,
                                           const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleBilinear<RayPacketSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                            const simd_fvec<RayPacketSize> uvs[2], const simd_ivec<RayPacketSize> &lod,
                                            const simd_ivec<RayPacketSize> &mask, simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleTrilinear<RayPacketSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                             const simd_fvec<RayPacketSize> uvs[2], const simd_fvec<RayPacketSize> &lod,
                                             const simd_ivec<RayPacketSize> &mask,
                                             simd_fvec<RayPacketSize> out_rgba[4]);
template void SampleLatlong_RGBE<RayPacketSize>(const Ref::TexStorageRGBA &storage, uint32_t index,
                                                const simd_fvec<RayPacketSize> dir[3], float y_rotation,
                                                const simd_ivec<RayPacketSize> &mask,
                                                simd_fvec<RayPacketSize> out_rgb[3]);

template simd_fvec<RayPacketSize> ComputeVisibility<RayPacketSize>(
    const simd_fvec<RayPacketSize> p[3], const simd_fvec<RayPacketSize> d[3], simd_fvec<RayPacketSize> dist,
    const simd_ivec<RayPacketSize> &mask, const float rand_val, const simd_ivec<RayPacketSize> &rand_hash2,
    const scene_data_t &sc, uint32_t node_index, const Ref::TexStorageBase *const textures[]);

template void ComputeDerivatives<RayPacketSize>(
    const simd_fvec<RayPacketSize> I[3], const simd_fvec<RayPacketSize> &t, const simd_fvec<RayPacketSize> do_dx[3],
    const simd_fvec<RayPacketSize> do_dy[3], const simd_fvec<RayPacketSize> dd_dx[3],
    const simd_fvec<RayPacketSize> dd_dy[3], const simd_fvec<RayPacketSize> p1[3], const simd_fvec<RayPacketSize> p2[3],
    const simd_fvec<RayPacketSize> p3[3], const simd_fvec<RayPacketSize> n1[3], const simd_fvec<RayPacketSize> n2[3],
    const simd_fvec<RayPacketSize> n3[3], const simd_fvec<RayPacketSize> u1[2], const simd_fvec<RayPacketSize> u2[2],
    const simd_fvec<RayPacketSize> u3[2], const simd_fvec<RayPacketSize> plane_N[3],
    const simd_fvec<RayPacketSize> xform[16], derivatives_t<RayPacketSize> &out_der);

template void ShadeSurface<RayPacketSize>(const simd_ivec<RayPacketSize> &index, const pass_settings_t &ps,
                                          const float *halton, const hit_data_t<RayPacketSize> &inter,
                                          const ray_data_t<RayPacketSize> &ray, const scene_data_t &sc,
                                          uint32_t node_index, const Ref::TexStorageBase *const textures[],
                                          simd_fvec<RayPacketSize> out_rgba[4],
                                          simd_ivec<RayPacketSize> out_secondary_masks[],
                                          ray_data_t<RayPacketSize> out_secondary_rays[], int *out_secondary_rays_count,
                                          simd_ivec<RayPacketSize> out_shadow_masks[],
                                          shadow_ray_t<RayPacketSize> out_shadow_rays[], int *out_shadow_rays_count);

template class RendererSIMD<RayPacketDimX, RayPacketDimY>;
} // namespace Avx2
} // namespace Ray
