#pragma once

#define NS Avx2
#define USE_AVX2
#include "RendererSIMD.h"
#undef USE_AVX2
#undef NS

namespace Ray {
namespace Avx2 {
const int RayPacketDimX = 4;
const int RayPacketDimY = 2;
const int RayPacketSize = RayPacketDimX * RayPacketDimY;

extern template void SortRays_CPU<RayPacketSize>(ray_data_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks,
                                                 int &secondary_rays_count, const float root_min[3],
                                                 const float cell_size[3], simd_ivec<RayPacketSize> *hash_values,
                                                 uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
extern template void SortRays_GPU<RayPacketSize>(ray_data_t<RayPacketSize> *rays, simd_ivec<RayPacketSize> *ray_masks,
                                                 int &secondary_rays_count, const float root_min[3],
                                                 const float cell_size[3], simd_ivec<RayPacketSize> *hash_values,
                                                 int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks,
                                                 ray_chunk_t *chunks_temp, uint32_t *skeleton);

extern template bool Traverse_MacroTree_WithStack_ClosestHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const tri_accel_t *tris, const uint32_t *tri_indices,
    hit_data_t<RayPacketSize> &inter);
extern template bool Traverse_MacroTree_WithStack_ClosestHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const mbvh_node_t *oct_nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const mtri_accel_t *mtris, const uint32_t *tri_indices,
    hit_data_t<RayPacketSize> &inter);
extern template simd_ivec<RayPacketSize> Traverse_MacroTree_WithStack_AnyHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const tri_accel_t *tris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
extern template simd_ivec<RayPacketSize> Traverse_MacroTree_WithStack_AnyHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const mbvh_node_t *oct_nodes, uint32_t node_index,
    const mesh_instance_t *mesh_instances, const uint32_t *mi_indices, const mesh_t *meshes,
    const transform_t *transforms, const mtri_accel_t *mtris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<RayPacketSize> &inter);
extern template bool Traverse_MicroTree_WithStack_ClosestHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris,
    const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
extern template bool Traverse_MicroTree_WithStack_ClosestHit<RayPacketSize>(
    const float ro[3], const float rd[3], int i, const mbvh_node_t *oct_nodes, uint32_t node_index,
    const mtri_accel_t *mtris, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
extern template simd_ivec<RayPacketSize> Traverse_MicroTree_WithStack_AnyHit<RayPacketSize>(
    const simd_fvec<RayPacketSize> ro[3], const simd_fvec<RayPacketSize> rd[3],
    const simd_ivec<RayPacketSize> &ray_mask, const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris,
    const tri_mat_data_t *materials, const uint32_t *tri_indices, int obj_index, hit_data_t<RayPacketSize> &inter);
extern template bool Traverse_MicroTree_WithStack_AnyHit(const float ro[3], const float rd[3], int i,
                                                         const mbvh_node_t *oct_nodes, uint32_t node_index,
                                                         const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                                         const uint32_t *tri_indices, int obj_index,
                                                         hit_data_t<RayPacketSize> &inter);

extern template void SampleNearest<RayPacketSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                                  const simd_fvec<RayPacketSize> uvs[2],
                                                  const simd_fvec<RayPacketSize> &lod,
                                                  const simd_ivec<RayPacketSize> &mask,
                                                  simd_fvec<RayPacketSize> out_rgba[4]);
extern template void SampleBilinear<RayPacketSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                                   const simd_fvec<RayPacketSize> uvs[2],
                                                   const simd_ivec<RayPacketSize> &lod,
                                                   const simd_ivec<RayPacketSize> &mask,
                                                   simd_fvec<RayPacketSize> out_rgba[4]);
extern template void SampleTrilinear<RayPacketSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                                    const simd_fvec<RayPacketSize> uvs[2],
                                                    const simd_fvec<RayPacketSize> &lod,
                                                    const simd_ivec<RayPacketSize> &mask,
                                                    simd_fvec<RayPacketSize> out_rgba[4]);
extern template void SampleLatlong_RGBE<RayPacketSize>(const Ref::TexStorageRGBA &storage, uint32_t index,
                                                       const simd_fvec<RayPacketSize> dir[3], float y_rotation,
                                                       const simd_ivec<RayPacketSize> &mask,
                                                       simd_fvec<RayPacketSize> out_rgb[3]);

extern template void ComputeDerivatives<RayPacketSize>(
    const simd_fvec<RayPacketSize> I[3], const simd_fvec<RayPacketSize> &t, const simd_fvec<RayPacketSize> do_dx[3],
    const simd_fvec<RayPacketSize> do_dy[3], const simd_fvec<RayPacketSize> dd_dx[3],
    const simd_fvec<RayPacketSize> dd_dy[3], const simd_fvec<RayPacketSize> p1[3], const simd_fvec<RayPacketSize> p2[3],
    const simd_fvec<RayPacketSize> p3[3], const simd_fvec<RayPacketSize> n1[3], const simd_fvec<RayPacketSize> n2[3],
    const simd_fvec<RayPacketSize> n3[3], const simd_fvec<RayPacketSize> u1[2], const simd_fvec<RayPacketSize> u2[2],
    const simd_fvec<RayPacketSize> u3[2], const simd_fvec<RayPacketSize> plane_N[3],
    const simd_fvec<RayPacketSize> xform[16], derivatives_t<RayPacketSize> &out_der);

extern template class RendererSIMD<RayPacketDimX, RayPacketDimY>;

class Renderer : public RendererSIMD<RayPacketDimX, RayPacketDimY> {
  public:
    Renderer(const settings_t &s, ILog *log) : RendererSIMD(s, log) {}

    eRendererType type() const override { return RendererAVX2; }
};
} // namespace Avx2
} // namespace Ray