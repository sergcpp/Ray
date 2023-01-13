#include "RendererNEON.h"

#define NS Neon
#define USE_NEON
#include "RendererSIMD.h"
#undef USE_NEON
#undef NS

namespace Ray {
namespace Neon {
template void SortRays_CPU<RPSize>(ray_data_t<RPSize> *rays, simd_ivec<RPSize> *ray_masks, int &secondary_rays_count,
                                   const float root_min[3], const float cell_size[3], simd_ivec<RPSize> *hash_values,
                                   uint32_t *scan_values, ray_chunk_t *chunks, ray_chunk_t *chunks_temp);
template void SortRays_GPU<RPSize>(ray_data_t<RPSize> *rays, simd_ivec<RPSize> *ray_masks, int &secondary_rays_count,
                                   const float root_min[3], const float cell_size[3], simd_ivec<RPSize> *hash_values,
                                   int *head_flags, uint32_t *scan_values, ray_chunk_t *chunks,
                                   ray_chunk_t *chunks_temp, uint32_t *skeleton);

template bool Traverse_MacroTree_WithStack_ClosestHit<RPSize>(
    const simd_fvec<RPSize> ro[3], const simd_fvec<RPSize> rd[3], const simd_ivec<RPSize> &ray_mask,
    const bvh_node_t *nodes, uint32_t node_index, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
    const mesh_t *meshes, const transform_t *transforms, const tri_accel_t *tris, const uint32_t *tri_indices,
    hit_data_t<RPSize> &inter);
template bool Traverse_MacroTree_WithStack_ClosestHit<RPSize>(
    const simd_fvec<RPSize> ro[3], const simd_fvec<RPSize> rd[3], const simd_ivec<RPSize> &ray_mask,
    const mbvh_node_t *oct_nodes, uint32_t node_index, const mesh_instance_t *mesh_instances,
    const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms, const mtri_accel_t *mtris,
    const uint32_t *tri_indices, hit_data_t<RPSize> &inter);
template simd_ivec<RPSize> Traverse_MacroTree_WithStack_AnyHit<RPSize>(
    const simd_fvec<RPSize> ro[3], const simd_fvec<RPSize> rd[3], const simd_ivec<RPSize> &ray_mask,
    const bvh_node_t *nodes, uint32_t node_index, const mesh_instance_t *mesh_instances, const uint32_t *mi_indices,
    const mesh_t *meshes, const transform_t *transforms, const tri_accel_t *tris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, hit_data_t<RPSize> &inter);
template simd_ivec<RPSize> Traverse_MacroTree_WithStack_AnyHit<RPSize>(
    const simd_fvec<RPSize> ro[3], const simd_fvec<RPSize> rd[3], const simd_ivec<RPSize> &ray_mask,
    const mbvh_node_t *oct_nodes, uint32_t node_index, const mesh_instance_t *mesh_instances,
    const uint32_t *mi_indices, const mesh_t *meshes, const transform_t *transforms, const mtri_accel_t *mtris,
    const tri_mat_data_t *materials, const uint32_t *tri_indices, hit_data_t<RPSize> &inter);
template bool Traverse_MicroTree_WithStack_ClosestHit<RPSize>(const simd_fvec<RPSize> ro[3],
                                                              const simd_fvec<RPSize> rd[3],
                                                              const simd_ivec<RPSize> &ray_mask,
                                                              const bvh_node_t *nodes, uint32_t node_index,
                                                              const tri_accel_t *tris, const uint32_t *tri_indices,
                                                              int obj_index, hit_data_t<RPSize> &inter);
template bool Traverse_MicroTree_WithStack_ClosestHit<RPSize>(const float ro[3], const float rd[3],
                                                              const mbvh_node_t *mnodes, uint32_t node_index,
                                                              const mtri_accel_t *mtris, const uint32_t *tri_indices,
                                                              int &inter_prim_index, float &inter_t, float &inter_u,
                                                              float &inter_v);
template simd_ivec<RPSize> Traverse_MicroTree_WithStack_AnyHit<RPSize>(
    const simd_fvec<RPSize> ro[3], const simd_fvec<RPSize> rd[3], const simd_ivec<RPSize> &ray_mask,
    const bvh_node_t *nodes, uint32_t node_index, const tri_accel_t *tris, const tri_mat_data_t *materials,
    const uint32_t *tri_indices, int obj_index, hit_data_t<RPSize> &inter);
template int Traverse_MicroTree_WithStack_AnyHit<RPSize>(const float ro[3], const float rd[3],
                                                         const mbvh_node_t *mnodes, uint32_t node_index,
                                                         const mtri_accel_t *mtris, const tri_mat_data_t *materials,
                                                         const uint32_t *tri_indices, int &inter_prim_index,
                                                         float &inter_t, float &inter_u, float &inter_v);

template void SampleNearest<RPSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                    const simd_fvec<RPSize> uvs[2], const simd_fvec<RPSize> &lod,
                                    const simd_ivec<RPSize> &mask, simd_fvec<RPSize> out_rgba[4]);
template void SampleBilinear<RPSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                     const simd_fvec<RPSize> uvs[2], const simd_ivec<RPSize> &lod,
                                     const simd_ivec<RPSize> &mask, simd_fvec<RPSize> out_rgba[4]);
template void SampleTrilinear<RPSize>(const Ref::TexStorageBase *const textures[], uint32_t index,
                                      const simd_fvec<RPSize> uvs[2], const simd_fvec<RPSize> &lod,
                                      const simd_ivec<RPSize> &mask, simd_fvec<RPSize> out_rgba[4]);
template void SampleLatlong_RGBE<RPSize>(const Ref::TexStorageRGBA &storage, uint32_t index,
                                         const simd_fvec<RPSize> dir[3], float y_rotation,
                                         const simd_ivec<RPSize> &mask, simd_fvec<RPSize> out_rgb[3]);

template void ComputeDerivatives<RPSize>(const simd_fvec<RPSize> I[3], const simd_fvec<RPSize> &t,
                                         const simd_fvec<RPSize> do_dx[3], const simd_fvec<RPSize> do_dy[3],
                                         const simd_fvec<RPSize> dd_dx[3], const simd_fvec<RPSize> dd_dy[3],
                                         const simd_fvec<RPSize> p1[3], const simd_fvec<RPSize> p2[3],
                                         const simd_fvec<RPSize> p3[3], const simd_fvec<RPSize> n1[3],
                                         const simd_fvec<RPSize> n2[3], const simd_fvec<RPSize> n3[3],
                                         const simd_fvec<RPSize> u1[2], const simd_fvec<RPSize> u2[2],
                                         const simd_fvec<RPSize> u3[2], const simd_fvec<RPSize> plane_N[3],
                                         const simd_fvec<RPSize> xform[16], derivatives_t<RPSize> &out_der);

template void IntersectAreaLights<RPSize>(const ray_data_t<RPSize> &r, const simd_ivec<RPSize> &ray_mask,
                                          const light_t lights[], Span<const uint32_t> visible_lights,
                                          const transform_t transforms[], hit_data_t<RPSize> &inout_inter);

template class RendererSIMD<RPDimX, RPDimY>;

class Renderer : public RendererSIMD<RPDimX, RPDimY> {
  public:
    Renderer(const settings_t &s, ILog *log) : RendererSIMD(s, log) {}

    eRendererType type() const override { return RendererNEON; }
};

RendererBase *CreateRenderer(const settings_t &s, ILog *log) { return new Renderer(s, log); }
} // namespace Neon
} // namespace Ray

#ifdef __GNUC__
// #pragma GCC pop_options
#endif
