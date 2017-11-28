R"(

float TraceShadowRay(const ray_packet_t *orig_r, 
                     __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                     __global const mesh_t *meshes, __global const transform_t *transforms,
                     __global const bvh_node_t *nodes, uint node_index,
                     __global const tri_accel_t *tris, __global const uint *tri_indices) {

    const float3 orig_inv_d = safe_invert(orig_r->d.xyz);
    const float *orig_rinv_d = (const float *)&orig_inv_d;

    return Traverse_MacroTree_Shadow(orig_r, orig_rinv_d, mesh_instances, mi_indices, 
                                     meshes, transforms, nodes, node_index, tris, tri_indices);
}

__kernel
void TracePrimaryRays(__global const ray_packet_t *rays, int w, 
                      __global const mesh_instance_t *mesh_instances,
                      __global const uint *mi_indices, 
                      __global const mesh_t *meshes, __global const transform_t *transforms,
                      __global const bvh_node_t *nodes, uint node_index,
                      __global const tri_accel_t *tris, __global const uint *tri_indices, 
                      __global hit_data_t *out_prim_inters) {

    const int index = get_global_id(1) * w + get_global_id(0);

    const ray_packet_t orig_r = rays[index];
    const float3 orig_inv_d = safe_invert(orig_r.d.xyz);
    const float *orig_rinv_d = (const float *)&orig_inv_d;

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r.o.w, orig_r.d.w);

    Traverse_MacroTree(&orig_r, orig_rinv_d, mesh_instances, mi_indices, meshes, transforms,
                       nodes, node_index, tris, tri_indices, &inter);

    out_prim_inters[index] = inter;
}

__kernel
void TraceSecondaryRays(__global const ray_packet_t *rays,
                      __global const mesh_instance_t *mesh_instances,
                      __global const uint *mi_indices, 
                      __global const mesh_t *meshes, __global const transform_t *transforms,
                      __global const bvh_node_t *nodes, uint node_index,
                      __global const tri_accel_t *tris, __global const uint *tri_indices, 
                      __global hit_data_t *out_prim_inters) {

    const int index = get_global_id(0);

    const ray_packet_t orig_r = rays[index];
    const float3 orig_inv_d = safe_invert(orig_r.d.xyz);
    const float *orig_rinv_d = (const float *)&orig_inv_d;

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r.o.w, orig_r.d.w);

    Traverse_MacroTree(&orig_r, orig_rinv_d, mesh_instances, mi_indices, meshes, transforms,
                       nodes, node_index, tris, tri_indices, &inter);

    out_prim_inters[index] = inter;
}

)"