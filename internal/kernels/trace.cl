R"(

#ifdef USE_STACKLESS_BVH_TRAVERSAL
float TraceOcclusionRay_Stackless(const float3 ro, const float3 rd, float max_dist,
                                  __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                                  __global const mesh_t *meshes, __global const transform_t *transforms,
                                  __global const bvh_node_t *nodes, uint node_index,
                                  __global const tri_accel_t *tris, __global const uint *tri_indices) {
    const float3 inv_d = safe_invert(rd);

    return Traverse_MacroTree_Occlusion_Stackless(ro, rd, inv_d, max_dist, mesh_instances, mi_indices, 
                                                  meshes, transforms, nodes, node_index, tris, tri_indices);
}
#endif

float TraceOcclusionRay_WithLocalStack(const float3 ro, const float3 rd, float max_dist,
                                       __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                                       __global const mesh_t *meshes, __global const transform_t *transforms,
                                       __global const bvh_node_t *nodes, uint node_index,
                                       __global const tri_accel_t *tris, __global const uint *tri_indices, __local uint *stack) {
    const float3 inv_d = safe_invert(rd);

    return Traverse_MacroTree_Occlusion_WithLocalStack(ro, rd, inv_d, max_dist, mesh_instances, mi_indices, 
                                                       meshes, transforms, nodes, node_index, tris, tri_indices, stack);
}

float TraceOcclusionRay_WithPrivateStack(const float3 ro, const float3 rd, float max_dist,
                                         __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                                         __global const mesh_t *meshes, __global const transform_t *transforms,
                                         __global const bvh_node_t *nodes, uint node_index,
                                         __global const tri_accel_t *tris, __global const uint *tri_indices, uint *stack) {
    const float3 inv_d = safe_invert(rd);

    return Traverse_MacroTree_Occlusion_WithPrivateStack(ro, rd, inv_d, max_dist, mesh_instances, mi_indices, 
                                                         meshes, transforms, nodes, node_index, tris, tri_indices, stack);
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

    const float4 orig_r_o = rays[index].o;
    const float4 orig_r_d = rays[index].d;
    const float3 orig_inv_d = safe_invert(orig_r_d.xyz);

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r_o.w, orig_r_d.w);

    __local uint shared_stack[MAX_STACK_SIZE * TRACE_GROUP_SIZE_X * TRACE_GROUP_SIZE_Y];
    __local uint *stack = &shared_stack[MAX_STACK_SIZE * (get_local_id(1) * TRACE_GROUP_SIZE_X + get_local_id(0))];

    if (node_index != 0xffffffff) {
        Traverse_MacroTree_WithLocalStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                          nodes, node_index, tris, tri_indices, stack, &inter);
    }

    out_prim_inters[index] = inter;
}

__kernel
void TracePrimaryRaysImg(__global const ray_packet_t *rays, int w, 
                      __global const mesh_instance_t *mesh_instances,
                      __global const uint *mi_indices, 
                      __global const mesh_t *meshes, __global const transform_t *transforms,
                      __read_only image1d_buffer_t nodes, uint node_index,
                      __global const tri_accel_t *tris, __global const uint *tri_indices, 
                      __global hit_data_t *out_prim_inters) {

    const int index = get_global_id(1) * w + get_global_id(0);

    const float4 orig_r_o = rays[index].o;
    const float4 orig_r_d = rays[index].d;
    const float3 orig_inv_d = safe_invert(orig_r_d.xyz);

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r_o.w, orig_r_d.w);

    __local uint shared_stack[MAX_STACK_SIZE * TRACE_GROUP_SIZE_X * TRACE_GROUP_SIZE_Y];
    __local uint *stack = &shared_stack[MAX_STACK_SIZE * (get_local_id(1) * TRACE_GROUP_SIZE_X + get_local_id(0))];

    if (node_index != 0xffffffff) {
        Traverse_MacroTreeImg_WithLocalStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                             nodes, node_index, tris, tri_indices, stack, &inter);
    }

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

    const float4 orig_r_o = rays[index].o;
    const float4 orig_r_d = rays[index].d;
    const float3 orig_inv_d = safe_invert(orig_r_d.xyz);

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r_o.w, orig_r_d.w);

    __local uint shared_stack[MAX_STACK_SIZE * TRACE_GROUP_SIZE_X * TRACE_GROUP_SIZE_Y];
    __local uint *stack = &shared_stack[MAX_STACK_SIZE * (get_local_id(1) * TRACE_GROUP_SIZE_X + get_local_id(0))];

    if (node_index != 0xffffffff) {
        Traverse_MacroTree_WithLocalStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                          nodes, node_index, tris, tri_indices, stack, &inter);
    }

    out_prim_inters[index] = inter;
}

__kernel
void TraceSecondaryRaysImg(__global const ray_packet_t *rays,
                           __global const mesh_instance_t *mesh_instances,
                           __global const uint *mi_indices, 
                           __global const mesh_t *meshes, __global const transform_t *transforms,
                           __read_only image1d_buffer_t nodes, uint node_index,
                           __global const tri_accel_t *tris, __global const uint *tri_indices, 
                           __global hit_data_t *out_prim_inters) {

    const int index = get_global_id(0);

    const float4 orig_r_o = rays[index].o;
    const float4 orig_r_d = rays[index].d;
    const float3 orig_inv_d = safe_invert(orig_r_d.xyz);

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r_o.w, orig_r_d.w);

    __local uint shared_stack[MAX_STACK_SIZE * TRACE_GROUP_SIZE_X * TRACE_GROUP_SIZE_Y];
    __local uint *stack = &shared_stack[MAX_STACK_SIZE * (get_local_id(1) * TRACE_GROUP_SIZE_X + get_local_id(0))];

    if (node_index != 0xffffffff) {
        Traverse_MacroTreeImg_WithLocalStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                             nodes, node_index, tris, tri_indices, stack, &inter);
    }

    out_prim_inters[index] = inter;
}

)"