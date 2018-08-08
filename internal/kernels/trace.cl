R"(

float TraceOcclusionRay(const float3 ro, const float3 rd, 
                        __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                        __global const mesh_t *meshes, __global const transform_t *transforms,
                        __global const bvh_node_t *nodes, uint node_index,
                        __global const tri_accel_t *tris, __global const uint *tri_indices) {
    const float3 inv_d = safe_invert(rd);

    return Traverse_MacroTree_Occlusion_Stackless(ro, rd, inv_d, mesh_instances, mi_indices, 
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

	const float4 orig_r_o = rays[index].o;
	const float4 orig_r_d = rays[index].d;
    const float3 orig_inv_d = safe_invert(orig_r_d.xyz);

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r_o.w, orig_r_d.w);

    Traverse_MacroTree_WithStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                 nodes, node_index, tris, tri_indices, &inter);

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

    Traverse_MacroTreeImg_WithStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
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

    const float4 orig_r_o = rays[index].o;
	const float4 orig_r_d = rays[index].d;
    const float3 orig_inv_d = safe_invert(orig_r_d.xyz);

    hit_data_t inter;
    inter.mask = 0;
    inter.t = FLT_MAX;
    inter.ray_id = (float2)(orig_r_o.w, orig_r_d.w);

    Traverse_MacroTree_WithStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                 nodes, node_index, tris, tri_indices, &inter);

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

    Traverse_MacroTreeImg_WithStack(orig_r_o.xyz, orig_r_d.xyz, orig_inv_d, mesh_instances, mi_indices, meshes, transforms,
                                    nodes, node_index, tris, tri_indices, &inter);

    out_prim_inters[index] = inter;
}

)"