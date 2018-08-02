R"(

bool _bbox_test(const float3 o, const float3 inv_d, const float t, __global const float *bbox_min, __global const float *bbox_max) {
    float low = inv_d.x * (bbox_min[0] - o.x);
    float high = inv_d.x * (bbox_max[0] - o.x);
    float tmin = fmin(low, high);
    float tmax = fmax(low, high);

    low = inv_d.y * (bbox_min[1] - o.y);
    high = inv_d.y * (bbox_max[1] - o.y);
    tmin = fmax(tmin, fmin(low, high));
    tmax = fmin(tmax, fmax(low, high));

    low = inv_d.z * (bbox_min[2] - o.z);
    high = inv_d.z * (bbox_max[2] - o.z);
    tmin = fmax(tmin, fmin(low, high));
    tmax = fmin(tmax, fmax(low, high));

    return tmin <= tmax && tmin <= t && tmax > 0;
}

bool bbox_test(const float3 o, const float3 inv_d, const float t, __global const bvh_node_t *node) {
    return _bbox_test(o, inv_d, t, node->bbox[0], node->bbox[1]);
}

float3 safe_invert(const float3 v) {
    float3 inv_v = 1.0f / v;

    if (v.x <= FLT_EPS && v.x >= 0) {
        inv_v.x = FLT_MAX;
    } else if (v.x >= -FLT_EPS && v.x < 0) {
        inv_v.x = -FLT_MAX;
    }

    if (v.y <= FLT_EPS && v.y >= 0) {
        inv_v.y = FLT_MAX;
    } else if (v.y >= -FLT_EPS && v.y < 0) {
        inv_v.y = -FLT_MAX;
    }

    if (v.z <= FLT_EPS && v.z >= 0) {
        inv_v.z = FLT_MAX;
    } else if (v.z >= -FLT_EPS && v.z < 0) {
        inv_v.z = -FLT_MAX;
    }

    return inv_v;
}

#define near_child(rd, n)   \
    (rd)[(n)->space_axis] < 0 ? (n)->right_child : (n)->left_child

#define far_child(rd, n)    \
    (rd)[(n)->space_axis] < 0 ? (n)->left_child : (n)->right_child

void Traverse_MicroTree(const float3 r_o, const float3 r_d, const float3 inv_d, uint obj_index,
                        __global const bvh_node_t *nodes, uint node_index,
                        __global const tri_accel_t *tris, __global const uint *tri_indices, 
                        hit_data_t *inter) {
    const float *ro = (const float *)&r_o;
    const float *rd = (const float *)&r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].tri_count) {
        cur = near_child(rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];
        
        if (n->tri_count) {
            IntersectTris(r_o, r_d, tris, tri_indices, n->tri_index, n->tri_count, obj_index, inter);
            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(rd, n);
        uint far = far_child(rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(r_o, inv_d, inter->t, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }
}

float Traverse_MicroTree_Shadow(const float3 r_o, const float3 r_d, const float3 inv_d, 
                                __global const bvh_node_t *nodes, uint node_index,
                                __global const tri_accel_t *tris, __global const uint *tri_indices) {

    const float *ro = (const float *)&r_o;
    const float *rd = (const float *)&r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].tri_count) {
        cur = near_child(rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];
        
        if (n->tri_count) {
            if (IntersectTris_Shadow(r_o, r_d, tris, tri_indices, n->tri_index, n->tri_count) < 1)  {
                return 0;
            }
            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(rd, n);
        uint far = far_child(rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(r_o, inv_d, FLT_MAX, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }

    return 1;
}

void Traverse_MacroTree(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                        __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                        __global const mesh_t *meshes, __global const transform_t *transforms, 
                        __global const bvh_node_t *nodes, uint node_index, 
                        __global const tri_accel_t *tris, __global const uint *tri_indices,
                        hit_data_t *inter) {

    const float *orig_ro = (const float *)&orig_r_o;
    const float *orig_rd = (const float *)&orig_r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].tri_count) {
        cur = near_child(orig_rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];

        if (n->tri_count) {
            for (uint i = n->tri_index; i < n->tri_index + n->tri_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, inter->t, mi->bbox_min, mi->bbox_max)) continue;

				const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
				const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);
                
                Traverse_MicroTree(r_o, r_d, inv_d, mi_indices[i], nodes, m->node_index, tris, tri_indices, inter);
            }

            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(orig_rd, n);
        uint far = far_child(orig_rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(orig_r_o, orig_r_inv_d, inter->t, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }
}

float Traverse_MacroTree_Shadow(const float3 orig_r_o, const float3 orig_r_d, const float3 orig_r_inv_d, 
                                __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, 
                                __global const mesh_t *meshes, __global const transform_t *transforms, 
                                __global const bvh_node_t *nodes, uint node_index, 
                                __global const tri_accel_t *tris, __global const uint *tri_indices) {

    const float *orig_ro = (const float *)&orig_r_o;
    const float *orig_rd = (const float *)&orig_r_d;

    uint cur = node_index;
    uint last = node_index;

    if (!nodes[cur].tri_count) {
        cur = near_child(orig_rd, &nodes[cur]);
    }

    while (cur != 0xffffffff) {
        __global const bvh_node_t *n = &nodes[cur];

        if (n->tri_count) {
            for (uint i = n->tri_index; i < n->tri_index + n->tri_count; i++) {
                __global const mesh_instance_t *mi = &mesh_instances[mi_indices[i]];
                __global const mesh_t *m = &meshes[mi->mesh_index];
                __global const transform_t *tr = &transforms[mi->tr_index];

                if (!_bbox_test(orig_r_o, orig_r_inv_d, FLT_MAX, mi->bbox_min, mi->bbox_max)) continue;

				const float3 r_o = TransformPoint(orig_r_o, &tr->inv_xform);
				const float3 r_d = TransformDirection(orig_r_d, &tr->inv_xform);
                const float3 inv_d = safe_invert(r_d);

                if (Traverse_MicroTree_Shadow(r_o, r_d, inv_d, nodes, m->node_index, tris, tri_indices) < 1) {
                    return 0;
                }
            }

            last = cur; cur = n->parent;
            continue;
        }

        uint near = near_child(orig_rd, n);
        uint far = far_child(orig_rd, n);

        if (last == far) {
            last = cur; cur = n->parent;
            continue;
        }

        uint try_child = (last == n->parent) ? near : far;
        if (bbox_test(orig_r_o, orig_r_inv_d, FLT_MAX, &nodes[try_child])) {
            last = cur; cur = try_child;
        } else {
            if (try_child == near) {
                last = near;
            } else {
                last = cur; cur = n->parent;
            }
        }
    }

    return 1;
}

#undef near_child
#undef far_child

)"