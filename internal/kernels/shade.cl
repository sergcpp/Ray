R"(

__constant sampler_t FBUF_SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float3 reflect(float3 V, float3 N) {
    return V - 2.0f * dot(V, N) * N;
}

float3 refract(float3 I, float3 N, float eta) {
    float cosi = dot(-I, N);
    float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
    float3 t = eta * I + ((eta * cosi - sqrt(fabs(cost2))) * N);
    return t * (float3)(cost2 > 0);
}

#define should_add_direct_light(pi) \
    (!(pi->settings.flags & SkipDirectLight) || pi->bounce > 2)

#define should_add_environment(pi) \
    (!(pi->settings.flags & NoBackground) || pi->bounce > 2)

#define should_consider_albedo(pi) \
    (!(pi->settings.flags & LightingOnly) || pi->bounce > 2)

#define use_uniform_sampling(pi) \
    ((pi->settings.flags & OutputSH) && pi->bounce <= 2)

float3 ComputeDirectLighting(const float3 P, const float3 N, const float3 B, const float3 plane_N,
                             __global const float *halton, const int hi, float rand_offset, float rand_offset2,
                             __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                             __global const mesh_t *meshes, __global const transform_t *transforms,
                             __global const uint *vtx_indices, __global const vertex_t *vertices,
                             __global const bvh_node_t *nodes, uint node_index, __global const tri_accel_t *tris,
                             __global const uint *tri_indices, __global const light_t *lights,
                             __global const uint *li_indices, uint light_node_index, __local uint *stack) {
    float3 col = (float3)(0.0f);    

    uint stack_size = 0;
    if (light_node_index != 0xffffffff) {
        stack[stack_size++] = light_node_index;
    }

    while (stack_size) {
        uint cur = stack[--stack_size];
        __global const bvh_node_t *n = &nodes[cur];

        if (!is_point_inside(P, n)) continue;

        if ((n->prim_index & LEAF_NODE_BIT) == 0) {
            stack[stack_size++] = n->left_child;
            stack[stack_size++] = n->right_child & RIGHT_CHILD_BITS;
        } else {
            uint prim_index = n->prim_index & PRIM_INDEX_BITS;
            for (uint i = prim_index; i < prim_index + n->prim_count; i++) {
                __global const light_t *l = &lights[li_indices[i]];

                float3 L = P - l->pos_and_radius.xyz;
                float distance = length(L);
                float d = max(distance - l->pos_and_radius.w, 0.0f);
                L /= distance;

                float _unused;
                const float z = fract(halton[hi + 0] + rand_offset, &_unused);

                const float dir = sqrt(z);
                const float phi = 2 * PI * fract(halton[hi + 1] + rand_offset2, &_unused);

                float cos_phi;
                const float sin_phi = sincos(phi, &cos_phi);

                float3 TT = cross(L, B);
                float3 BB = cross(L, TT);
                const float3 V = dir * sin_phi * BB + sqrt(1.0f - dir) * L + dir * cos_phi * TT;

                L = normalize(l->pos_and_radius.xyz + V * l->pos_and_radius.w - P);

                float denom = d / l->pos_and_radius.w + 1.0f;
                float atten = 1.0f / (denom * denom);

                atten = (atten - LIGHT_ATTEN_CUTOFF / l->col_and_brightness.w) / (1.0f - LIGHT_ATTEN_CUTOFF);
                atten = max(atten, 0.0f);

                float _dot1 = max(dot(L, N), 0.0f);
                float _dot2 = dot(L, l->dir_and_spot.xyz);

                if (_dot1 > FLT_EPS && _dot2 > l->dir_and_spot.w && (l->col_and_brightness.w * atten) > FLT_EPS) {
                    const float3 r_o = P + HIT_BIAS * plane_N;
                    const float3 r_d = L;

                    float _v = TraceOcclusionRay_WithLocalStack(r_o, r_d, distance, mesh_instances, mi_indices, meshes, transforms, nodes, node_index, tris, tri_indices, &stack[stack_size]);

                    col += l->col_and_brightness.xyz * _dot1 * _v * atten;
                }
            }
        }
    }

    return col;
}

void ComputeDerivatives(const float3 I, float t, const float3 do_dx, const float3 do_dy, const float3 dd_dx, const float3 dd_dy,
                        const float3 p1, const float3 p2, const float3 p3, const float3 n1, const float3 n2, const float3 n3,
                        const float2 u1, const float2 u2, const float2 u3, const float3 plane_N, derivatives_t *out_der) {
    // From 'Tracing Ray Differentials' [1999]

    float dot_I_N = dot(-I, plane_N);
    float inv_dot = fabs(dot_I_N) < FLT_EPS ? 0.0f : 1.0f/dot_I_N;
    float dt_dx = -dot(do_dx + t * dd_dx, plane_N) * inv_dot;
    float dt_dy = -dot(do_dy + t * dd_dy, plane_N) * inv_dot;
    
    out_der->do_dx = (do_dx + t * dd_dx) + dt_dx * I;
    out_der->do_dy = (do_dy + t * dd_dy) + dt_dy * I;
    out_der->dd_dx = dd_dx;
    out_der->dd_dy = dd_dy;

    // From 'Physically Based Rendering: ...' book

    const float2 duv13 = u1 - u3, duv23 = u2 - u3;
    const float3 dp13 = p1 - p3, dp23 = p2 - p3;

    const float det_uv = duv13.x * duv23.y - duv13.y * duv23.x;
    const float inv_det_uv = fabs(det_uv) > FLT_EPS ? 1.0f / det_uv : 0.0f;
    const float3 dpdu = (duv23.y * dp13 - duv13.y * dp23) * inv_det_uv;
    const float3 dpdv = (-duv23.x * dp13 + duv13.x * dp23) * inv_det_uv;

    float2 A[2] = { dpdu.xy, dpdv.xy };
    float2 Bx = out_der->do_dx.xy;
    float2 By = out_der->do_dy.xy;

    if (fabs(plane_N.x) > fabs(plane_N.y) && fabs(plane_N.x) > fabs(plane_N.z)) {
        A[0] = dpdu.yz; A[1] = dpdv.yz;
        Bx = out_der->do_dx.yz;  By = out_der->do_dy.yz;
    } else if (fabs(plane_N.y) > fabs(plane_N.z)) {
        A[0] = dpdu.xz; A[1] = dpdv.xz;
        Bx = out_der->do_dx.xz;  By = out_der->do_dy.xz;
    }

    const float det = A[0].x * A[1].y - A[1].x * A[0].y;
    const float inv_det = fabs(det) > FLT_EPS ? 1.0f / det : 0.0f;

    out_der->duv_dx = (float2)(A[0].x * Bx.x - A[0].y * Bx.y, A[1].x * Bx.x - A[1].y * Bx.y) * inv_det;
    out_der->duv_dy = (float2)(A[0].x * By.x - A[0].y * By.y, A[1].x * By.x - A[1].y * By.y) * inv_det;

    // Derivative for normal
    const float3 dn1 = n1 - n3, dn2 = n2 - n3;
    const float3 dndu = (duv23.y * dn1 - duv13.y * dn2) * inv_det_uv;
    const float3 dndv = (-duv23.x * dn1 + duv13.x * dn2) * inv_det_uv;

    out_der->dndx = dndu * out_der->duv_dx.x + dndv * out_der->duv_dx.y;
    out_der->dndy = dndu * out_der->duv_dy.x + dndv * out_der->duv_dy.y;

    out_der->ddn_dx = dot(out_der->dd_dx, plane_N) + dot(I, out_der->dndx);
    out_der->ddn_dy = dot(out_der->dd_dy, plane_N) + dot(I, out_der->dndy);
}

float4 ShadeSurface(const pass_info_t *pi, __global const float *halton,
                    __global const hit_data_t *prim_inters, __global const ray_packet_t *prim_rays,
                    __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                    __global const mesh_t *meshes, __global const transform_t *transforms,
                    __global const uint *vtx_indices, __global const vertex_t *vertices,
                    __global const bvh_node_t *nodes, uint node_index, 
                    __global const tri_accel_t *tris, __global const uint *tri_indices, 
                    const environment_t env, __global const material_t *materials, __global const texture_t *textures, __read_only image2d_array_t texture_atlas,
                    __global const light_t *lights, __global const uint *li_indices, uint light_node_index,
                    __local uint *stack, __global ray_packet_t *out_secondary_rays, __global int *out_secondary_rays_count) {

    __global const ray_packet_t *orig_ray = &prim_rays[pi->index];
    __global const hit_data_t *inter = &prim_inters[pi->index];

    const int2 px = (int2)(orig_ray->o.w, orig_ray->d.w);

    if (!inter->mask) {
        float4 env_col = 0.0f;
        if (should_add_environment(pi)) {
            env_col.xyz = SampleTextureLatlong_RGBE(texture_atlas, &textures[env.env_map], orig_ray->d.xyz).xyz;
            if (env.env_col_and_clamp.w > FLT_EPS) {
                env_col = min(env_col, env.env_col_and_clamp.w);
            }
            env_col.w = 1.0f;
        }
        return (float4)(env_col.xyz * orig_ray->c.xyz * env.env_col_and_clamp.xyz, env_col.w);
    }

    const float3 I = orig_ray->d.xyz;
    const float3 P = orig_ray->o.xyz + inter->t * I;

    __global const tri_accel_t *tri = &tris[inter->prim_index];
    __global const material_t *mat = &materials[tri->mi];
    __global const transform_t *tr = &transforms[mesh_instances[inter->obj_index].tr_index];
    __global const vertex_t *v1 = &vertices[vtx_indices[inter->prim_index * 3 + 0]];
    __global const vertex_t *v2 = &vertices[vtx_indices[inter->prim_index * 3 + 1]];
    __global const vertex_t *v3 = &vertices[vtx_indices[inter->prim_index * 3 + 2]];

    const float3 n1 = (float3)(v1->n[0], v1->n[1], v1->n[2]);
    const float3 n2 = (float3)(v2->n[0], v2->n[1], v2->n[2]);
    const float3 n3 = (float3)(v3->n[0], v3->n[1], v3->n[2]);

    const float2 u1 = (float2)(v1->t[0][0], v1->t[0][1]);
    const float2 u2 = (float2)(v2->t[0][0], v2->t[0][1]);
    const float2 u3 = (float2)(v3->t[0][0], v3->t[0][1]);

    const float _w = 1.0f - inter->u - inter->v;
    float3 N = fast_normalize(n1 * _w + n2 * inter->u + n3 * inter->v);
    float2 uvs = u1 * _w + u2 * inter->u + u3 * inter->v;

    //

    const float2 tex_atlas_size = (float2)(get_image_width(texture_atlas), get_image_height(texture_atlas));

    const float3 p1 = (float3)(v1->p[0], v1->p[1], v1->p[2]);
    const float3 p2 = (float3)(v2->p[0], v2->p[1], v2->p[2]);
    const float3 p3 = (float3)(v3->p[0], v3->p[1], v3->p[2]);

    const int _iw = tri->ci & TRI_W_BITS;
    float3 plane_N = (float3)(1.0f, tri->nu, tri->nv);
    if (_iw == 1) {
        plane_N = (float3)(tri->nu, 1.0f, tri->nv);
    } else if (_iw == 2) {
        plane_N = (float3)(tri->nu, tri->nv, 1.0f);
    }
    plane_N = fast_normalize(plane_N);
    if (tri->ci & TRI_INV_NORMAL_BIT) plane_N = -plane_N;

    plane_N = TransformNormal(plane_N, &tr->inv_xform);

    float backfacing_param = 0.0f;

    if (dot(plane_N, I) > 0.0f) {
        if (tri->back_mi == 0xffffffff) {
            return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            mat = &materials[tri->back_mi];
            plane_N = -plane_N;
            N = -N;
            backfacing_param = 1.0f;
        }
    }

    derivatives_t surf_der;
    ComputeDerivatives(I, inter->t, orig_ray->do_dx.xyz, orig_ray->do_dy.xyz,
                       orig_ray->dd_dx.xyz, orig_ray->dd_dy.xyz,
                       p1, p2, p3, n1, n2, n3, u1, u2, u3, plane_N, &surf_der);

    // used to randomize halton sequence among pixels
    int rand_hash = hash(pi->index),
        rand_hash2 = hash(rand_hash),
        rand_hash3 = hash(rand_hash2);
    float rand_offset = construct_float(rand_hash),
          rand_offset2 = construct_float(rand_hash2),
          rand_offset3 = construct_float(rand_hash3);

    const int hi = (pi->iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT + pi->bounce * 2;

    float _unused;
    float mix_rand = fract(halton[hi + 0] + rand_offset, &_unused);

    // resolve mix material
    while (mat->type == MixMaterial) {
        float mix_val = SampleTextureBilinear(texture_atlas, &textures[mat->textures[MAIN_TEXTURE]], uvs, 0).x * mat->strength;

        // shlick fresnel
        const float mix_ior = mix(mat->int_ior, mat->ext_ior, backfacing_param);

        float R0 = (orig_ray->c.w - mix_ior) / (orig_ray->c.w + mix_ior);
        R0 *= R0;

        float RR = R0 + (1.0f - R0) * native_powr(1.0f + dot(I, N), 5.0f);
        if (orig_ray->c.w > mix_ior) {
            float eta = orig_ray->c.w / mix_ior;
            float cosi = -dot(I, N);
            float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
            
            if (cost2 >= 0.0f) {
                float m = eta * cosi - sqrt(cost2);
                float3 V = eta * I + m * N;
                RR = R0 + (1.0f - R0) * native_powr(1.0f + dot(V, N), 5.0f);
            } else {
                RR = 1.0f;
            }
        }

        mix_val *= clamp(RR, 0.0f, 1.0f);

        if (mix_rand > mix_val) {
            mat = &materials[mat->textures[MIX_MAT1]];
            mix_rand = (mix_rand - mix_val) / (1.0f - mix_val);
        } else {
            mat = &materials[mat->textures[MIX_MAT2]];
            mix_rand = mix_rand / mix_val;
        }
    }

    //
    const float3 b1 = (float3)(v1->b[0], v1->b[1], v1->b[2]);
    const float3 b2 = (float3)(v2->b[0], v2->b[1], v2->b[2]);
    const float3 b3 = (float3)(v3->b[0], v3->b[1], v3->b[2]);

    float3 B = b1 * _w + b2 * inter->u + b3 * inter->v;
    float3 T = cross(B, N);

    float4 normals = 2 * SampleTextureBilinear(texture_atlas, &textures[mat->textures[NORMALS_TEXTURE]], uvs, 0) - 1;
    N = normals.x * B + normals.z * N + normals.y * T;
    
    N = TransformNormal(N, &tr->inv_xform);
    B = TransformNormal(B, &tr->inv_xform);
    T = TransformNormal(T, &tr->inv_xform);

    float4 albedo = SampleTextureAnisotropic(texture_atlas, &textures[mat->textures[MAIN_TEXTURE]], uvs, surf_der.duv_dx, surf_der.duv_dy);
    albedo = native_powr(albedo, 2.2f);
    albedo.xyz *= mat->main_color;

)" // workaround for 16k string literal limitation on msvc
R"(

    int diff_depth = as_int(orig_ray->do_dx.w);
    int gloss_depth = as_int(orig_ray->do_dy.w);
    int refr_depth = as_int(orig_ray->dd_dx.w);
    int transp_depth = as_int(orig_ray->dd_dy.w);
    int total_depth = diff_depth + gloss_depth + refr_depth + transp_depth;

    float3 col = 0.0f;

    // Evaluate materials
    if (mat->type == DiffuseMaterial) {
        if (should_add_direct_light(pi)) {
            col = ComputeDirectLighting(P, N, B, plane_N, halton, hi, rand_offset, rand_offset2,
                                        mesh_instances, mi_indices, meshes, transforms, vtx_indices, vertices,
                                        nodes, node_index, tris, tri_indices, lights, li_indices, light_node_index, stack);
            if (should_consider_albedo(pi)) {
                col *= albedo.xyz;
            }
        }

        if (diff_depth < pi->settings.max_diff_depth && total_depth < pi->settings.max_total_depth) {
            float _unused;
            const float u1 = fract(halton[hi + 0] + rand_offset, &_unused);
            const float u2 = fract(halton[hi + 1] + rand_offset2, &_unused);

            const float phi = 2 * PI * u2;
            float cos_phi;
            const float sin_phi = sincos(phi, &cos_phi);

            float3 V;
            float weight = 1.0f;

            if (use_uniform_sampling(pi)) {
                const float dir = sqrt(1.0f - u1 * u1);
                V = dir * sin_phi * B + u1 * N + dir * cos_phi * T;
                weight = 2.0f * u1;
            } else {
                const float dir = sqrt(u1);
                V = dir * sin_phi * B + sqrt(1.0f - u1) * N + dir * cos_phi * T;
            }

            ray_packet_t r;
            r.o = (float4)(P + HIT_BIAS * plane_N, (float)px.x);
            r.d = (float4)(V, (float)px.y);
            r.c = (float4)(orig_ray->c.xyz * weight, orig_ray->c.w);
            if (should_consider_albedo(pi)) {
                r.c.xyz *= albedo.xyz;
            }
            r.do_dx = (float4)(surf_der.do_dx, as_float(diff_depth + 1));
            r.do_dy = (float4)(surf_der.do_dy, orig_ray->do_dy.w);
            r.dd_dx.xyz = surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N);
            r.dd_dx.w = orig_ray->dd_dx.w;
            r.dd_dy.xyz = surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N);
            r.dd_dy.w = orig_ray->dd_dy.w;

            const float thr = max(r.c.x, max(r.c.y, r.c.z));
            const float p = fract(halton[hi + 0] + rand_offset3, &_unused);
            if (p < thr / RAY_TERM_THRES) {
                if (thr < RAY_TERM_THRES) r.c.xyz *= RAY_TERM_THRES / thr;
                const int index = atomic_inc(out_secondary_rays_count);
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == GlossyMaterial) {
        col = (float3)(0, 0, 0);

        if (gloss_depth < pi->settings.max_glossy_depth && total_depth < pi->settings.max_total_depth) {
            float3 V = reflect(I, N);

            float _unused;
            const float h = 1.0f - native_cos(0.5f * PI * mat->roughness * mat->roughness);
            const float z = h * fract(halton[hi + 0] + rand_offset, &_unused);

            const float dir = native_sqrt(z);
            const float phi = 2 * PI * fract(halton[hi + 1] + rand_offset2, &_unused);
            float cos_phi;
            const float sin_phi = sincos(phi, &cos_phi);

            float3 TT = cross(V, B);
            float3 BB = cross(V, TT);

            if (dot(V, plane_N) > 0) {
                V = dir * sin_phi * BB + native_sqrt(1.0f - dir) * V + dir * cos_phi * TT;
            } else {
                V = -dir * sin_phi * BB + native_sqrt(1.0f - dir) * V - dir * cos_phi * TT;
            }

            ray_packet_t r;
            r.o = (float4)(P + HIT_BIAS * plane_N, (float)px.x);
            r.d = (float4)(V, (float)px.y);
            r.c = orig_ray->c;
            if (should_consider_albedo(pi)) {
                r.c.xyz *= albedo.xyz;
            }
            r.do_dx = (float4)(surf_der.do_dx, orig_ray->do_dx.w);
            r.do_dy = (float4)(surf_der.do_dy, as_float(gloss_depth + 1));
            r.dd_dx.xyz = surf_der.dd_dx - 2 * (dot(I, plane_N) * surf_der.dndx + surf_der.ddn_dx * plane_N);
            r.dd_dx.w = orig_ray->dd_dx.w;
            r.dd_dy.xyz = surf_der.dd_dy - 2 * (dot(I, plane_N) * surf_der.dndy + surf_der.ddn_dy * plane_N);
            r.dd_dy.w = orig_ray->dd_dy.w;

            const float thr = max(r.c.x, max(r.c.y, r.c.z));
            const float p = fract(halton[hi + 0] + rand_offset3, &_unused);
            if (p < thr / RAY_TERM_THRES) {
                if (thr < RAY_TERM_THRES) r.c.xyz *= RAY_TERM_THRES / thr;
                const int index = atomic_inc(out_secondary_rays_count);
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == RefractiveMaterial) {
        col = (float3)(0, 0, 0);

        if (refr_depth < pi->settings.max_refr_depth && total_depth < pi->settings.max_total_depth) {
            const float ior = mix(mat->int_ior, mat->ext_ior, backfacing_param);

            float eta = orig_ray->c.w / ior;
            float cosi = -dot(I, N);
            float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
            float m = eta * cosi - sqrt(cost2);
            float3 V = eta * I + m * N;

            const float z = 1.0f - halton[hi + 0] * mat->roughness;
            const float temp = native_sqrt(1.0f - z * z);

            const float phi = halton[(((hash(hi) + pi->iteration) & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT + pi->bounce * 2) + 0] * 2 * PI;
            float cos_phi;
            const float sin_phi = sincos(phi, &cos_phi);

            float3 TT = normalize(cross(V, B));
            float3 BB = normalize(cross(V, TT));
            V = temp * sin_phi * BB + z * V + temp * cos_phi * TT;

            float k = (eta - eta * eta * dot(I, plane_N) / dot(V, plane_N));
            float dmdx = k * surf_der.ddn_dx;
            float dmdy = k * surf_der.ddn_dy;

            ray_packet_t r;
            r.o = (float4)(P + HIT_BIAS * I, (float)px.x);
            r.d = (float4)(V, (float)px.y);
            r.c = (float4)(orig_ray->c.xyz * z, ior);
            r.do_dx = (float4)(surf_der.do_dx, orig_ray->do_dx.w);
            r.do_dy = (float4)(surf_der.do_dy, orig_ray->do_dy.w);
            r.dd_dx.xyz = eta * surf_der.dd_dx - (m * surf_der.dndx + dmdx * plane_N);
            r.dd_dx.w = as_float(refr_depth + 1);
            r.dd_dy.xyz = eta * surf_der.dd_dy - (m * surf_der.dndy + dmdy * plane_N);
            r.dd_dy.w = orig_ray->dd_dy.w;

            float _unused;

            const float thr = max(r.c.x, max(r.c.y, r.c.z));
            const float p = fract(halton[hi + 0] + rand_offset3, &_unused);
            if (cost2 >= 0 && p < thr / RAY_TERM_THRES) {
                if (thr < RAY_TERM_THRES) r.c.xyz *= RAY_TERM_THRES / thr;
                const int index = atomic_inc(out_secondary_rays_count);
                out_secondary_rays[index] = r;
            }
        }
    } else if (mat->type == EmissiveMaterial) {
        col = mat->strength * albedo.xyz;
    } else if (mat->type == TransparentMaterial) {
        col = (float3)(0, 0, 0);

        if (transp_depth < pi->settings.max_transp_depth && total_depth < pi->settings.max_total_depth) {
            ray_packet_t r;
            r.o = (float4)(P + HIT_BIAS * I, (float)px.x);
            r.d = orig_ray->d;
            r.c = orig_ray->c;
            r.do_dx = (float4)(surf_der.do_dx, orig_ray->do_dx.w);
            r.do_dy = (float4)(surf_der.do_dy, orig_ray->do_dy.w);
            r.dd_dx = (float4)(surf_der.dd_dx, orig_ray->dd_dx.w);
            r.dd_dy = (float4)(surf_der.dd_dy, as_float(transp_depth + 1));

            float _unused;

            const float thr = max(r.c.x, max(r.c.y, r.c.z));
            const float p = fract(halton[hi + 0] + rand_offset3, &_unused);
            if (p < thr / RAY_TERM_THRES) {
                if (thr < RAY_TERM_THRES) r.c.xyz *= RAY_TERM_THRES / thr;
                const int index = atomic_inc(out_secondary_rays_count);
                out_secondary_rays[index] = r;
            }
        }
    }

    return (float4)(orig_ray->c.xyz * col, 1);
}

__kernel
void ShadePrimary(pass_info_t pi, __global const float *halton, int w,
                  __global const hit_data_t *prim_inters, __global const ray_packet_t *prim_rays,
                  __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                  __global const mesh_t *meshes, __global const transform_t *transforms,
                  __global const uint *vtx_indices, __global const vertex_t *vertices,
                  __global const bvh_node_t *nodes, uint node_index, 
                  __global const tri_accel_t *tris, __global const uint *tri_indices, 
                  const environment_t env, __global const material_t *materials, __global const texture_t *textures, __read_only image2d_array_t texture_atlas,
                  __global const light_t *lights, __global const uint *li_indices, uint light_node_index, __write_only image2d_t frame_buf,
                  __global ray_packet_t *out_secondary_rays, __global int *out_secondary_rays_count) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    
    __local uint shared_stack[MAX_STACK_SIZE * TRACE_GROUP_SIZE_X * TRACE_GROUP_SIZE_Y];
    __local uint *stack = &shared_stack[MAX_STACK_SIZE * (get_local_id(1) * TRACE_GROUP_SIZE_X + get_local_id(0))];

    pi.index = j * w + i;

    float4 res = ShadeSurface(&pi, halton, prim_inters, prim_rays,
                  mesh_instances, mi_indices, meshes, transforms, vtx_indices, vertices,
                  nodes, node_index, tris, tri_indices, env, materials, textures, texture_atlas,
                  lights, li_indices, light_node_index,
                  stack, out_secondary_rays, out_secondary_rays_count);

    write_imagef(frame_buf, (int2)(i, j), res);
}

__kernel
void ShadeSecondary(pass_info_t pi, __global const float *halton, __global const hit_data_t *prim_inters, __global const ray_packet_t *prim_rays,
                    __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices, __global const mesh_t *meshes, __global const transform_t *transforms,
                    __global const uint *vtx_indices, __global const vertex_t *vertices, __global const bvh_node_t *nodes, uint node_index, 
                    __global const tri_accel_t *tris, __global const uint *tri_indices, const environment_t env, __global const material_t *materials, __global const texture_t *textures, __read_only image2d_array_t texture_atlas,
                    __global const light_t *lights, __global const uint *li_indices, uint light_node_index,
                    __write_only image2d_t frame_buf, __read_only image2d_t frame_buf2, __global ray_packet_t *out_secondary_rays, __global int *out_secondary_rays_count) {
    const int index = get_global_id(0);

    __global const ray_packet_t *orig_ray = &prim_rays[index];

    const int2 px = (int2)(orig_ray->o.w, orig_ray->d.w);

    float4 col = read_imagef(frame_buf2, FBUF_SAMPLER, px);

    __local uint shared_stack[MAX_STACK_SIZE * TRACE_GROUP_SIZE_X * TRACE_GROUP_SIZE_Y];
    __local uint *stack = &shared_stack[MAX_STACK_SIZE * (get_local_id(1) * TRACE_GROUP_SIZE_X + get_local_id(0))];

    pi.index = index;

    float4 res = ShadeSurface(&pi, halton, prim_inters, prim_rays,
                  mesh_instances, mi_indices, meshes, transforms, vtx_indices, vertices,
                  nodes, node_index, tris, tri_indices, env, materials, textures, texture_atlas,
                  lights, li_indices, light_node_index,
                  stack, out_secondary_rays, out_secondary_rays_count);
    res.w = 0.0f;

    write_imagef(frame_buf, px, col + res);
}

)"