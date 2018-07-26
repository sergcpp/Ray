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

float4 ShadeSurface(const int index, const int iteration, __global const float *halton,
                    __global const hit_data_t *prim_inters, __global const ray_packet_t *prim_rays,
                    __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                    __global const mesh_t *meshes, __global const transform_t *transforms,
                    __global const uint *vtx_indices, __global const vertex_t *vertices,
                    __global const bvh_node_t *nodes, uint node_index, 
                    __global const tri_accel_t *tris, __global const uint *tri_indices, 
                    const environment_t env, __global const material_t *materials, __global const texture_t *textures, __read_only image2d_array_t texture_atlas,
                    __global ray_packet_t *out_secondary_rays, __global int *out_secondary_rays_count) {

    __global const ray_packet_t *orig_ray = &prim_rays[index];
    __global const hit_data_t *inter = &prim_inters[index];

    const int2 px = (int2)(orig_ray->o.w, orig_ray->d.w);

    if (!inter->mask) {
        // TODO: sample environment map or spherical garm.
        return (float4)(orig_ray->c.xyz * env.sky_col, 1);
    }

    const float3 I = orig_ray->d.xyz;
    const float3 P = orig_ray->o.xyz + inter->t * I;

    __global const tri_accel_t *tri = &tris[inter->prim_index];
    __global const material_t *mat = &materials[tri->mi];
    __global const vertex_t *v1 = &vertices[vtx_indices[inter->prim_index * 3 + 0]];
    __global const vertex_t *v2 = &vertices[vtx_indices[inter->prim_index * 3 + 1]];
    __global const vertex_t *v3 = &vertices[vtx_indices[inter->prim_index * 3 + 2]];

    const float3 n1 = (float3)(v1->n[0], v1->n[1], v1->n[2]);
    const float3 n2 = (float3)(v2->n[0], v2->n[1], v2->n[2]);
    const float3 n3 = (float3)(v3->n[0], v3->n[1], v3->n[2]);

    const float2 u1 = (float2)(v1->t0[0], v1->t0[1]);
    const float2 u2 = (float2)(v2->t0[0], v2->t0[1]);
    const float2 u3 = (float2)(v3->t0[0], v3->t0[1]);

    const float _w = 1.0f - inter->u - inter->v;
    float3 N = fast_normalize(n1 * _w + n2 * inter->u + n3 * inter->v);
    float2 uvs = u1 * _w + u2 * inter->u + u3 * inter->v;

    //////////////////////////////////////////

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

    float dot_I_N = dot(-I, plane_N);

    // From 'Tracing Ray Differentials' [1999]

    float inv_dot = fabs(dot_I_N) < FLT_EPS ? 0.0f : 1.0f/dot_I_N;
    float dt_dx = -dot(orig_ray->do_dx + inter->t * orig_ray->dd_dx, plane_N) * inv_dot;
    float dt_dy = -dot(orig_ray->do_dy + inter->t * orig_ray->dd_dy, plane_N) * inv_dot;
    
    const float3 do_dx = (orig_ray->do_dx + inter->t * orig_ray->dd_dx) + dt_dx * I;
    const float3 do_dy = (orig_ray->do_dy + inter->t * orig_ray->dd_dy) + dt_dy * I;
    const float3 dd_dx = orig_ray->dd_dx;
    const float3 dd_dy = orig_ray->dd_dy;

    ////////////////////////////////////////////////////////

    // From 'Physically Based Rendering: ...' book

    const float2 duv13 = u1 - u3, duv23 = u2 - u3;
    const float3 dp13 = p1 - p3, dp23 = p2 - p3;

    const float det_uv = duv13.x * duv23.y - duv13.y * duv23.x;
    const float inv_det_uv = fabs(det_uv) > FLT_EPS ? 1.0f / det_uv : 0.0f;
    const float3 dpdu = (duv23.y * dp13 - duv13.y * dp23) * inv_det_uv;
    const float3 dpdv = (-duv23.x * dp13 + duv13.x * dp23) * inv_det_uv;

    float2 A[2] = { dpdu.xy, dpdv.xy };
    float2 Bx = do_dx.xy;
    float2 By = do_dy.xy;

    if (fabs(plane_N.x) > fabs(plane_N.y) && fabs(plane_N.x) > fabs(plane_N.z)) {
        A[0] = dpdu.yz; A[1] = dpdv.yz;
        Bx = do_dx.yz;  By = do_dy.yz;
    } else if (fabs(plane_N.y) > fabs(plane_N.z)) {
        A[0] = dpdu.xz; A[1] = dpdv.xz;
        Bx = do_dx.xz;  By = do_dy.xz;
    }

    const float det = A[0].x * A[1].y - A[1].x * A[0].y;
    const float inv_det = fabs(det) > FLT_EPS ? 1.0f / det : 0.0f;

    const float2 duv_dx = (float2)(A[0].x * Bx.x - A[0].y * Bx.y, A[1].x * Bx.x - A[1].y * Bx.y) * inv_det;
    const float2 duv_dy = (float2)(A[0].x * By.x - A[0].y * By.y, A[1].x * By.x - A[1].y * By.y) * inv_det;

    ////////////////////////////////////////////////////////

    // used to randomize halton sequence among pixels
    int rand_hash = hash(index), rand_hash2;
    float rand_offset = construct_float(rand_hash), rand_offset2;

    const int hi = iteration & (HaltonSeqLen - 1);

    // resolve mix material
    while (mat->type == MixMaterial) {
        const float4 mix = SampleTextureBilinear(texture_atlas, &textures[mat->textures[MAIN_TEXTURE]], uvs, 0) * mat->strength;

        float _unused;
        const float r = fract(halton[hi * 2] + rand_offset, &_unused);

        rand_hash = hash(rand_hash);
        rand_offset = construct_float(rand_hash);

        // shlick fresnel
        float RR = mat->fresnel + (1.0f - mat->fresnel) * native_powr(1.0f + dot(I, N), 5.0f);
        RR = clamp(RR, 0.0f, 1.0f);

        mat = (r * RR < mix.x) ? &materials[mat->textures[MIX_MAT1]] : &materials[mat->textures[MIX_MAT2]];
    }

    rand_hash2 = hash(rand_hash);
    rand_offset2 = construct_float(rand_hash2);

    // Derivative for normal
    const float3 dn1 = n1 - n3, dn2 = n2 - n3;
    const float3 dndu = (duv23.y * dn1 - duv13.y * dn2) * inv_det_uv;
    const float3 dndv = (-duv23.x * dn1 + duv13.x * dn2) * inv_det_uv;

    const float3 dndx = dndu * duv_dx.x + dndv * duv_dx.y;
    const float3 dndy = dndu * duv_dy.x + dndv * duv_dy.y;

    const float ddn_dx = dot(dd_dx, plane_N) + dot(I, dndx);
    const float ddn_dy = dot(dd_dy, plane_N) + dot(I, dndy);

    //
    const float3 b1 = (float3)(v1->b[0], v1->b[1], v1->b[2]);
    const float3 b2 = (float3)(v2->b[0], v2->b[1], v2->b[2]);
    const float3 b3 = (float3)(v3->b[0], v3->b[1], v3->b[2]);

    float3 B = b1 * _w + b2 * inter->u + b3 * inter->v;
    float3 T = cross(B, N);

    float4 normals = SampleTextureBilinear(texture_atlas, &textures[mat->textures[NORMALS_TEXTURE]], uvs, 0);

    normals = 2.0f * normals - 1.0f;

    N = normals.x * B + normals.z * N + normals.y * T;
    
    //////////////////////////////////////////

    __global const transform_t *tr = &transforms[mesh_instances[inter->obj_index].tr_index];

    N = TransformNormal(&N, &tr->inv_xform);
    B = TransformNormal(&B, &tr->inv_xform);
    T = TransformNormal(&T, &tr->inv_xform);

    float4 albedo = SampleTextureAnisotropic(texture_atlas, &textures[mat->textures[MAIN_TEXTURE]], uvs, duv_dx, duv_dy);
    albedo.xyz *= mat->main_color;
    albedo = native_powr(albedo, 2.2f);

    float3 col;

    // Generate secondary ray
    if (mat->type == DiffuseMaterial) { 
        float k = dot(N, env.sun_dir);

        float v = 1;
        if (k > FLT_EPS) {
            float _unused;
            const float z = 1.0f - fract(halton[hi * 2] + rand_offset, &_unused) * env.sun_softness;
            const float temp = native_sqrt(1.0f - z * z);

            const float phi = 2 * PI * fract(halton[hi * 2 + 1] + rand_offset2, &_unused);
            float cos_phi;
            const float sin_phi = sincos(phi, &cos_phi);

            float3 TT = cross(env.sun_dir, B);
            float3 BB = cross(env.sun_dir, TT);
            const float3 V = temp * sin_phi * BB + z * env.sun_dir + temp * cos_phi * TT;

            ray_packet_t r;
            r.o = (float4)(P + HIT_BIAS * N, 0);
            r.d = (float4)(V, 0);
            v = TraceShadowRay(&r, mesh_instances, mi_indices, meshes, transforms, nodes, node_index, tris, tri_indices);
        }

        k = clamp(k, 0.0f, 1.0f);

        col = albedo.xyz * env.sun_col * v * k;

        float _unused;
        const float z = fract(halton[hi * 2] + rand_offset, &_unused);

        const float dir = native_sqrt(z);
        const float phi = 2 * PI * fract(halton[hi * 2 + 1] + rand_offset2, &_unused);
        float cos_phi;
        const float sin_phi = sincos(phi, &cos_phi);

        const float3 V = dir * sin_phi * B + native_sqrt(1.0f - dir) * N + dir * cos_phi * T;
        
        ray_packet_t r;
        r.o = (float4)(P + HIT_BIAS * N, (float)px.x);
        r.d = (float4)(V, (float)px.y);
        r.c = orig_ray->c;
        r.c.xyz *= albedo.xyz;
        r.do_dx = do_dx;
        r.do_dy = do_dy;
        r.dd_dx = dd_dx - 2 * (dot(I, plane_N) * dndx + ddn_dx * plane_N);
        r.dd_dy = dd_dy - 2 * (dot(I, plane_N) * dndy + ddn_dy * plane_N);

        if (dot(r.c.xyz, r.c.xyz) > 0.005f) {
            const int index = atomic_inc(out_secondary_rays_count);
            out_secondary_rays[index] = r;
        }
    } else if (mat->type == GlossyMaterial) {
        col = (float3)(0, 0, 0);

        float3 V = reflect(I, dot(I, N) > 0 ? N : -N);

        float _unused;
        const float h = 1.0f - native_cos(0.5f * PI * mat->roughness * mat->roughness);
        const float z = h * fract(halton[hi * 2] + rand_offset, &_unused);

        const float dir = native_sqrt(z);
        const float phi = 2 * PI * fract(halton[hi * 2 + 1] + rand_offset2, &_unused);
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
        r.do_dx = do_dx;
        r.do_dy = do_dy;
        r.dd_dx = dd_dx - 2 * (dot(I, plane_N) * dndx + ddn_dx * plane_N);
        r.dd_dy = dd_dy - 2 * (dot(I, plane_N) * dndy + ddn_dy * plane_N);

        if (dot(r.c.xyz, r.c.xyz) > 0.005f) {
            const int index = atomic_inc(out_secondary_rays_count);
            out_secondary_rays[index] = r;
        }
    } else if (mat->type == RefractiveMaterial) {
        col = (float3)(0, 0, 0);

        const float3 _N = dot(I, N) > 0 ? -N : N;

        float eta = (dot(I, N) > 0) ? orig_ray->c.w : (orig_ray->c.w / mat->ior);
        float cosi = dot(-I, _N);
        float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
        float m = eta * cosi - sqrt(cost2);
        float3 V = eta * I + m * _N;

        const float z = 1.0f - halton[hi * 2] * mat->roughness;
        const float temp = native_sqrt(1.0f - z * z);

        const float phi = halton[((hash(hi) + iteration) & (HaltonSeqLen - 1)) * 2 + 0] * 2 * PI;
        float cos_phi;
        const float sin_phi = sincos(phi, &cos_phi);

        float3 TT = normalize(cross(V, B));
        float3 BB = normalize(cross(V, TT));
        V = temp * sin_phi * BB + z * V + temp * cos_phi * TT;

        float k = (eta - eta * eta * dot(I, plane_N) / dot(V, plane_N));
        float dmdx = k * ddn_dx;
        float dmdy = k * ddn_dy;

        ray_packet_t r;
        r.o = (float4)(P + HIT_BIAS * I, (float)px.x);
        r.d = (float4)(V, (float)px.y);
        r.c.xyz = orig_ray->c.xyz * z;
        r.c.w = mat->ior;
        r.do_dx = do_dx;
        r.do_dy = do_dy;
        r.dd_dx = eta * dd_dx - (m * dndx + dmdx * plane_N);
        r.dd_dy = eta * dd_dy - (m * dndy + dmdy * plane_N);

        if (cost2 >= 0 && dot(r.c.xyz, r.c.xyz) > 0.005f) {
            const int index = atomic_inc(out_secondary_rays_count);
            out_secondary_rays[index] = r;
        }
    } else if (mat->type == EmissiveMaterial) {
        col = mat->strength * albedo.xyz;
    } else if (mat->type == TransparentMaterial) {
        col = (float3)(0, 0, 0);

        ray_packet_t r;
        r.o = (float4)(P + HIT_BIAS * I, (float)px.x);
        r.d = orig_ray->d;
        r.c = orig_ray->c;
        r.do_dx = do_dx;
        r.do_dy = do_dy;
        r.dd_dx = dd_dx;
        r.dd_dy = dd_dy;

        if (dot(r.c.xyz, r.c.xyz) > 0.005f) {
            const int index = atomic_inc(out_secondary_rays_count);
            out_secondary_rays[index] = r;
        }
    }

    return (float4)(orig_ray->c.xyz * col, 1);
}

__kernel
void ShadePrimary(const int iteration, __global const float *halton, int w,
                  __global const hit_data_t *prim_inters, __global const ray_packet_t *prim_rays,
                  __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                  __global const mesh_t *meshes, __global const transform_t *transforms,
                  __global const uint *vtx_indices, __global const vertex_t *vertices,
                  __global const bvh_node_t *nodes, uint node_index, 
                  __global const tri_accel_t *tris, __global const uint *tri_indices, 
                  const environment_t env, __global const material_t *materials, __global const texture_t *textures, __read_only image2d_array_t texture_atlas, __write_only image2d_t frame_buf,
                  __global ray_packet_t *out_secondary_rays, __global int *out_secondary_rays_count) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float4 res = ShadeSurface((j * w + i), iteration, halton,
                  prim_inters, prim_rays,
                  mesh_instances, mi_indices,
                  meshes, transforms,
                  vtx_indices, vertices,
                  nodes, node_index, 
                  tris, tri_indices, 
                  env, materials, textures, texture_atlas,
                  out_secondary_rays, out_secondary_rays_count);

    write_imagef(frame_buf, (int2)(i, j), res);
}

__kernel
void ShadeSecondary(const int iteration, __global const float *halton,
                    __global const hit_data_t *prim_inters, __global const ray_packet_t *prim_rays,
                    __global const mesh_instance_t *mesh_instances, __global const uint *mi_indices,
                    __global const mesh_t *meshes, __global const transform_t *transforms,
                    __global const uint *vtx_indices, __global const vertex_t *vertices,
                    __global const bvh_node_t *nodes, uint node_index, 
                    __global const tri_accel_t *tris, __global const uint *tri_indices, 
                    const environment_t env, __global const material_t *materials, __global const texture_t *textures, __read_only image2d_array_t texture_atlas,
                    __write_only image2d_t frame_buf, __read_only image2d_t frame_buf2,
                    __global ray_packet_t *out_secondary_rays, __global int *out_secondary_rays_count) {
    const int index = get_global_id(0);

    __global const ray_packet_t *orig_ray = &prim_rays[index];

    const int2 px = (int2)(orig_ray->o.w, orig_ray->d.w);

    float4 col = read_imagef(frame_buf2, FBUF_SAMPLER, px);

    float4 res = ShadeSurface(index, iteration, halton,
                  prim_inters, prim_rays,
                  mesh_instances, mi_indices,
                  meshes, transforms,
                  vtx_indices, vertices,
                  nodes, node_index, 
                  tris, tri_indices, 
                  env, materials, textures, texture_atlas,
                  out_secondary_rays, out_secondary_rays_count);

    write_imagef(frame_buf, px, col + res);
}

)"