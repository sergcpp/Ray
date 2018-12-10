R"(

int hash(int x) {
    uint ret = as_uint(x);
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = ((ret >> 16) ^ ret) * 0x45d9f3b;
    ret = (ret >> 16) ^ ret;
    return as_int(ret);
}

float construct_float(uint m) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = as_float(m);                // Range [1:2]
    return f - 1.0f;                       // Range [0:1]
}

float3 get_cam_dir(const float x, const float y, float3 origin, const camera_t *cam, int w, int h, float prop) {
    float k = native_tan(0.5f * cam->origin.w * PI / 180.0f) * cam->side.w;
    float3 p = (float3)(2 * k * x / w - k, 2 * k * -y / h + k, cam->side.w);
    p = cam->origin.xyz + prop * p.x * cam->side.xyz + p.y * cam->up.xyz + p.z * cam->fwd.xyz;
    return fast_normalize(p - origin);
}

__kernel
void GeneratePrimaryRays(const int iteration, camera_t cam, int w, int h, __global const float *halton, __global ray_packet_t *out_rays) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int index = j * w + i;
    const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;

    __global ray_packet_t *r = &out_rays[index];
    r->do_dx = r->do_dy = (float4)((float3)(0.0f), as_float(0));
    r->c = (float4)(1.0f);

    float k = ((float)w) / h;

    float x = (float)i;
    float y = (float)j;

    float _unused;
    int hash_val = hash(index);
    if (cam.flags & CAM_USE_TENT_FILTER) {
        float rx = fract(halton[hi + 0] + construct_float(hash_val), &_unused);
        if (rx < 0.5f) {
            rx = native_sqrt(2 * rx) - 1.0f;
        } else {
            rx = 1.0f - native_sqrt(2.0f - 2 * rx);
        }

        float ry = fract(halton[hi + 1] + construct_float(hash(hash_val)), &_unused);
        if (ry < 0.5f) {
            ry = native_sqrt(2 * ry) - 1.0f;
        } else {
            ry = 1.0f - native_sqrt(2.0f - 2 * ry);
        }

        x += 0.5f + rx;
        y += 0.5f + ry;
    } else {
        x += fract(halton[hi + 0] + construct_float(hash_val), &_unused);
        y += fract(halton[hi + 1] + construct_float(hash(hash_val)), &_unused);
    }

    float ff1 = cam.up.w * (-0.5f + fract(halton[hi + 2 + 0] + construct_float(hash_val), &_unused));
    float ff2 = cam.up.w * (-0.5f + fract(halton[hi + 2 + 1] + construct_float(hash(hash_val)), &_unused));

    float3 origin = cam.origin.xyz + cam.side.xyz * ff1 + cam.up.xyz * ff2;
    r->o = (float4)(origin, (float)i);

    float3 d = get_cam_dir(x, y, origin, &cam, w, h, k);
    r->d = (float4)(d, (float)j);
    
    float3 _dx = get_cam_dir(x + 1, y, origin, &cam, w, h, k);
    r->dd_dx.xyz = _dx - d;
    r->dd_dx.w = as_float(0);

    float3 _dy = get_cam_dir(x, y + 1, origin, &cam, w, h, k);
    r->dd_dy.xyz = _dy - d;
    r->dd_dy.w = as_float(0);
}

__kernel
void SampleMeshInTextureSpace_ResetBins(__global uint *out_tri_bins) {
    const int i = get_global_id(0);

    __global uint *counter = &out_tri_bins[i * 1024];
    *counter = 0;
}

__kernel
void SampleMeshInTextureSpace_BinStage(int uv_layer, uint tri_offset, __global const uint *vtx_indices, __global const vertex_t *vertices,
                                       int w, int h, __global uint *out_tri_bins) {
    const int i = get_global_id(0);
    
    const float2 size = (float2)(w, h);

    const float2 rect_min = (float2)(0.0f, 0.0f),
                 rect_max = (float2)(w - 1, h - 1);

    int tw = w / TRI_RAST_X + ((w % TRI_RAST_X) ? 1 : 0);

    uint tri = tri_offset + i;

    __global const vertex_t *v0 = &vertices[vtx_indices[tri * 3 + 0]];
    __global const vertex_t *v1 = &vertices[vtx_indices[tri * 3 + 1]];
    __global const vertex_t *v2 = &vertices[vtx_indices[tri * 3 + 2]];

    const float2 t0 = (float2)(v0->t[uv_layer][0], 1.0f - v0->t[uv_layer][1]) * size;
    const float2 t1 = (float2)(v1->t[uv_layer][0], 1.0f - v1->t[uv_layer][1]) * size;
    const float2 t2 = (float2)(v2->t[uv_layer][0], 1.0f - v2->t[uv_layer][1]) * size;

    float2 bbox_min = t0, bbox_max = t0;

    bbox_min = fmin(bbox_min, t1);
    bbox_min = fmin(bbox_min, t2);

    bbox_max = fmax(bbox_max, t1);
    bbox_max = fmax(bbox_max, t2);

    bbox_min = max(bbox_min, rect_min);
    bbox_max = min(bbox_max, rect_max);

    int2 ibbox_min = convert_int2(bbox_min),
         ibbox_max = convert_int2(round(bbox_max));

    for (int y = ibbox_min.y / TRI_RAST_Y; y <= ibbox_max.y / TRI_RAST_Y; y++) {
        for (int x = ibbox_min.x / TRI_RAST_X; x <= ibbox_max.x / TRI_RAST_X; x++) {
            __global uint *bins = &out_tri_bins[(y * tw + x) * 1024];

            int index = atomic_inc(bins);
            if (index < 1023) {
                bins[index + 1] = tri;
            }
        }
    }
}

__kernel
void SampleMeshInTextureSpace_RasterStage(int uv_layer, int iteration, uint tr_index, uint obj_index, __global const transform_t *transforms,
                                          __global const uint *vtx_indices, __global const vertex_t *vertices,
                                          int w, int h, __global const float *halton, __global const uint *tri_bins,
                                          __global ray_packet_t *out_rays, __global hit_data_t *out_inters) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int index = j * w + i;
    const int hi = (iteration & (HALTON_SEQ_LEN - 1)) * HALTON_COUNT;
    const int hash_val = hash(index);

    const int bx = i / TRI_RAST_X,
              by = j / TRI_RAST_Y;

    int tw = w / TRI_RAST_X + ((w % TRI_RAST_X) ? 1 : 0);

    float2 size = (float2)(w, h);

    __global const uint *bins = &tri_bins[(by * tw + bx) * 1024];
    __global const transform_t *tr = &transforms[tr_index];
    
    __global ray_packet_t *ray = &out_rays[index];
    __global hit_data_t *inter = &out_inters[index];

    ray->d = 0.0f;

    inter->mask = 0;
    inter->ray_id = (float2)(i, j);

    uint tri_count = *bins;

    for (uint p = 0; p < tri_count; p++) {
        uint tri = bins[p + 1];

        __global const vertex_t *v0 = &vertices[vtx_indices[tri * 3 + 0]];
        __global const vertex_t *v1 = &vertices[vtx_indices[tri * 3 + 1]];
        __global const vertex_t *v2 = &vertices[vtx_indices[tri * 3 + 2]];

        const float2 t0 = (float2)(v0->t[uv_layer][0], 1.0f - v0->t[uv_layer][1]) * size;
        const float2 t1 = (float2)(v1->t[uv_layer][0], 1.0f - v1->t[uv_layer][1]) * size;
        const float2 t2 = (float2)(v2->t[uv_layer][0], 1.0f - v2->t[uv_layer][1]) * size;

        float2 bbox_min = t0, bbox_max = t0;

        bbox_min = fmin(bbox_min, t1);
        bbox_min = fmin(bbox_min, t2);

        bbox_max = fmax(bbox_max, t1);
        bbox_max = fmax(bbox_max, t2);

        int2 ibbox_min = convert_int2(bbox_min),
             ibbox_max = convert_int2(round(bbox_max));

        if (i < ibbox_min.x || i > ibbox_max.x || j < ibbox_min.y || j > ibbox_max.y) continue;

        const float2 d01 = t0 - t1, d12 = t1 - t2, d20 = t2 - t0;

        float area = d01.x * d20.y - d20.x * d01.y;
        if (area < FLT_EPS) continue;

        float inv_area = 1.0f / area;

        float _unused;
        float _x = (float)(i) + fract(halton[hi + 0] + construct_float(hash_val), &_unused);
        float _y = (float)(j) + fract(halton[hi + 1] + construct_float(hash(hash_val)), &_unused);

        float u = d01.x * (_y - t0.y) - d01.y * (_x - t0.x),
              v = d12.x * (_y - t1.y) - d12.y * (_x - t1.x),
              w = d20.x * (_y - t2.y) - d20.y * (_x - t2.x);

        if (u >= -FLT_EPS && v >= -FLT_EPS && w >= -FLT_EPS) {
            const float3 p0 = (float3)(v0->p[0], v0->p[1], v0->p[2]),
                         p1 = (float3)(v1->p[0], v1->p[1], v1->p[2]),
                         p2 = (float3)(v2->p[0], v2->p[1], v2->p[2]);

            const float3 n0 = (float3)(v0->n[0], v0->n[1], v0->n[2]),
                         n1 = (float3)(v1->n[0], v1->n[1], v1->n[2]),
                         n2 = (float3)(v2->n[0], v2->n[1], v2->n[2]);

            u *= inv_area; v *= inv_area; w *= inv_area;

            const float3 p = TransformPoint(p0 * v + p1 * w + p2 * u, &tr->xform),
                         n = TransformNormal(n0 * v + n1 * w + n2 * u, &tr->inv_xform);

            const float3 o = p + n, d = -n;

            ray->o = (float4)(o, (float)i);
            ray->d = (float4)(d, (float)j);
   
            ray->c = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
            ray->do_dx = ray->do_dy = (float4)(0.0f, 0.0f, 0.0f, as_float(0));
            ray->dd_dx = ray->dd_dy = (float4)(0.0f, 0.0f, 0.0f, as_float(0));

            inter->mask = 0xffffffff;
            inter->obj_index = obj_index;
            inter->prim_index = tri;
            inter->t = 1.0f;
            inter->u = w;
            inter->v = u;
        }
    }
}

)"