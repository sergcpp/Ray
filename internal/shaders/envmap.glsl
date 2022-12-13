#ifndef ENVMAP_GLSL
#define ENVMAP_GLSL

vec3 CanonicalToDir(const vec2 p, const float y_rotation) {
    const float cos_theta = 2 * p[0] - 1;
    float phi = 2 * PI * p[1] + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    const float sin_theta = sqrt(1 - cos_theta * cos_theta);

    const float sin_phi = sin(phi);
    const float cos_phi = cos(phi);

    return vec3(sin_theta * cos_phi, cos_theta, -sin_theta * sin_phi);
}

vec2 DirToCanonical(const vec3 d, const float y_rotation) {
    const float cos_theta = clamp(d[1], -1.0, 1.0);

    float phi = -atan(d[2], d[0]) + y_rotation;
    if (phi < 0) {
        phi += 2 * PI;
    }
    if (phi > 2 * PI) {
        phi -= 2 * PI;
    }

    return vec2((cos_theta + 1.0) / 2.0, phi / (2.0 * PI));
}

float Evaluate_EnvQTree(const float y_rotation, sampler2D qtree_tex, const int qtree_levels, const vec3 L) {
    int res = 2;
    int lod = qtree_levels - 1;

    vec2 p = DirToCanonical(L, y_rotation);

    float factor = 1.0;

    while (lod >= 0) {
        const ivec2 coords = clamp(ivec2(p * res), ivec2(0), ivec2(res - 1));

        int index = 0;
        index |= (coords.x & 1) << 0;
        index |= (coords.y & 1) << 1;

        const ivec2 q = coords / 2;
        const vec4 quad = texelFetch(qtree_tex, q, lod);
        const float total = quad[0] + quad[1] + quad[2] + quad[3];
        if (total <= 0.0) {
            break;
        }

        factor *= 4.0 * quad[index] / total;

        --lod;
        res *= 2;
    }

    return factor / (4.0 * PI);
}

vec4 Sample_EnvQTree(const float y_rotation, sampler2D qtree_tex, const int qtree_levels,
                     const float rand, const float rx, const float ry) {
    int res = 2;
    float _step = 1.0 / float(res);

    float _sample = rand;
    int lod = qtree_levels - 1;

    vec2 origin = vec2(0.0);
    float factor = 1.0;

    while (lod >= 0) {
        const ivec2 q = ivec2(origin * res) / 2;
        const vec4 quad = texelFetch(qtree_tex, q, lod);

        const float top_left = quad[0];
        const float top_right = quad[1];
        float partial = top_left + quad[2];
        const float total = partial + top_right + quad[3];
        if (total <= 0.0) {
            break;
        }

        float boundary = partial / total;

        int index = 0;
        if (_sample < boundary) {
            _sample /= boundary;
            boundary = top_left / partial;
        } else {
            partial = total - partial;
            origin[0] += _step;
            _sample = (_sample - boundary) / (1.0 - boundary);
            boundary = top_right / partial;
            index |= (1 << 0);
        }

        if (_sample < boundary) {
            _sample /= boundary;
        } else {
            origin[1] += _step;
            _sample = (_sample - boundary) / (1.0 - boundary);
            index |= (1 << 1);
        }

        factor *= 4.0 * quad[index] / total;

        --lod;
        res *= 2;
        _step *= 0.5;
    }

    origin += 2 * _step * vec2(rx, ry);

    return vec4(CanonicalToDir(origin, y_rotation), factor / (4.0 * PI));
}

#endif // ENVMAP_GLSL