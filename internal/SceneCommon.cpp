#include "SceneCommon.h"

#include "Core.h"

void Ray::SceneCommon::GetEnvironment(environment_desc_t &env) {
    std::shared_lock<std::shared_timed_mutex> lock(mtx_);

    memcpy(env.env_col, env_.env_col, 3 * sizeof(float));
    env.env_map = TextureHandle{env_.env_map};
    memcpy(env.back_col, env_.back_col, 3 * sizeof(float));
    env.back_map = TextureHandle{env_.back_map};
    env.env_map_rotation = env_.env_map_rotation;
    env.back_map_rotation = env_.back_map_rotation;
    env.multiple_importance = env_.multiple_importance;
}

void Ray::SceneCommon::SetEnvironment(const environment_desc_t &env) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    memcpy(env_.env_col, env.env_col, 3 * sizeof(float));
    env_.env_map = env.env_map._index;
    memcpy(env_.back_col, env.back_col, 3 * sizeof(float));
    env_.back_map = env.back_map._index;
    env_.env_map_rotation = env.env_map_rotation;
    env_.back_map_rotation = env.back_map_rotation;
    env_.multiple_importance = env.multiple_importance;
}

Ray::CameraHandle Ray::SceneCommon::AddCamera(const camera_desc_t &c) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    const std::pair<uint32_t, uint32_t> al = cams_.Allocate(1);
    const Ray::CameraHandle i = {al.first, al.second};

    SetCamera_nolock(i, c);
    if (current_cam_ == InvalidCameraHandle) {
        current_cam_ = i;
    }
    return i;
}

void Ray::SceneCommon::GetCamera(const CameraHandle i, camera_desc_t &c) const {
    std::shared_lock<std::shared_timed_mutex> lock(mtx_);

    const camera_t &cam = cams_[i._index];
    c.type = cam.type;
    c.view_transform = cam.view_transform;
    c.exposure = cam.exposure;
    c.gamma = cam.gamma;
    if (c.type != eCamType::Geo) {
        c.filter = cam.filter;
        c.filter_width = cam.filter_width;
        memcpy(&c.origin[0], &cam.origin[0], 3 * sizeof(float));
        memcpy(&c.fwd[0], &cam.fwd[0], 3 * sizeof(float));
        memcpy(&c.up[0], &cam.up[0], 3 * sizeof(float));
        memcpy(&c.shift[0], &cam.shift[0], 2 * sizeof(float));
        c.fov = cam.fov;
        c.focus_distance = cam.focus_distance;
        c.focal_length = cam.focal_length;
        c.fstop = cam.fstop;
        c.sensor_height = cam.sensor_height;
        c.lens_rotation = cam.lens_rotation;
        c.lens_ratio = cam.lens_ratio;
        c.lens_blades = cam.lens_blades;
        c.clip_start = cam.clip_start;
        c.clip_end = cam.clip_end;
    } else {
        c.mi_index = cam.mi_index;
        c.uv_index = cam.uv_index;
    }

    c.lighting_only = (cam.pass_settings.flags & ePassFlags::LightingOnly);
    c.skip_direct_lighting = (cam.pass_settings.flags & ePassFlags::SkipDirectLight);
    c.skip_indirect_lighting = (cam.pass_settings.flags & ePassFlags::SkipIndirectLight);
    c.no_background = (cam.pass_settings.flags & ePassFlags::NoBackground);
    c.output_sh = (cam.pass_settings.flags & ePassFlags::OutputSH);
    c.output_base_color = (cam.pass_settings.flags & ePassFlags::OutputBaseColor);
    c.output_depth_normals = (cam.pass_settings.flags & ePassFlags::OutputDepthNormals);

    c.max_diff_depth = cam.pass_settings.max_diff_depth;
    c.max_spec_depth = cam.pass_settings.max_spec_depth;
    c.max_refr_depth = cam.pass_settings.max_refr_depth;
    c.max_transp_depth = cam.pass_settings.max_transp_depth;
    c.max_total_depth = cam.pass_settings.max_total_depth;
    c.min_total_depth = cam.pass_settings.min_total_depth;
    c.min_transp_depth = cam.pass_settings.min_transp_depth;

    c.clamp_direct = cam.pass_settings.clamp_direct;
    c.clamp_indirect = cam.pass_settings.clamp_indirect;

    c.min_samples = cam.pass_settings.min_samples;
    c.variance_threshold = cam.pass_settings.variance_threshold;
    c.regularize_alpha = cam.pass_settings.regularize_alpha;
}

void Ray::SceneCommon::RemoveCamera(const CameraHandle i) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);
    cams_.Erase(i._block);
}

void Ray::SceneCommon::SetCamera_nolock(const CameraHandle i, const camera_desc_t &c) {
    assert(i._index < uint32_t(cams_.size()));
    camera_t &cam = cams_[i._index];
    if (c.type != eCamType::Geo) {
        if (c.ltype == eLensUnits::FOV) {
            ConstructCamera(c.type, c.filter, c.filter_width, c.view_transform, c.origin, c.fwd, c.up, c.shift, c.fov,
                            c.sensor_height, c.exposure, c.gamma, c.focus_distance, c.fstop, c.lens_rotation,
                            c.lens_ratio, c.lens_blades, c.clip_start, c.clip_end, &cam);
        } else if (c.ltype == eLensUnits::FLength) {
        }
    } else {
        cam.type = eCamType::Geo;
        cam.exposure = c.exposure;
        cam.gamma = c.gamma;
        cam.mi_index = c.mi_index;
        cam.uv_index = c.uv_index;
    }

    cam.pass_settings.flags = {};
    if (c.lighting_only) {
        cam.pass_settings.flags |= ePassFlags::LightingOnly;
    }
    if (c.skip_direct_lighting) {
        cam.pass_settings.flags |= ePassFlags::SkipDirectLight;
    }
    if (c.skip_indirect_lighting) {
        cam.pass_settings.flags |= ePassFlags::SkipIndirectLight;
    }
    if (c.no_background) {
        cam.pass_settings.flags |= ePassFlags::NoBackground;
    }
    if (c.output_sh) {
        cam.pass_settings.flags |= ePassFlags::OutputSH;
    }
    if (c.output_base_color) {
        cam.pass_settings.flags |= ePassFlags::OutputBaseColor;
    }
    if (c.output_depth_normals) {
        cam.pass_settings.flags |= ePassFlags::OutputDepthNormals;
    }

    cam.pass_settings.max_diff_depth = c.max_diff_depth;
    cam.pass_settings.max_spec_depth = c.max_spec_depth;
    cam.pass_settings.max_refr_depth = c.max_refr_depth;
    cam.pass_settings.max_transp_depth = c.max_transp_depth;
    cam.pass_settings.max_total_depth = c.max_total_depth;
    cam.pass_settings.min_total_depth = c.min_total_depth;
    cam.pass_settings.min_transp_depth = c.min_transp_depth;

    // make sure to not exceed allowed bounces
    while (cam.pass_settings.max_transp_depth + cam.pass_settings.max_total_depth > MAX_BOUNCES) {
        cam.pass_settings.max_transp_depth = std::max(cam.pass_settings.max_transp_depth - 1, 0);
        cam.pass_settings.max_total_depth = std::max(cam.pass_settings.max_total_depth - 1, 0);
    }

    cam.pass_settings.clamp_direct = c.clamp_direct;
    cam.pass_settings.clamp_indirect = c.clamp_indirect;

    cam.pass_settings.min_samples = c.min_samples;
    cam.pass_settings.variance_threshold = c.variance_threshold;
    cam.pass_settings.regularize_alpha = c.regularize_alpha;
}
