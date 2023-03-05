#include "SceneBase.h"

#include <cassert>
#include <cstring>

#include "internal/Core.h"

Ray::CameraHandle Ray::SceneBase::AddCamera(const camera_desc_t &c) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    CameraHandle i;
    if (cam_first_free_ == InvalidCameraHandle) {
        i = CameraHandle{uint32_t(cams_.size())};
        cams_.emplace_back();
    } else {
        i = cam_first_free_;
        cam_first_free_ = cams_[i._index].next_free;
    }
    SetCamera_nolock(i, c);
    if (current_cam_ == InvalidCameraHandle) {
        current_cam_ = i;
    }
    return i;
}

void Ray::SceneBase::GetCamera(const CameraHandle i, camera_desc_t &c) const {
    std::shared_lock<std::shared_timed_mutex> lock(mtx_);

    const camera_t &cam = cams_[i._index].cam;
    c.type = cam.type;
    c.dtype = cam.dtype;
    c.exposure = cam.exposure;
    c.gamma = cam.gamma;
    if (c.type != Geo) {
        c.filter = cam.filter;
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

    c.lighting_only = (cam.pass_settings.flags & LightingOnly) != 0;
    c.skip_direct_lighting = (cam.pass_settings.flags & SkipDirectLight) != 0;
    c.skip_indirect_lighting = (cam.pass_settings.flags & SkipIndirectLight) != 0;
    c.no_background = (cam.pass_settings.flags & NoBackground) != 0;
    c.clamp = (cam.pass_settings.flags & Clamp) != 0;
    c.output_sh = (cam.pass_settings.flags & OutputSH) != 0;

    c.max_diff_depth = cam.pass_settings.max_diff_depth;
    c.max_spec_depth = cam.pass_settings.max_spec_depth;
    c.max_refr_depth = cam.pass_settings.max_refr_depth;
    c.max_transp_depth = cam.pass_settings.max_transp_depth;
    c.max_total_depth = cam.pass_settings.max_total_depth;
    c.min_total_depth = cam.pass_settings.min_total_depth;
    c.min_transp_depth = cam.pass_settings.min_transp_depth;
}

void Ray::SceneBase::SetCamera_nolock(const CameraHandle i, const camera_desc_t &c) {
    assert(i._index < uint32_t(cams_.size()));
    camera_t &cam = cams_[i._index].cam;
    if (c.type != Geo) {
        if (c.ltype == eLensUnits::FOV) {
            ConstructCamera(c.type, c.filter, c.dtype, c.origin, c.fwd, c.up, c.shift, c.fov, c.sensor_height,
                            c.exposure, c.gamma, c.focus_distance, c.fstop, c.lens_rotation, c.lens_ratio,
                            c.lens_blades, c.clip_start, c.clip_end, &cam);
        } else if (c.ltype == eLensUnits::FLength) {
        }
    } else {
        cam.type = Geo;
        cam.exposure = c.exposure;
        cam.gamma = c.gamma;
        cam.mi_index = c.mi_index;
        cam.uv_index = c.uv_index;
    }

    cam.pass_settings.flags = 0;
    if (c.lighting_only) {
        cam.pass_settings.flags |= LightingOnly;
    }
    if (c.skip_direct_lighting) {
        cam.pass_settings.flags |= SkipDirectLight;
    }
    if (c.skip_indirect_lighting) {
        cam.pass_settings.flags |= SkipIndirectLight;
    }
    if (c.no_background) {
        cam.pass_settings.flags |= NoBackground;
    }
    if (c.clamp) {
        cam.pass_settings.flags |= Clamp;
    }
    if (c.output_sh) {
        cam.pass_settings.flags |= OutputSH;
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
}

void Ray::SceneBase::RemoveCamera(const CameraHandle i) {
    std::unique_lock<std::shared_timed_mutex> lock(mtx_);

    assert(i._index < uint32_t(cams_.size()));
    cams_[i._index].next_free = cam_first_free_;
    cam_first_free_ = i;
}