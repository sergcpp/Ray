#include "SceneBase.h"

#include <cassert>
#include <cstring>

#include "internal/Core.h"

uint32_t Ray::SceneBase::AddCamera(const camera_desc_t &c) {
    uint32_t i;
    if (cam_first_free_ == -1) {
        i = uint32_t(cams_.size());
        cams_.emplace_back();
    } else {
        i = cam_first_free_;
        cam_first_free_ = cams_[i].next_free;
    }
    SetCamera(i, c);
    if (current_cam_ == 0xffffffff) {
        current_cam_ = i;
    }
    return i;
}

void Ray::SceneBase::GetCamera(const uint32_t i, camera_desc_t &c) const {
    const camera_t &cam = cams_[i].cam;
    c.type = cam.type;
    c.dtype = cam.dtype;
    c.gamma = cam.gamma;
    if (c.type != Geo) {
        c.filter = cam.filter;
        memcpy(&c.origin[0], &cam.origin[0], 3 * sizeof(float));
        memcpy(&c.fwd[0], &cam.fwd[0], 3 * sizeof(float));
        memcpy(&c.up[0], &cam.up[0], 3 * sizeof(float));
        c.fov = cam.fov;
        c.focus_distance = cam.focus_distance;
        c.focal_length = cam.focal_length;
        c.fstop = cam.fstop;
        c.sensor_height = cam.sensor_height;
        c.lens_rotation = cam.lens_rotation;
        c.lens_ratio = cam.lens_ratio;
        c.lens_blades = cam.lens_blades;
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
    c.termination_start_depth = cam.pass_settings.termination_start_depth;
}

void Ray::SceneBase::SetCamera(const uint32_t i, const camera_desc_t &c) {
    assert(i < uint32_t(cams_.size()));
    camera_t &cam = cams_[i].cam;
    if (c.type != Geo) {
        if (c.ltype == eLensUnits::FOV) {
            ConstructCamera(c.type, c.filter, c.dtype, c.origin, c.fwd, c.up, c.fov, c.sensor_height, c.gamma,
                            c.focus_distance, c.fstop, c.lens_rotation, c.lens_ratio, c.lens_blades, &cam);
        } else if (c.ltype == eLensUnits::FLength) {
        }
    } else {
        cam.type = Geo;
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
    cam.pass_settings.termination_start_depth = c.termination_start_depth;
}

void Ray::SceneBase::RemoveCamera(const uint32_t i) {
    assert(i < uint32_t(cams_.size()));
    cams_[i].next_free = cam_first_free_;
    cam_first_free_ = i;
}