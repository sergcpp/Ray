#include "SceneBase.h"

#include <cassert>
#include <cstring>

#include "internal/Core.h"

uint32_t Ray::SceneBase::AddCamera(const camera_desc_t &c) {
    uint32_t i;
    if (cam_first_free_ == -1) {
        i = (uint32_t)cams_.size();
        cams_.emplace_back();
    } else {
        i = cam_first_free_;
        cam_first_free_ = cams_[i].next_free;
    }
    SetCamera(i, c);
    if (current_cam_ == 0xffffffff) current_cam_ = i;
    return i;
}

void Ray::SceneBase::GetCamera(uint32_t i, camera_desc_t &c) const {
    const auto &cam = cams_[i].cam;
    c.type = cam.type;
    c.gamma = cam.gamma;
    if (c.type != Geo) {
        c.filter = cam.filter;
        memcpy(&c.origin[0], &cam.origin[0], 3 * sizeof(float));
        memcpy(&c.fwd[0], &cam.fwd[0], 3 * sizeof(float));
        c.fov = cam.fov;
        c.focus_distance = cam.focus_distance;
        c.focus_factor = cam.focus_factor;
    } else {
        c.mi_index = cam.mi_index;
        c.uv_index = cam.uv_index;
    }

    c.lighting_only = (cam.pass_flags & LightingOnly) != 0;
    c.skip_direct_lighting = (cam.pass_flags & SkipDirectLight) != 0;
    c.skip_indirect_lighting = (cam.pass_flags & SkipIndirectLight) != 0;
    c.no_background = (cam.pass_flags & NoBackground) != 0;
    c.clamp = (cam.pass_flags & Clamp) != 0;
}

void Ray::SceneBase::SetCamera(uint32_t i, const camera_desc_t &c) {
    assert(i < (uint32_t)cams_.size());
    auto &cam = cams_[i].cam;
    if (c.type != Geo) {
        ConstructCamera(c.type, c.filter, c.origin, c.fwd, c.fov, c.gamma, c.focus_distance, c.focus_factor, &cam);
    } else {
        cam.type = Geo;
        cam.gamma = c.gamma;
        cam.mi_index = c.mi_index;
        cam.uv_index = c.uv_index;
    }

    cam.pass_flags = 0;
    if (c.lighting_only) cam.pass_flags |= LightingOnly;
    if (c.skip_direct_lighting) cam.pass_flags |= SkipDirectLight;
    if (c.skip_indirect_lighting) cam.pass_flags |= SkipIndirectLight;
    if (c.no_background) cam.pass_flags |= NoBackground;
    if (c.clamp) cam.pass_flags |= Clamp;
}

void Ray::SceneBase::RemoveCamera(uint32_t i) {
    assert(i < (uint32_t)cams_.size());
    cams_[i].next_free = cam_first_free_;
    cam_first_free_ = i;
}