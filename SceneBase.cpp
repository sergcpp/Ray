#include "SceneBase.h"

#include <cassert>
#include <cstring>

#include "internal/Core.h"

uint32_t Ray::SceneBase::AddCamera(eCamType type, eFilterType filter, const float origin[3], const float fwd[3], float fov, float gamma, float focus_distance, float focus_factor) {
    uint32_t i;
    if (cam_first_free_ == -1) {
        i = (uint32_t)cams_.size();
        cams_.emplace_back();
    } else {
        i = cam_first_free_;
        cam_first_free_ = cams_[i].next_free;
    }
    ConstructCamera(type, filter, origin, fwd, fov, gamma, focus_distance, focus_factor, &cams_[i].cam);
    if (current_cam_ == 0xffffffff) current_cam_ = i;
    return i;
}

void Ray::SceneBase::SetCamera(uint32_t i, eCamType type, eFilterType filter, const float origin[3], const float fwd[3], float fov, float gamma, float focus_distance, float focus_factor) {
    assert(i < (uint32_t)cams_.size());
    ConstructCamera(type, filter, origin, fwd, fov, gamma, focus_distance, focus_factor, &cams_[i].cam);
}

void Ray::SceneBase::RemoveCamera(uint32_t i) {
    assert(i < (uint32_t)cams_.size());
    cams_[i].next_free = cam_first_free_;
    cam_first_free_ = i;
}