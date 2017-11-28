#include "SceneBase.h"

#include <cassert>
#include <cstring>

#include "internal/Core.h"

uint32_t ray::SceneBase::AddCamera(eCamType type, const float origin[3], const float fwd[3], float fov) {
    uint32_t i;
    if (cam_first_free_ == -1) {
        i = (uint32_t)cams_.size();
        cams_.emplace_back();
    } else {
        i = cam_first_free_;
        cam_first_free_ = cams_[i].next_free;
    }
    ConstructCamera(type, origin, fwd, fov, &cams_[i].cam);
    if (current_cam_ == 0xffffffff) current_cam_ = i;
    return i;
}

void ray::SceneBase::SetCamera(uint32_t i, eCamType type, const float origin[3], const float fwd[3], float fov) {
    assert(i < (uint32_t)cams_.size());
    ConstructCamera(type, origin, fwd, fov, &cams_[i].cam);
}

void ray::SceneBase::RemoveCamera(uint32_t i) {
    assert(i < (uint32_t)cams_.size());
    memset(&cams_[i], 0, sizeof(cam_storage_t));
    cams_[i].next_free = cam_first_free_;
    cam_first_free_ = i;
}