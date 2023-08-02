#pragma once

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "../SceneBase.h"

namespace Ray {
class SceneCommon : public SceneBase {
  protected:
    struct cam_storage_t {
        camera_t cam;
        CameraHandle next_free;
    };

    mutable std::shared_timed_mutex mtx_;

    std::vector<cam_storage_t> cams_;                   ///< scene cameras
    CameraHandle cam_first_free_ = InvalidCameraHandle; ///< index to first free cam in cams_ array

    CameraHandle current_cam_ = InvalidCameraHandle; ///< index of current camera

    void SetCamera_nolock(CameraHandle i, const camera_desc_t &c);

  public:
    CameraHandle current_cam() const override {
        std::shared_lock<std::shared_timed_mutex> lock(mtx_);
        return current_cam_;
    }

    void set_current_cam(CameraHandle i) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        current_cam_ = i;
    }

    CameraHandle AddCamera(const camera_desc_t &c) override;
    void RemoveCamera(CameraHandle i) override;

    void GetCamera(CameraHandle i, camera_desc_t &c) const override;

    void SetCamera(CameraHandle i, const camera_desc_t &c) override {
        std::unique_lock<std::shared_timed_mutex> lock(mtx_);
        SetCamera_nolock(i, c);
    }
};
} // namespace Ray