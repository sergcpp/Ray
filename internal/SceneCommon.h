#pragma once

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "../SceneBase.h"
#include "SparseStorageCPU.h"

namespace Ray {
struct atmosphere_params_t;
class SceneCommon : public SceneBase {
  protected:
    mutable std::shared_timed_mutex mtx_;

    Cpu::SparseStorage<camera_t> cams_;

    CameraHandle current_cam_ = InvalidCameraHandle;

    environment_t env_;
    aligned_vector<float, 16> sky_transmittance_lut_, sky_multiscatter_lut_;

    void UpdateSkyTransmittanceLUT(const atmosphere_params_t &params);
    void UpdateMultiscatterLUT(const atmosphere_params_t &params);
    std::vector<color_rgba8_t>
    CalcSkyEnvTexture(const atmosphere_params_t &params, const int res[2], const light_t lights[],
                      Span<const uint32_t> dir_lights,
                      const std::function<void(int, int, ParallelForFunction &&)> &parallel_for);
    void SetCamera_nolock(CameraHandle i, const camera_desc_t &c);

  public:
    void GetEnvironment(environment_desc_t &env) override;
    void SetEnvironment(const environment_desc_t &env) override;

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