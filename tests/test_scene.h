#pragma once

#include <cstdarg>
#include <cstdio>

#include <atomic>
#include <mutex>

#include "../Log.h"

enum class eTestScene {
    Standard,
    Standard_SphereLight,
    Standard_InsideLight,
    Standard_SpotLight,
    Standard_MeshLights,
    Standard_DirLight,
    Standard_SunLight,
    Standard_MoonLight,
    Standard_HDRLight,
    Standard_NoLight,
    Standard_DOF0,
    Standard_DOF1,
    Standard_GlassBall0,
    Standard_GlassBall1,
    Standard_Clipped,
    Refraction_Plane,
    Ray_Flags,
    Two_Sided
};

enum class eDenoiseMethod { None, NLM, UNet };

class ThreadPool;

namespace Ray {
class RendererBase;
class SceneBase;

struct settings_t;
} // namespace Ray

extern std::atomic_bool g_log_contains_errors;
extern bool g_catch_flt_exceptions;

class LogErr final : public Ray::ILog {
    FILE *err_out_ = nullptr;
    std::mutex mtx_;

  public:
    LogErr() {
#pragma warning(suppress : 4996)
        err_out_ = fopen("test_data/errors.txt", "w");
    }
    ~LogErr() override { fclose(err_out_); }

    void Info(const char *fmt, ...) override {
        // ignored
    }
    void Warning(const char *fmt, ...) override {
        // ignored
    }
    void Error(const char *fmt, ...) override {
        std::lock_guard<std::mutex> _(mtx_);

        va_list vl;
        va_start(vl, fmt);
        vfprintf(err_out_, fmt, vl);
        va_end(vl);
        putc('\n', err_out_);
        fflush(err_out_);
        g_log_contains_errors = true;
    }
};

extern LogErr g_log_err;

template <typename MatDesc>
void setup_test_scene(ThreadPool &threads, Ray::SceneBase &scene, int min_samples, float variance_threshold,
                      const MatDesc &main_mat_desc, const char *textures[], eTestScene test_scene);

void schedule_render_jobs(ThreadPool &threads, Ray::RendererBase &renderer, const Ray::SceneBase *scene,
                          const Ray::settings_t &settings, int max_samples, eDenoiseMethod denoise, bool partial,
                          const char *log_str);