#pragma once

#include <memory>

#include "Config.h"
#include "SceneBase.h"
#include "Types.h"

/**
  @file RendererBase.h
*/

namespace Ray {
/// Renderer flags used to choose backend
enum eRendererType : uint32_t {
    // Reference renderer, slightly vectorized, the easiest to modify and debug
    RendererRef = (1 << 0),
    // SIMD renderers, heavily vectorized
    RendererSSE2 = (1 << 1),
    RendererSSE41 = (1 << 2),
    RendererAVX = (1 << 3),
    RendererAVX2 = (1 << 4),
    RendererAVX512 = (1 << 5),
    RendererNEON = (1 << 6),
    // GPU renderer
    RendererVK = (1 << 7),
    // All CPU renderers
    RendererCPU =
        RendererRef | RendererSSE2 | RendererSSE41 | RendererNEON | RendererAVX | RendererAVX2 /*| RendererAVX512 */,
    // All GPU renderers
    RendererGPU = RendererVK
};

const char *RendererTypeName(eRendererType rt);
eRendererType RendererTypeFromName(const char *name);

/// Returns whether it is safe to call Render function for non-overlaping regions from different threads
bool RendererSupportsMultithreading(eRendererType rt);

/// Renderer settings
struct settings_t {
    int w = 0, h = 0;
#ifdef ENABLE_GPU_IMPL
    const char *preferred_device = nullptr;
    bool use_tex_compression = true; // temporarily GPU only
#endif // ENABLE_GPU_IMPL
    bool use_hwrt = true;
    bool use_bindless = true;
    bool use_wide_bvh = true;
};

/** Render region context,
    holds information for specific rectangle on image
*/
class RegionContext {
    /// Rectangle on image
    rect_t rect_;

  public:
    int iteration = 0;                   ///< Number of rendered samples per pixel
    std::unique_ptr<float[]> halton_seq; ///< Sequence of random 2D points

    explicit RegionContext(const rect_t &rect) : rect_(rect) {}

    rect_t rect() const { return rect_; }

    /// Clear region context (used to start again)
    void Clear() {
        iteration = 0;
        halton_seq = nullptr;
    }
};

class ILog;

/** Base class for all renderer backends
 */
class RendererBase {
  public:
    virtual ~RendererBase() = default;

    /// Type of renderer
    virtual eRendererType type() const = 0;

    /// Log
    virtual ILog *log() const { return nullptr; }

    /// Name of the device
    virtual const char *device_name() const = 0;

    virtual bool is_hwrt() const { return false; }

    /// Returns size of rendered image
    virtual std::pair<int, int> size() const = 0;

    /// Returns pointer to rendered image
    virtual const pixel_color_t *get_pixels_ref() const = 0;

    /// Returns pointer to 'raw' untonemapped image
    virtual const pixel_color_t *get_raw_pixels_ref() const = 0;

    /// Returns pointer to SH data
    virtual const shl1_data_t *get_sh_data_ref() const = 0;

    /** @brief Resize framebuffer
        @param w new image width
        @param h new image height
    */
    virtual void Resize(int w, int h) = 0;

    /** @brief Clear framebuffer
        @param c color used to fill image
    */
    virtual void Clear(const pixel_color_t &c) = 0;

    /** @brief Create new scene
        @return pointer to new scene for specific backend
    */
    virtual SceneBase *CreateScene() = 0;

    /** @brief Render image region
        @param scene reference to a scene
        @param region image region to render
    */
    virtual void RenderScene(const SceneBase *scene, RegionContext &region) = 0;

    struct stats_t {
        unsigned long long time_primary_ray_gen_us;
        unsigned long long time_primary_trace_us;
        unsigned long long time_primary_shade_us;
        unsigned long long time_primary_shadow_us;
        unsigned long long time_secondary_sort_us;
        unsigned long long time_secondary_trace_us;
        unsigned long long time_secondary_shade_us;
        unsigned long long time_secondary_shadow_us;
    };
    virtual void GetStats(stats_t &st) = 0;
    virtual void ResetStats() = 0;
};
} // namespace Ray