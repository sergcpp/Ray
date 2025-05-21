#pragma once

#include <cstring>
#include <memory>
#include <string_view>

#include "Config.h"
#include "SceneBase.h"
#include "Types.h"
#include "VulkanFunctions.h"

struct ID3D12Resource;
struct ID3D12DescriptorHeap;
struct ID3D12GraphicsCommandList;

/**
  @file RendererBase.h
*/

namespace Ray {
/// Renderer flags used to choose backend
enum class eRendererType : uint32_t {
    // Reference CPU renderer, slightly vectorized, the easiest to modify and debug
    Reference,
    // SIMD CPU renderers, heavily vectorized in SPMD fashion
    SIMD_SSE2,
    SIMD_SSE41,
    SIMD_AVX,
    SIMD_AVX2,
    SIMD_AVX512,
    SIMD_NEON,
    // GPU renderers
    Vulkan,
    DirectX12
};

// All CPU renderers
const Bitmask<eRendererType> RendererCPU = Bitmask<eRendererType>{eRendererType::Reference} | eRendererType::SIMD_SSE2 |
                                           eRendererType::SIMD_SSE41 | eRendererType::SIMD_NEON |
                                           eRendererType::SIMD_AVX | eRendererType::SIMD_AVX2;
// All GPU renderers
const Bitmask<eRendererType> RendererGPU = Bitmask<eRendererType>{eRendererType::Vulkan} | eRendererType::DirectX12;

std::string_view RendererTypeName(eRendererType rt);
eRendererType RendererTypeFromName(std::string_view name);

/// Returns whether it is safe to call Render function for non-overlapping regions from different threads
bool RendererSupportsMultithreading(eRendererType rt);
/// Returns whether this type of renderer supports hardware raytracing
bool RendererSupportsHWRT(eRendererType rt);

/// Renderer settings
struct settings_t {
    int w = 0, h = 0;
    std::string_view preferred_device;
    bool use_tex_compression = true;
    bool use_hwrt = true;
    bool use_bindless = true;
    bool use_spatial_cache = false;
    int validation_level = 0;

    VulkanDevice vk_device = {};
    VulkanFunctions vk_functions = {};
};

enum class eGPUResState {
    RenderTarget = 4,
    UnorderedAccess = 5,
    DepthRead = 6,
    DepthWrite = 7,
    ShaderResource = 9,
    CopyDst = 11,
    CopySrc = 12
};

/** Render region context,
    holds information for specific rectangle on image
*/
class RegionContext {
    /// Rectangle on image
    rect_t rect_;

  public:
    int iteration = 0; ///< Number of rendered samples per pixel
    int cache_iteration = 0;

    explicit RegionContext(const rect_t &rect) : rect_(rect) {}

    const rect_t &rect() const { return rect_; }

    /// Clear region context (used to start again)
    void Clear() { iteration = 0; }
};

class ILog;

struct GpuImage {
    union {
        VkImage vk_image;
        ID3D12Resource *dx_image;
    };
    union {
        VkImageView vk_image_view;
        struct {
            ID3D12DescriptorHeap *heap;
            uint32_t offset;
        } dx_image_view;
    };
    eGPUResState state;

    GpuImage() { memset(this, 0, sizeof(GpuImage)); }
    GpuImage(const VkImage _vk_image, const VkImageView _vk_image_view, const eGPUResState _state)
        : vk_image(_vk_image), vk_image_view(_vk_image_view), state(_state) {}
    GpuImage(ID3D12Resource *const _dx_image, ID3D12DescriptorHeap *const dx_view_heap, const uint32_t dx_view_offset,
             const eGPUResState _state)
        : dx_image(_dx_image), dx_image_view{dx_view_heap, dx_view_offset}, state(_state) {}
};

struct GpuCommandBuffer {
    union {
        VkCommandBuffer vk_cmd_buf;
        ID3D12GraphicsCommandList *dx_cmd_buf;
    };
    int index;

    GpuCommandBuffer() { memset(this, 0, sizeof(GpuCommandBuffer)); }
    GpuCommandBuffer(const VkCommandBuffer _vk_cmd_buf, const int _index) : vk_cmd_buf(_vk_cmd_buf), index(_index) {}
    GpuCommandBuffer(ID3D12GraphicsCommandList *const _dx_cmd_buf, const int _index)
        : dx_cmd_buf(_dx_cmd_buf), index(_index) {}
};

/** Base class for all renderer backends
 */
class RendererBase {
  public:
    virtual ~RendererBase() = default;

    /// Type of renderer
    virtual eRendererType type() const = 0;

    /// Log
    virtual ILog *log() const = 0;

    /// Name of the device
    virtual std::string_view device_name() const = 0;

    /// Tells whether this is a hardware accelerated renderer
    virtual bool is_hwrt() const { return false; }

    /// Tells whether spatial caching is enabled
    virtual bool is_spatial_caching_enabled() const { return false; }

    /// Returns size of rendered image
    virtual std::pair<int, int> size() const = 0;

    /// Returns pointer to rendered image
    virtual color_data_rgba_t get_pixels_ref() const = 0;

    /// Returns pointer to 'raw' untonemapped image
    virtual color_data_rgba_t get_raw_pixels_ref() const = 0;

    /// Returns pointer to auxiliary image buffers
    virtual color_data_rgba_t get_aux_pixels_ref(eAUXBuffer buf) const = 0;

    /// Returns pointer to SH data
    virtual const shl1_data_t *get_sh_data_ref() const = 0;

    /// Returns native GPU image that holds rendered pixels
    virtual GpuImage get_native_raw_pixels() const { return {}; }

    /// Allows to set native GPU image state
    virtual void set_native_raw_pixels_state(const eGPUResState) {}

    /// Allows to set native GPU command buffer to use
    virtual void set_command_buffer(const GpuCommandBuffer) {}

    /** @brief Resize framebuffer
        @param w new image width
        @param h new image height
    */
    virtual void Resize(int w, int h) = 0;

    /** @brief Clear framebuffer
        @param c color used to fill image
    */
    virtual void Clear(const color_rgba_t &c) = 0;

    /** @brief Create new scene
        @return pointer to new scene for specific backend
    */
    virtual SceneBase *CreateScene() = 0;

    /** @brief Render image region
        @param scene reference to a scene
        @param region image region to render
    */
    virtual void RenderScene(const SceneBase &scene, RegionContext &region) = 0;

    /** @brief Denoise image region using NLM filter
        @param region image region to denoise
    */
    virtual void DenoiseImage(const RegionContext &region) = 0;

    /** @brief Denoise image region using UNet filter
        @param pass UNet filter pass
        @param region image region to denoise
    */
    virtual void DenoiseImage(int pass, const RegionContext &region) = 0;

    /** @brief Update spatial radiance cache
        @param scene reference to a scene
        @param region image region to render
    */
    virtual void UpdateSpatialCache(const SceneBase &scene, RegionContext &region) = 0;

    /** @brief Resolve spatial radiance cache
        @param scene reference to a scene
    */
    virtual void ResolveSpatialCache(
        const SceneBase &scene,
        const std::function<void(int, int, ParallelForFunction &&)> &parallel_for = parallel_for_serial) = 0;

    /** @brief Reset spatial radiance cache
        @param scene reference to a scene
    */
    virtual void ResetSpatialCache(
        const SceneBase &scene,
        const std::function<void(int, int, ParallelForFunction &&)> &parallel_for = parallel_for_serial) = 0;

    /// Structure that holds render timings (in microseconds)
    struct stats_t {
        unsigned long long time_primary_ray_gen_us;
        unsigned long long time_primary_trace_us;
        unsigned long long time_primary_shade_us;
        unsigned long long time_primary_shadow_us;
        unsigned long long time_secondary_sort_us;
        unsigned long long time_secondary_trace_us;
        unsigned long long time_secondary_shade_us;
        unsigned long long time_secondary_shadow_us;
        unsigned long long time_denoise_us;
        unsigned long long time_cache_update_us;
        unsigned long long time_cache_resolve_us;
    };
    virtual void GetStats(stats_t &st) = 0;
    virtual void ResetStats() = 0;

    /** @brief Initialize UNet filter (neural denoiser)
        @param alias_memory enable tensom memory aliasing (to lower memory usage)
        @param out_props output filter properties
    */
    virtual unet_filter_properties_t
    InitUNetFilter(bool alias_memory,
                   const std::function<void(int, int, ParallelForFunction &&)> &parallel_for = parallel_for_serial) = 0;
};
} // namespace Ray