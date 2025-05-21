#pragma once

#include "Config.h"
#include "Log.h"
#include "RendererBase.h"

/**
  @file Ray.h
*/

namespace Ray {
/// Null log, can be used to disable log output
extern LogNull g_null_log;
/// Standard log to stdout
extern LogStdout g_stdout_log;

/// Default renderer flags used to choose backend, by default tries to create gpu renderer first
const Bitmask<eRendererType> DefaultEnabledRenderTypes =
    Bitmask<eRendererType>{eRendererType::Reference} | eRendererType::SIMD_SSE2 | eRendererType::SIMD_AVX |
    eRendererType::SIMD_AVX2 | eRendererType::SIMD_NEON | eRendererType::Vulkan | eRendererType::DirectX12;

/** @brief Creates renderer
    @return pointer to created renderer
*/
RendererBase *
CreateRenderer(const settings_t &s, ILog *log = &g_null_log,
               const std::function<void(int, int, ParallelForFunction &&)> &parallel_for = parallel_for_serial,
               Bitmask<eRendererType> enabled_types = DefaultEnabledRenderTypes);

/** @brief Queries available GPU devices
    @param log output log
    @param out_devices output array of available devices
    @param capacity capacity of previous parameter array
    @return available devices count (writter into out_devices parameter)
*/
int QueryAvailableGPUDevices(ILog *log, gpu_device_t out_devices[], int capacity);

/** @brief Matches device name using regex pattern
    @param name name of the device
    @param pattern regex pattern
    @return true if name matches pattern
*/
bool MatchDeviceNames(std::string_view name, std::string_view pattern);

/** @brief Get version string
    @return version string
*/
const char *Version();
} // namespace Ray