#pragma once

#include "../../Config.h"

namespace Ray::Vk {
class Context;
struct DebugMarker {
    explicit DebugMarker(Context *ctx, VkCommandBuffer cmd_buf, const char *name);
    ~DebugMarker() { End(); }

    void End();

    const Api &api_;
    VkCommandBuffer cmd_buf_ = {};
};
} // namespace Ray::Vk

inline Ray::Vk::DebugMarker::DebugMarker(Context *ctx, VkCommandBuffer cmd_buf, const char *name)
    : api_(ctx->api()), cmd_buf_(cmd_buf) {
#ifdef ENABLE_GPU_DEBUG
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;

    api_.vkCmdBeginDebugUtilsLabelEXT(cmd_buf_, &label);
#endif
}

inline void Ray::Vk::DebugMarker::End() {
#ifdef ENABLE_GPU_DEBUG
    if (cmd_buf_) {
        api_.vkCmdEndDebugUtilsLabelEXT(cmd_buf_);
        cmd_buf_ = {};
    }
#endif
}