#pragma once

#include "../../Config.h"

namespace Ray {
namespace Vk {
class Context;
struct DebugMarker {
    explicit DebugMarker(Context *ctx, VkCommandBuffer cmd_buf, const char *name);
    ~DebugMarker();

    Context *ctx_ = nullptr;
    VkCommandBuffer cmd_buf_ = {};
};
} // namespace Vk
} // namespace Ray

inline Ray::Vk::DebugMarker::DebugMarker(Context *ctx, VkCommandBuffer _cmd_buf, const char *name)
    : ctx_(ctx), cmd_buf_(_cmd_buf) {
#ifdef ENABLE_DEBUG_MARKERS
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;

    ctx_->api().vkCmdBeginDebugUtilsLabelEXT(cmd_buf_, &label);
#endif
}

inline Ray::Vk::DebugMarker::~DebugMarker() {
#ifdef ENABLE_DEBUG_MARKERS
    ctx_->api().vkCmdEndDebugUtilsLabelEXT(cmd_buf_);
#endif
}
