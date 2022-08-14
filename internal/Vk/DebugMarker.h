#pragma once

namespace Ray {
namespace Vk {
struct DebugMarker {
    explicit DebugMarker(void *_cmd_buf, const char *name);
    ~DebugMarker();

    void *cmd_buf_ = nullptr;
};
} // namespace Vk
} // namespace Ray

#include "VK.h"

inline Ray::Vk::DebugMarker::DebugMarker(void *_cmd_buf, const char *name) : cmd_buf_(_cmd_buf) {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;

    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);
    vkCmdBeginDebugUtilsLabelEXT(cmd_buf, &label);
}

inline Ray::Vk::DebugMarker::~DebugMarker() {
    VkCommandBuffer cmd_buf = reinterpret_cast<VkCommandBuffer>(cmd_buf_);
    vkCmdEndDebugUtilsLabelEXT(cmd_buf);
}
