#pragma once

namespace Ray {
namespace Vk {
class Context;
struct DebugMarker {
    explicit DebugMarker(Context *ctx, void *_cmd_buf, const char *name);
    ~DebugMarker();

    Context *ctx_ = nullptr;
    void *cmd_buf_ = nullptr;
};
} // namespace Vk
} // namespace Ray

inline Ray::Vk::DebugMarker::DebugMarker(Context *ctx, void *_cmd_buf, const char *name)
    : ctx_(ctx), cmd_buf_(_cmd_buf) {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;

    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(_cmd_buf);
    ctx_->api().vkCmdBeginDebugUtilsLabelEXT(cmd_buf, &label);
}

inline Ray::Vk::DebugMarker::~DebugMarker() {
    auto cmd_buf = reinterpret_cast<VkCommandBuffer>(cmd_buf_);
    ctx_->api().vkCmdEndDebugUtilsLabelEXT(cmd_buf);
}
