#pragma once

struct ID3D12GraphicsCommandList;

namespace Ray {
namespace Dx {
struct DebugMarker {
    explicit DebugMarker(void *_cmd_buf, const char *name);
    ~DebugMarker();

    ID3D12GraphicsCommandList *cmd_buf_ = nullptr;
};
} // namespace Vk
} // namespace Ray
