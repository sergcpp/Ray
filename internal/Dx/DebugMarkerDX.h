#pragma once

struct ID3D12GraphicsCommandList;

namespace Ray::Dx {
class Context;
struct DebugMarker {
    explicit DebugMarker(Context *ctx, ID3D12GraphicsCommandList *cmd_buf, const char *name);
    ~DebugMarker();

    ID3D12GraphicsCommandList *cmd_buf_ = nullptr;
};
} // namespace Ray::Dx
