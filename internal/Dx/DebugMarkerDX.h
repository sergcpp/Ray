#pragma once

#include <string_view>

struct ID3D12GraphicsCommandList;

namespace Ray::Dx {
class Context;
struct DebugMarker {
    explicit DebugMarker(Context *ctx, ID3D12GraphicsCommandList *cmd_buf, std::string_view name);
    ~DebugMarker();

    ID3D12GraphicsCommandList *cmd_buf_ = nullptr;
};
} // namespace Ray::Dx
