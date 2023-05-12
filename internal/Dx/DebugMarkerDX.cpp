#include "DebugMarkerDX.h"

#include <codecvt>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <d3d12.h>

Ray::Dx::DebugMarker::DebugMarker(void *_cmd_buf, const char *name) : cmd_buf_((ID3D12GraphicsCommandList *)_cmd_buf) {
    std::wstring wstr(name, name + strlen(name));
    cmd_buf_->BeginEvent(0, wstr.c_str(), UINT(wstr.length() * sizeof(wchar_t)));
}

Ray::Dx::DebugMarker::~DebugMarker() { cmd_buf_->EndEvent(); }
