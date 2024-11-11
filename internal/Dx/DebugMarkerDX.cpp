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

#include "../../Config.h"

#ifdef ENABLE_PIX
#include <WinPixEventRuntime/pix3.h>
#endif

Ray::Dx::DebugMarker::DebugMarker(Context *, ID3D12GraphicsCommandList *_cmd_buf, const char *name)
    : cmd_buf_(_cmd_buf) {
#ifdef ENABLE_GPU_DEBUG
#ifdef ENABLE_PIX
    PIXBeginEvent(cmd_buf_, 0, name);
#else
    std::wstring wstr(name, name + strlen(name));
    cmd_buf_->BeginEvent(0, wstr.c_str(), UINT(wstr.length() * sizeof(wchar_t)));
#endif
#endif
}

Ray::Dx::DebugMarker::~DebugMarker() {
#ifdef ENABLE_GPU_DEBUG
#ifdef ENABLE_PIX
    PIXEndEvent(cmd_buf_);
#else
    cmd_buf_->EndEvent();
#endif
#endif
}
