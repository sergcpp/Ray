#include "DebugMarkerDX.h"

#include <cassert>
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

Ray::Dx::DebugMarker::DebugMarker(Context *, ID3D12GraphicsCommandList *_cmd_buf, std::string_view name)
    : cmd_buf_(_cmd_buf) {
#ifdef ENABLE_GPU_DEBUG
#ifdef ENABLE_PIX
    PIXBeginEvent(cmd_buf_, 0, name.data());
#else
    const int req_size = MultiByteToWideChar(CP_UTF8, 0, name.data(), -1, nullptr, 0);
    assert(req_size > 0);

    std::wstring wstr(req_size, 0);
    MultiByteToWideChar(CP_UTF8, 0, name.data(), -1, wstr.data(), req_size);
    wstr.pop_back();

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
