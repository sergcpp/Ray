#include "ContextMtl.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "../../Log.h"
#include "../../Types.h"

namespace Ray::Mtl {
static std::mutex g_cb_registry_mtx;
static std::unordered_map<MTL::CommandBuffer *, Context *> g_cb_registry;

Context *Context::LookupForCommandBuffer(MTL::CommandBuffer *cb) {
    const std::lock_guard<std::mutex> _(g_cb_registry_mtx);
    const auto it = g_cb_registry.find(cb);
    return it != g_cb_registry.end() ? it->second : nullptr;
}

Context::~Context() { Destroy(); }

bool Context::Init(ILog *log, std::string_view preferred_device, const int) {
    log_ = log; // Metal validation (frame capture / shader warnings) not wired up yet

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    NS::Array *devices = MTL::CopyAllDevices();
    if (!devices || devices->count() == 0) {
        if (devices) {
            devices->release();
        }
        MTL::Device *def = MTL::CreateSystemDefaultDevice(); // +1: owned by us
        if (def) {
            const NS::Object *objs[] = {def};
            devices = NS::Array::alloc()->init(objs, 1);
            def->release();
        } else {
            devices = NS::Array::alloc()->init();
        }
    }

    device_ = nullptr;
    if (!preferred_device.empty()) {
        NS::String *wanted = NS::String::string(std::string(preferred_device).c_str(), NS::UTF8StringEncoding);
        for (NS::UInteger i = 0; i < devices->count(); i++) {
            MTL::Device *dev = static_cast<MTL::Device *>(devices->object(i));
            if (dev->name()->rangeOfString(wanted, NS::CaseInsensitiveSearch).location != NS::NotFound) {
                device_ = dev->retain();
                break;
            }
        }
    }
    if (!device_ && !preferred_device.empty()) {
        log->Warning("No Metal device matching \"%s\", falling back to the default Metal device",
                     std::string(preferred_device).c_str());
    }
    if (!device_ && devices->count() > 0) {
        MTL::Device *def = MTL::CreateSystemDefaultDevice();
        device_ = def ? def : static_cast<MTL::Device *>(devices->object(0))->retain();
    }
    devices->release();

    if (!device_) {
        log->Error("Failed to find a Metal-capable device!");
        pool->release();
        return false;
    }

    cmd_queue_ = device_->newCommandQueue();
    if (!cmd_queue_) {
        log->Error("Failed to create Metal command queue!");
        device_->release();
        device_ = nullptr;
        pool->release();
        return false;
    }

    astc_supported_ = device_->supportsFamily(MTL::GPUFamilyApple7); // ASTC LDR is native on Apple7+
    bc_supported_ = device_->supportsBCTextureCompression();

    default_mem_allocs_ = std::make_unique<MemAllocators>("Default Allocs", 32 * 1024 * 1024, 1.5f, 128 * 1024 * 1024);

    device_name_ = device_->name()->utf8String();
    log->Info("Metal device: %s (recommendedMaxWorkingSetSize %llu MB, ASTC %i, BC %i)", device_->name()->utf8String(),
              device_->recommendedMaxWorkingSetSize() >> 20, int(astc_supported_), int(bc_supported_));

    pool->release();
    return true;
}

void Context::Destroy() {
    EndEncoders();
    if (cmd_queue_) {
        // one-shot empty buffer to drain the pipe; commandBuffer() returns a +0 accessor
        // object - own it before the pool drains, symmetric with BeginSingleTimeCommands
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer *cb = cmd_queue_->commandBuffer();
        cb->retain();
        pool->release();
        cb->commit();
        cb->waitUntilCompleted();
        cb->release();
    }

    for (int i = 0; i < MaxFramesInFlight; i++) {
        DestroyDeferredResources(i);
        if (uniform_ring_[i]) {
            uniform_ring_[i]->release();
            uniform_ring_[i] = nullptr;
        }
        uniform_ring_pos_[i] = 0;
    }

    if (dummy_sampler_) {
        dummy_sampler_->release();
        dummy_sampler_ = nullptr;
    }

    default_mem_allocs_ = {};
    if (cmd_queue_) {
        cmd_queue_->release();
        cmd_queue_ = nullptr;
    }
    if (device_) {
        device_->release();
        device_ = nullptr;
    }
    log_ = nullptr;
}

// CommandBuffer encoder factories return +0 (autoreleased) ObjC objects, but we hold them
// until EndEncoders: take an owning ref before the autorelease pool drains, else the later
// release() over-frees and the stale pool entry SIGSEGVs on worker-thread exit.
template <typename F> static auto *OwnedEncoder(F &&create) {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    auto *enc = create();
    if (enc) {
        enc->retain();
    }
    pool->release();
    return enc;
}

MTL::ComputeCommandEncoder *Context::ComputeEncoder(const CommandBuffer cmd_buf) {
    if (encoded_cmd_buf_ != cmd_buf) {
        EndEncoders();
    }
    encoded_cmd_buf_ = cmd_buf;

    if (blit_encoder_) {
        blit_encoder_->endEncoding();
        blit_encoder_->release();
        blit_encoder_ = nullptr;
    }
    if (render_encoder_) {
        render_encoder_->endEncoding();
        render_encoder_->release();
        render_encoder_ = nullptr;
    }
    if (!compute_encoder_) {
        compute_encoder_ = OwnedEncoder([&] { return cmd_buf->computeCommandEncoder(); });
    }
    return compute_encoder_;
}

MTL::BlitCommandEncoder *Context::BlitEncoder(const CommandBuffer cmd_buf) {
    if (encoded_cmd_buf_ != cmd_buf) {
        EndEncoders();
    }
    encoded_cmd_buf_ = cmd_buf;

    if (compute_encoder_) {
        compute_encoder_->endEncoding();
        compute_encoder_->release();
        compute_encoder_ = nullptr;
    }
    if (render_encoder_) {
        render_encoder_->endEncoding();
        render_encoder_->release();
        render_encoder_ = nullptr;
    }
    if (!blit_encoder_) {
        blit_encoder_ = OwnedEncoder([&] { return cmd_buf->blitCommandEncoder(); });
    }
    return blit_encoder_;
}

// Draw-free render pass: the attachment load action is Metal's only clear primitive
// (fillTexture was removed with Metal 3, and clearColorUI is not in the metal-cpp binding).
MTL::RenderCommandEncoder *Context::RenderEncoder(const CommandBuffer cmd_buf, MTL::RenderPassDescriptor *rpd) {
    if (encoded_cmd_buf_ != cmd_buf) {
        EndEncoders();
    }
    encoded_cmd_buf_ = cmd_buf;

    if (compute_encoder_) {
        compute_encoder_->endEncoding();
        compute_encoder_->release();
        compute_encoder_ = nullptr;
    }
    if (blit_encoder_) {
        blit_encoder_->endEncoding();
        blit_encoder_->release();
        blit_encoder_ = nullptr;
    }
    if (!render_encoder_) {
        render_encoder_ = OwnedEncoder([&] { return cmd_buf->renderCommandEncoder(rpd); });
    }
    return render_encoder_;
}

void Context::EndEncoders() {
    if (compute_encoder_) {
        compute_encoder_->endEncoding();
        compute_encoder_->release();
        compute_encoder_ = nullptr;
    }
    if (blit_encoder_) {
        blit_encoder_->endEncoding();
        blit_encoder_->release();
        blit_encoder_ = nullptr;
    }
    if (render_encoder_) {
        render_encoder_->endEncoding();
        render_encoder_->release();
        render_encoder_ = nullptr;
    }
    encoded_cmd_buf_ = nullptr;
}

CommandBuffer Context::BeginSingleTimeCommands() {
    // Metal requires an autorelease pool around command-buffer creation (callers are
    // worker threads); the returned buffer is owned by our retain until End
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    CommandBuffer cmd_buf = cmd_queue_->commandBuffer();
    cmd_buf->retain(); // held until EndSingleTimeCommands, mirrors the VK/DX helper lifetimes
    pool->release();
    {
        const std::lock_guard<std::mutex> _(g_cb_registry_mtx);
        g_cb_registry[cmd_buf] = this; // lets TransitionResourceStates find the encoder owner
    }
    return cmd_buf;
}

void Context::EndSingleTimeCommands(CommandBuffer cmd_buf) {
    {
        const std::lock_guard<std::mutex> _(g_cb_registry_mtx);
        g_cb_registry.erase(cmd_buf);
    }
    EndEncoders();
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
    const CFTimeInterval gs = cmd_buf->GPUStartTime(), ge = cmd_buf->GPUEndTime(); // 0 when n/a
    if (gs > 0.0 && ge > gs) {
        gpu_busy_us_ += uint64_t((ge - gs) * 1e6);
    }
    if (NS::Error *err = cmd_buf->error()) {
        log_->Error("Metal command buffer error: %s", err->localizedDescription()->utf8String());
    }
    cmd_buf->release();
}

MTL::SamplerState *Context::dummy_sampler() {
    if (!dummy_sampler_) {
        MTL::SamplerDescriptor *desc = MTL::SamplerDescriptor::alloc()->init();
        dummy_sampler_ = device_->newSamplerState(desc);
        desc->release();
    }
    return dummy_sampler_;
}

MTL::Buffer *Context::UniformSlice(const uint32_t size, uint32_t &out_offset) {
    const int i = backend_frame;
    if (!uniform_ring_[i]) {
        uniform_ring_[i] = device_->newBuffer(UniformRingSize, MTL::StorageModeShared);
    }

    const uint32_t pos = (uniform_ring_pos_[i] + 255u) & ~255u;
    if (pos + size > UniformRingSize) {
        log()->Error("Uniform params ring overflow (frame %d)", i);
        return nullptr;
    }
    uniform_ring_pos_[i] = pos + size;
    out_offset = pos;
    return uniform_ring_[i];
}

void Context::ResetUniformRing() {
    for (int i = 0; i < MaxFramesInFlight; i++) {
        uniform_ring_pos_[i] = 0;
    }
}

int Context::WriteTimestamp(CommandBuffer, const bool) {
    // coarse CPU stamps; exact GPU timestamps would need MTLCounterSampleBuffer. Boundless frame counts
    // would grow ts_cpu_ unbounded - on cap, restart the epoch (stats degrade to 0 for one frame).
    if (ts_cpu_.size() >= (1u << 21)) {
        ts_cpu_.clear();
        ts_segs_.clear();
        open_seg_ = {-1, -1, 0, 0};
    }
    if (open_seg_.lo_id < 0) {
        open_seg_ = {int(ts_cpu_.size()), -1, 0, 0};
    }
    ts_cpu_.push_back(uint64_t(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()));
    return int(ts_cpu_.size()) - 1;
}

uint64_t Context::GetTimestampIntervalDurationUs(const int query_start, const int query_end) const {
    if (query_start < 0 || query_end < 0 || query_end < query_start || size_t(query_end) >= ts_cpu_.size()) {
        return 0;
    }
    for (auto it = ts_segs_.rbegin(); it != ts_segs_.rend(); ++it) {
        if (it->lo_id <= query_start && query_end <= it->hi_id) {
            if (it->gpu_busy_us == 0 || it->span_us == 0) {
                return 0;
            }
            return (ts_cpu_[size_t(query_end)] - ts_cpu_[size_t(query_start)]) * it->gpu_busy_us / it->span_us;
        }
    }
    return 0;
}

bool Context::ReadbackTimestampQueries(const int) {
    if (open_seg_.lo_id >= 0 && !ts_cpu_.empty()) {
        open_seg_.hi_id = int(ts_cpu_.size()) - 1;
        open_seg_.span_us = ts_cpu_[size_t(open_seg_.hi_id)] - ts_cpu_[size_t(open_seg_.lo_id)];
        open_seg_.gpu_busy_us = gpu_busy_us_;
        ts_segs_.push_back(open_seg_);
        if (ts_segs_.size() > 8) {
            ts_segs_.erase(ts_segs_.begin(), ts_segs_.end() - 8); // stats read at most the previous frame's segs
        }
        open_seg_ = {-1, -1, 0, 0};
    }
    gpu_busy_us_ = 0;
    return true;
}

void Context::DestroyDeferredResources(const int i) {
    for (MTL::Buffer *buf : bufs_to_destroy[i]) {
        buf->release();
    }
    bufs_to_destroy[i].clear();
    for (MTL::Texture *img : images_to_destroy[i]) {
        img->release();
    }
    images_to_destroy[i].clear();
    for (MTL::SamplerState *s : samplers_to_destroy[i]) {
        s->release();
    }
    samplers_to_destroy[i].clear();
}

int Context::QueryAvailableDevices(ILog *, gpu_device_t out_devices[], const int capacity) {
    int n = 0;

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    NS::Array *devices = MTL::CopyAllDevices();

    for (NS::UInteger i = 0; devices && i < devices->count(); i++) {
        if (n >= capacity) {
            break;
        }
        MTL::Device *dev = static_cast<MTL::Device *>(devices->object(i));
        strncpy(out_devices[n].name, dev->name()->utf8String(), sizeof(gpu_device_t::name) - 1);
        out_devices[n].name[sizeof(gpu_device_t::name) - 1] = '\0';
        n++;
    }
    if (devices) {
        devices->release();
    }
    pool->release();

    return n;
}
} // namespace Ray::Mtl
