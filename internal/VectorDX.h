#pragma once

#include "CoreDX.h"
#include "Dx/BufferDX.h"

#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant

namespace Ray {
namespace Dx {
template <typename T> class Vector {
    Context *ctx_ = nullptr;
    mutable Buffer buf_;
    size_t size_, cap_;

  public:
    explicit Vector(Context *ctx, const char *name, const size_t capacity = 16) : ctx_(ctx), size_(0), cap_(capacity) {
        buf_ = Buffer{name, ctx_, eBufType::Storage, uint32_t(sizeof(T) * cap_), sizeof(T)};
    }

    const Buffer &buf() const { return buf_; }

    size_t size() const { return size_; }

    void Reserve(const size_t req_cap) {
        if (cap_ < req_cap) {
            while (cap_ < req_cap) {
                cap_ *= 2;
            }

            buf_.Resize(uint32_t(sizeof(T) * cap_));
        }
    }

    void Resize(size_t new_size) {
        Reserve(new_size);
        size_ = new_size;
    }

    void Append(const T *vec, size_t num) {
        Reserve(size_ + num);

        { // Write buffer
            Buffer temp_stage_buf{"Temp Stage", ctx_, eBufType::Upload, uint32_t(sizeof(T) * num), uint32_t(sizeof(T))};

            { // Prepare stage buffer
                uint8_t *ptr = temp_stage_buf.Map();
                memcpy(ptr, vec, sizeof(T) * num);
                temp_stage_buf.Unmap();
            }

            ID3D12GraphicsCommandList *cmd_buf =
                BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());

            CopyBufferToBuffer(temp_stage_buf, 0, buf_, uint32_t(sizeof(T) * size_), uint32_t(sizeof(T) * num),
                               cmd_buf);

            EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf,
                                  ctx_->temp_command_pool());

            temp_stage_buf.FreeImmediate();
        }

        size_ += num;
    }

    void PushBack(const T &v) { Append(&v, 1); }

    void Erase(const size_t i) { Erase(i, 1); }

    void Erase(const size_t offset, const size_t count) {
#ifndef NDEBUG
        if (offset + count > size_) {
            throw std::out_of_range("VectorVK::Erase");
        }
#endif
        if (offset + count != size_) {
            size_t pos = offset;
            size_t to_copy = size_ - offset - count;

            while (to_copy) {
                const size_t portion = std::min(count, to_copy);

                // const cl_int error = queue_.enqueueCopyBuffer(buf_, buf_, sizeof(T) * (pos + count), sizeof(T) * pos,
                //                                               sizeof(T) * portion);
                // if (error != CL_SUCCESS) {
                //     throw std::runtime_error("Cannot access OpenCL buffer!");
                // }

                pos += portion;
                to_copy -= portion;
            }
        }

        size_ -= count;
    }

    void Clear() { size_ = 0; }

    void Get(const size_t i, T &v) const {
#ifndef NDEBUG
        if (i >= size_) {
            throw std::out_of_range("VectorVK::Get");
        }
#endif

        Buffer temp_stage_buf{"Temp Stage", ctx_, eBufType::Readback, uint32_t(sizeof(T)), uint32_t(sizeof(T))};

        ID3D12GraphicsCommandList *cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        CopyBufferToBuffer(buf_, uint32_t(sizeof(T) * i), temp_stage_buf, 0, uint32_t(sizeof(T)), cmd_buf);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        const uint8_t *ptr = temp_stage_buf.Map();
        memcpy(&v, ptr, sizeof(T));
        temp_stage_buf.Unmap();
    }

    void Get(T *p, const size_t offset, const size_t count) const {
#ifndef NDEBUG
        if (offset + count > size_) {
            throw std::out_of_range("VectorVK::Get");
        }
#endif

        Buffer temp_stage_buf{"Temp Stage", ctx_, eBufType::Readback, uint32_t(sizeof(T) * count), uint32_t(sizeof(T))};

        ID3D12GraphicsCommandList *cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        CopyBufferToBuffer(buf_, uint32_t(sizeof(T) * offset), temp_stage_buf, 0, uint32_t(sizeof(T) * count), cmd_buf);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());

        const uint8_t *ptr = temp_stage_buf.Map();
        memcpy(p, ptr, sizeof(T) * count);
        temp_stage_buf.Unmap();
    }

    void Set(const size_t i, const T &v) {
#ifndef NDEBUG
        if (i >= size_) {
            throw std::out_of_range("VectorVK::Set");
        }
#endif

        Buffer temp_stage_buf{"Temp Stage", ctx_, eBufType::Upload, uint32_t(sizeof(T)), uint32_t(sizeof(T))};

        uint8_t *ptr = temp_stage_buf.Map();
        memcpy(ptr, &v, sizeof(T));
        temp_stage_buf.Unmap();

        ID3D12GraphicsCommandList *cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        CopyBufferToBuffer(temp_stage_buf, 0, buf_, uint32_t(sizeof(T) * i), uint32_t(sizeof(T)), cmd_buf);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }

    void Set(const T *p, const size_t offset, const size_t count) {
#ifndef NDEBUG
        if (offset + count > size_) {
            throw std::out_of_range("VectorVK::Set");
        }
#endif

        Buffer temp_stage_buf{"Temp Stage", ctx_, eBufType::Upload, uint32_t(sizeof(T) * count), uint32_t(sizeof(T))};

        uint8_t *ptr = temp_stage_buf.Map();
        memcpy(ptr, p, sizeof(T) * count);
        temp_stage_buf.Unmap();

        ID3D12GraphicsCommandList *cmd_buf = BegSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->temp_command_pool());
        CopyBufferToBuffer(temp_stage_buf, 0, buf_, uint32_t(sizeof(T) * offset), uint32_t(sizeof(T) * count), cmd_buf);
        EndSingleTimeCommands(ctx_->api(), ctx_->device(), ctx_->graphics_queue(), cmd_buf, ctx_->temp_command_pool());
    }
};
} // namespace Dx
} // namespace Ray

#pragma warning(pop)
