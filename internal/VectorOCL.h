#pragma once

#include "CoreOCL.h"

namespace Ray {
namespace Ocl {
template <typename T>
class Vector {
    const cl::Context &context_;
    const cl::CommandQueue &queue_;
    size_t max_img_buf_size_;
    cl_mem_flags flags_;
    cl::Buffer buf_;
    cl::Image1DBuffer img_buf_;
    size_t size_, cap_;
public:
    Vector(cl_mem_flags flags, size_t capacity = 16)
        : Vector(cl::Context::getDefault(), cl::CommandQueue::getDefault(), flags, capacity) {
    }
    Vector(const cl::Context &context, const cl::CommandQueue &queue, cl_mem_flags flags, size_t capacity = 16, size_t max_img_buf_size = 0)
        : context_(context), queue_(queue), flags_(flags), size_(0), cap_(capacity), max_img_buf_size_(max_img_buf_size){
        cl_int error = CL_SUCCESS;
        buf_ = cl::Buffer(context_, flags_, sizeof(T) * cap_, nullptr, &error);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot allocate OpenCL buffer!");
    }

    const cl::Buffer &buf() const {
        return buf_;
    }

    const cl::Image1DBuffer &img_buf() const {
        return img_buf_;
    }

    size_t size() const {
        return size_;
    }

    void Reserve(size_t req_cap) {
        if (cap_ < req_cap) {
            cl_int error = CL_SUCCESS;

            while (cap_ < req_cap) cap_ *= 2;

            cl::Buffer new_buf = cl::Buffer(context_, flags_, sizeof(T) * cap_, nullptr, &error);
            if (error != CL_SUCCESS) throw std::runtime_error("Cannot allocate OpenCL buffer!");

            if (size_) {
                error = queue_.enqueueCopyBuffer(buf_, new_buf, 0, 0, sizeof(T) * size_);
                if (error != CL_SUCCESS) throw std::runtime_error("Cannot copy OpenCL buffer!");
            }

            img_buf_ = {};
            buf_ = std::move(new_buf);
            if (sizeof(T) % 16 == 0 && ((sizeof(T) / 16)) * cap_ <= max_img_buf_size_) {
                img_buf_ = cl::Image1DBuffer(context_, 0, { CL_RGBA, CL_UNSIGNED_INT32 }, (sizeof(T) / 16) * cap_, buf_, &error);
                if (error != CL_SUCCESS) throw std::runtime_error("Cannot create image buffer!");
            }
        }
    }

    void Resize(size_t new_size) {
        Reserve(new_size);

        size_ = new_size;
    }

    void Append(const T *vec, size_t num) {
        Reserve(size_ + num);

        cl_int error = CL_SUCCESS;
        error = queue_.enqueueWriteBuffer(buf_, CL_TRUE, sizeof(T) * size_, sizeof(T) * num, vec);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot write OpenCL buffer!");

        size_ += num;
    }

    void PushBack(const T &v) {
        Append(&v, 1);
    }

    void Erase(size_t i) {
        Erase(i, 1);
    }

    void Erase(size_t offset, size_t count) {
#ifndef NDEBUG
        if (offset + count > size_) throw std::out_of_range("VectorOCL::Erase");
#endif
        if (offset + count != size_) {
            size_t pos = offset;
            size_t to_copy = size_ - offset - count;

            while (to_copy) {
                size_t portion = std::min(count, to_copy);

                cl_int error = queue_.enqueueCopyBuffer(buf_, buf_, sizeof(T) * (pos + count), sizeof(T) * pos, sizeof(T) * portion);
                if (error != CL_SUCCESS) throw std::runtime_error("Cannot access OpenCL buffer!");

                pos += portion;
                to_copy -= portion;
            }
        }

        size_ -= count;
    }

    void Clear() {
        size_ = 0;
    }

    void Get(size_t i, T &v) {
#ifndef NDEBUG
        if (i >= size_) throw std::out_of_range("VectorOCL::Get");
#endif
        cl_int error = queue_.enqueueReadBuffer(buf_, CL_TRUE, sizeof(T) * i, sizeof(T), &v);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot read OpenCL buffer!");
    }

    void Get(T *p, size_t offset, size_t count) {
#ifndef NDEBUG
        if (offset + count > size_) throw std::out_of_range("VectorOCL::Get");
#endif
        cl_int error = queue_.enqueueReadBuffer(buf_, CL_TRUE, sizeof(T) * offset, sizeof(T) * count, p);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot read OpenCL buffer!");
    }

    void Set(size_t i, const T &v) {
#ifndef NDEBUG
        if (i >= size_) throw std::out_of_range("VectorOCL::Set");
#endif
        cl_int error = queue_.enqueueWriteBuffer(buf_, CL_TRUE, sizeof(T) * i, sizeof(T), &v);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot write OpenCL buffer!");
    }

    void Set(const T *p, size_t offset, size_t count) {
#ifndef NDEBUG
        if (offset + count > size_) throw std::out_of_range("VectorOCL::Set");
#endif
        cl_int error = queue_.enqueueWriteBuffer(buf_, CL_TRUE, sizeof(T) * offset, sizeof(T) * count, p);
        if (error != CL_SUCCESS) throw std::runtime_error("Cannot write OpenCL buffer!");
    }
};
}
}