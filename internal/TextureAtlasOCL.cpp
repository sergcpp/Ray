#include "TextureAtlasOCL.h"

ray::ocl::TextureAtlas::TextureAtlas(const cl::Context &context, const cl::CommandQueue &queue,
                                     const math::ivec2 &res, int pages_count)
    : context_(context), queue_(queue), res_(res), pages_count_(0) {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

int ray::ocl::TextureAtlas::Allocate(const pixel_color8_t *data, const math::ivec2 &_res, math::ivec2 &pos) {
    // add 1px border
    math::ivec2 res = _res + math::ivec2{ 2, 2 };

    if (res.x > res_.x || res.y > res_.y) return -1;

    for (int page = 0; page < pages_count_; page++) {
        int index = splitters_[page].Allocate(res, pos);
        if (index != -1) {
            cl_int error = queue_.enqueueWriteImage(atlas_, CL_TRUE, { (size_t)pos.x + 1, (size_t)pos.y + 1, (size_t)page }, { (size_t)_res.x, (size_t)_res.y, 1 }, 0, 0, data);
            if (error != CL_SUCCESS) return -1;

            {
                // add 1px border
                error = queue_.enqueueWriteImage(atlas_, CL_TRUE, { (size_t)pos.x + 1, (size_t)pos.y, (size_t)page }, { (size_t)_res.x, 1, 1 }, 0, 0, &data[(_res.y - 1) * _res.x]);
                if (error != CL_SUCCESS) return -1;

                error = queue_.enqueueWriteImage(atlas_, CL_TRUE, { (size_t)pos.x + 1, (size_t)pos.y + res.y - 1, (size_t)page }, { (size_t)_res.x, 1, 1 }, 0, 0, &data[0]);
                if (error != CL_SUCCESS) return -1;

                std::vector<pixel_color8_t> vertical_border(res.y);
                vertical_border[0] = data[(_res.y - 1) * _res.x + _res.y - 1];
                for (int i = 0; i < _res.y; i++) {
                    vertical_border[i + 1] = data[i * _res.x + _res.y - 1];
                }
                vertical_border[res.y - 1] = data[0 * _res.x + _res.y - 1];

                error = queue_.enqueueWriteImage(atlas_, CL_TRUE, { (size_t)pos.x, (size_t)pos.y, (size_t)page }, { 1, (size_t)res.y, 1 }, 0, 0, &vertical_border[0]);
                if (error != CL_SUCCESS) return -1;

                vertical_border[0] = data[(_res.y - 1) * _res.x];
                for (int i = 0; i < _res.y; i++) {
                    vertical_border[i + 1] = data[i * _res.x];
                }
                vertical_border[res.y - 1] = data[0];

                error = queue_.enqueueWriteImage(atlas_, CL_TRUE, { (size_t)pos.x + res.x - 1, (size_t)pos.y, (size_t)page }, { 1, (size_t)res.y, 1 }, 0, 0, &vertical_border[0]);
                if (error != CL_SUCCESS) return -1;
            }

            return page;
        }
    }

    Resize(pages_count_ * 2);
    return Allocate(data, _res, pos);
}

bool ray::ocl::TextureAtlas::Free(int page, const math::ivec2 &pos) {
    if (page < 0 || page > pages_count_) return false;
    return splitters_[page].Free(pos);
}

bool ray::ocl::TextureAtlas::Resize(int pages_count) {
    for (int i = pages_count; i < pages_count_; i++) {
        if (!splitters_[i].empty()) return false;
    }

    cl_int error = CL_SUCCESS;
    auto new_atlas = cl::Image2DArray(context_, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, { CL_RGBA, CL_UNORM_INT8 },
                                      pages_count, res_.x, res_.y, 0, 0, nullptr, &error);
    if (error != CL_SUCCESS) return false;

    if (pages_count_) {
        error = queue_.enqueueCopyImage(atlas_, new_atlas, {}, {}, { (size_t)res_.x, (size_t)res_.y, (size_t)pages_count_ });
        if (error != CL_SUCCESS) return false;
    }

    atlas_ = std::move(new_atlas);

	splitters_.resize(pages_count, TextureSplitter{ res_ });
    pages_count_ = pages_count;

    return true;
}