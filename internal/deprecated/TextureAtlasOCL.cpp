#include "TextureAtlasOCL.h"

Ray::Ocl::TextureAtlas::TextureAtlas(const cl::Context &context, const cl::CommandQueue &queue, int resx, int resy,
                                     int pages_count)
    : context_(context), queue_(queue), res_{resx, resy}, page_count_(0) {
    if (!Resize(pages_count)) {
        throw std::runtime_error("TextureAtlas cannot be resized!");
    }
}

int Ray::Ocl::TextureAtlas::Allocate(const pixel_color8_t *data, const int _res[2], int pos[2]) {
    // add 1px border
    int res[2] = {_res[0] + 2, _res[1] + 2};

    if (res[0] > res_[0] || res[1] > res_[1]) {
        return -1;
    }

    for (int page = 0; page < page_count_; page++) {
        int index = splitters_[page].Allocate(res, &pos[0]);
        if (index != -1) {
            cl_int error =
                queue_.enqueueWriteImage(atlas_, CL_TRUE, {size_t(pos[0]) + 1, size_t(pos[1]) + 1, size_t(page)},
                                         {size_t(_res[0]), size_t(_res[1]), 1}, 0, 0, data);
            if (error != CL_SUCCESS) {
                return -1;
            }

            { // add 1px border
                error = queue_.enqueueWriteImage(atlas_, CL_TRUE, {size_t(pos[0]) + 1, size_t(pos[1]), size_t(page)},
                                                 {size_t(_res[0]), 1, 1}, 0, 0, &data[(_res[1] - 1) * _res[0]]);
                if (error != CL_SUCCESS) {
                    return -1;
                }

                error = queue_.enqueueWriteImage(atlas_, CL_TRUE,
                                                 {size_t(pos[0]) + 1, size_t(pos[1]) + res[1] - 1, size_t(page)},
                                                 {size_t(_res[0]), 1, 1}, 0, 0, &data[0]);
                if (error != CL_SUCCESS) {
                    return -1;
                }

                std::vector<pixel_color8_t> vertical_border(res[1]);
                vertical_border[0] = data[(_res[1] - 1) * _res[0] + _res[0] - 1];
                for (int i = 0; i < _res[1]; i++) {
                    vertical_border[i + 1] = data[i * _res[0] + _res[0] - 1];
                }
                vertical_border[res[1] - 1] = data[0 * _res[0] + _res[0] - 1];

                error = queue_.enqueueWriteImage(atlas_, CL_TRUE, {size_t(pos[0]), size_t(pos[1]), size_t(page)},
                                                 {1, size_t(res[1]), 1}, 0, 0, &vertical_border[0]);
                if (error != CL_SUCCESS) {
                    return -1;
                }

                vertical_border[0] = data[(_res[1] - 1) * _res[0]];
                for (int i = 0; i < _res[1]; i++) {
                    vertical_border[i + 1] = data[i * _res[0]];
                }
                vertical_border[res[1] - 1] = data[0];

                error = queue_.enqueueWriteImage(atlas_, CL_TRUE,
                                                 {size_t(pos[0]) + res[0] - 1, size_t(pos[1]), size_t(page)},
                                                 {1, size_t(res[1]), 1}, 0, 0, &vertical_border[0]);
                if (error != CL_SUCCESS) {
                    return -1;
                }
            }

            return page;
        }
    }

    Resize(page_count_ * 2);
    return Allocate(data, _res, pos);
}

bool Ray::Ocl::TextureAtlas::Free(const int page, const int pos[2]) {
    if (page < 0 || page > page_count_) {
        return false;
    }
    // TODO: fill with black in debug
    return splitters_[page].Free(pos);
}

bool Ray::Ocl::TextureAtlas::Resize(const int pages_count) {
    // if we shrink atlas, all redundant pages required to be empty
    for (int i = pages_count; i < page_count_; i++) {
        if (!splitters_[i].empty()) {
            return false;
        }
    }

    cl_int error = CL_SUCCESS;
    auto new_atlas = cl::Image2DArray(context_, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, {CL_RGBA, CL_UNORM_INT8},
                                      pages_count, res_[0], res_[1], 0, 0, nullptr, &error);
    if (error != CL_SUCCESS) {
        return false;
    }

    if (page_count_) {
        error =
            queue_.enqueueCopyImage(atlas_, new_atlas, {}, {}, {size_t(res_[0]), size_t(res_[1]), size_t(page_count_)});
        if (error != CL_SUCCESS) {
            return false;
        }
    }

    atlas_ = std::move(new_atlas);

    splitters_.resize(pages_count, TextureSplitter{res_});
    page_count_ = pages_count;

    return true;
}