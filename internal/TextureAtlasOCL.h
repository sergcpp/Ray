#pragma once

#include "CoreOCL.h"
#include "TextureSplitter.h"

namespace ray {
namespace ocl {
class TextureAtlas {
    const cl::Context &context_;
    const cl::CommandQueue &queue_;

    cl::Image2DArray atlas_;

    const math::ivec2 res_;
    int pages_count_;

    std::vector<TextureSplitter> splitters_;
public:
    TextureAtlas(const cl::Context &context, const cl::CommandQueue &queue,
                 const math::ivec2 &res, int pages_count = 4);

    math::ivec2 res() const {
        return res_;
    }

    const cl::Image2DArray &atlas() const {
        return atlas_;
    }

    int Allocate(const pixel_color8_t *data, const math::ivec2 &res, math::ivec2 &pos);
    bool Free(int page, const math::ivec2 &pos);

    bool Resize(int pages_count);
};
}
}