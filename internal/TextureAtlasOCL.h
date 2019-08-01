#pragma once

#include "CoreOCL.h"
#include "TextureSplitter.h"

namespace Ray {
namespace Ocl {
class TextureAtlas {
    const cl::Context       &context_;
    const cl::CommandQueue  &queue_;

    cl::Image2DArray        atlas_;

    const int               res_[2];
    int                     page_count_;

    std::vector<TextureSplitter> splitters_;
public:
    TextureAtlas(const cl::Context &context, const cl::CommandQueue &queue,
                 int resx, int resy, int page_count = 1);

    const cl::Image2DArray &atlas() const {
        return atlas_;
    }

    int Allocate(const pixel_color8_t *data, const int res[2], int pos[2]);
    bool Free(int page, const int pos[2]);

    bool Resize(int pages_count);
};
}
}