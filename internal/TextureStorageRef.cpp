#include "TextureStorageRef.h"

#include <cstring>

#include <algorithm> // for std::max

template <typename T, int N>
int Ray::Ref::TexStorageLinear<T, N>::Allocate(const ColorType data[], const int _res[2], const bool mips) {
    int index = -1;
    if (!free_slots_.empty()) {
        index = free_slots_.back();
        free_slots_.pop_back();
    } else {
        index = int(images_.size());
        images_.resize(images_.size() + 1);
    }

    const int res[2] = {_res[0] + 1, _res[1] + 1};

    ImgData &p = images_[index];

    p.lod_offsets[0] = 0;
    p.res[0][0] = res[0];
    p.res[0][1] = res[1];

    int total_size = (res[0] * res[1]);

    for (int i = 1; i < NUM_MIP_LEVELS; ++i) {
        if (mips && (p.res[i - 1][0] > 1 || p.res[i - 1][1] > 1)) {
            p.lod_offsets[i] = total_size;
            p.res[i][0] = p.res[i - 1][0] / 2 + 1;
            p.res[i][1] = p.res[i - 1][1] / 2 + 1;
            total_size += (p.res[i][0] * p.res[i][1]);
        } else {
            p.lod_offsets[i] = p.lod_offsets[i - 1];
            p.res[i][0] = p.res[i - 1][0];
            p.res[i][1] = p.res[i - 1][1];
        }
    }

    p.pixels.reset(new ColorType[total_size]);

    for (int y = 0; y < _res[1]; ++y) {
        memcpy(&p.pixels[y * res[0]], &data[y * _res[0]], _res[0] * sizeof(ColorType));
        p.pixels[y * res[0] + _res[0]] = data[y * _res[0]];
    }

    memcpy(&p.pixels[_res[1] * res[0]], data, _res[0] * sizeof(ColorType));
    p.pixels[_res[1] * res[0] + _res[0]] = data[0];

    p.lod_offsets[0] = 0;
    p.res[0][0] = res[0];
    p.res[0][1] = res[1];

    for (int i = 1; i < NUM_MIP_LEVELS && mips; ++i) {
        ColorType *out_pixels = &p.pixels[p.lod_offsets[i]];

        for (int y = 0; y < p.res[i - 1][1] - 1; y += 2) {
            for (int x = 0; x < p.res[i - 1][0] - 1; x += 2) {
                const ColorType c00 = Get(index, x + 0, y + 0, i - 1);
                const ColorType c10 = Get(index, x + 1, y + 0, i - 1);
                const ColorType c11 = Get(index, x + 1, y + 1, i - 1);
                const ColorType c01 = Get(index, x + 0, y + 1, i - 1);

                ColorType res;
                for (int i = 0; i < N; ++i) {
                    res.v[i] = (c00.v[i] + c10.v[i] + c11.v[i] + c01.v[i]) / 4;
                }

                out_pixels[(y / 2) * p.res[i][0] + (x / 2)] = res;
            }
        }

        for (int y = 0; y < p.res[i][1]; ++y) {
            out_pixels[y * p.res[i][0] + p.res[i][0] - 1] = out_pixels[y * p.res[i][0]];
        }

        memcpy(&out_pixels[(p.res[i][1] - 1) * p.res[i][0]], out_pixels, p.res[i][0] * sizeof(ColorType));
        out_pixels[(p.res[i][1] - 1) * p.res[i][0] + p.res[i][0] - 1] = out_pixels[0];
    }

    return index;
}

template <typename T, int N> bool Ray::Ref::TexStorageLinear<T, N>::Free(const int index) {
    if (index < 0 || index > images_.size()) {
        return false;
    }

#ifndef NDEBUG
    memset(images_[index].res, 0, sizeof(images_[index].res));
    memset(images_[index].lod_offsets, 0, sizeof(images_[index].lod_offsets));
#endif

    images_[index].pixels.reset();
    free_slots_.push_back(index);

    return true;
}

template <typename T, int N>
void Ray::Ref::TexStorageLinear<T, N>::WriteImageData(const int index, const int lod, const ColorType data[]) {
    const ImgData &p = images_[index];
    const int w = p.res[lod][0], h = p.res[lod][1];
    memcpy(&p.pixels[p.lod_offsets[lod]], data, w * h * sizeof(ColorType));
}

template class Ray::Ref::TexStorageLinear<uint8_t, 4>;
template class Ray::Ref::TexStorageLinear<uint8_t, 3>;
template class Ray::Ref::TexStorageLinear<uint8_t, 2>;
template class Ray::Ref::TexStorageLinear<uint8_t, 1>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
int Ray::Ref::TexStorageTiled<T, N>::Allocate(const ColorType *data, const int _res[2], bool mips) {
    int index = -1;
    if (!free_slots_.empty()) {
        index = free_slots_.back();
        free_slots_.pop_back();
    } else {
        index = int(images_.size());
        images_.resize(images_.size() + 1);
    }

    ImgData &p = images_[index];

    p.lod_offsets[0] = 0;
    p.res[0][0] = _res[0] + 1;
    p.res[0][1] = _res[1] + 1;
    p.res_in_tiles[0][0] = (p.res[0][0] + TileSize - 1) / TileSize;
    p.res_in_tiles[0][1] = (p.res[0][1] + TileSize - 1) / TileSize;

    int total_size = (p.res_in_tiles[0][0] * p.res_in_tiles[0][1]) * (TileSize * TileSize);

    for (int i = 1; i < NUM_MIP_LEVELS; ++i) {
        if (mips && (p.res[i - 1][0] > 1 || p.res[i - 1][1] > 1)) {
            p.lod_offsets[i] = total_size;

            p.res[i][0] = p.res[i - 1][0] / 2 + 1;
            p.res[i][1] = p.res[i - 1][1] / 2 + 1;

            p.res_in_tiles[i][0] = (p.res[i][0] + TileSize - 1) / TileSize;
            p.res_in_tiles[i][1] = (p.res[i][1] + TileSize - 1) / TileSize;

            total_size += (p.res_in_tiles[i][0] * p.res_in_tiles[i][1]) * (TileSize * TileSize);
        } else {
            p.lod_offsets[i] = p.lod_offsets[i - 1];
            p.res[i][0] = p.res[i - 1][0];
            p.res[i][1] = p.res[i - 1][1];
            p.res_in_tiles[i][0] = p.res_in_tiles[i - 1][0];
            p.res_in_tiles[i][1] = p.res_in_tiles[i - 1][1];
        }
    }

    p.pixels.reset(new ColorType[total_size]);

    for (int y = 0; y < _res[1]; ++y) {
        const int tiley = y / TileSize, in_tiley = y % TileSize;

        for (int x = 0; x < _res[0]; ++x) {
            const int tilex = x / TileSize, in_tilex = x % TileSize;

            p.pixels[(tiley * p.res_in_tiles[0][0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
                data[y * _res[0] + x];
        }

        { // write additional row to the right
            const int tilex = _res[0] / TileSize, in_tilex = _res[0] % TileSize;
            p.pixels[(tiley * p.res_in_tiles[0][0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
                data[y * _res[0] + _res[0]];
        }
    }

    { // write additional line at the bottom
        const int tiley = _res[1] / TileSize, in_tiley = _res[1] % TileSize;

        for (int x = 0; x < _res[0]; ++x) {
            const int tilex = x / TileSize, in_tilex = x % TileSize;

            p.pixels[(tiley * p.res_in_tiles[0][0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
                data[x];
        }
    }

    { // write additional corner pixel
        const int tiley = _res[1] / TileSize, in_tiley = _res[1] % TileSize;
        const int tilex = _res[0] / TileSize, in_tilex = _res[0] % TileSize;

        p.pixels[(tiley * p.res_in_tiles[0][0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
            data[0];
    }

    for (int i = 1; i < NUM_MIP_LEVELS && mips; ++i) {
        ColorType *out_pixels = &p.pixels[p.lod_offsets[i]];

        for (int y = 0; y < p.res[i - 1][1] - 1; y += 2) {
            const int tiley = (y / 2) / TileSize, in_tiley = (y / 2) % TileSize;

            for (int x = 0; x < p.res[i - 1][0] - 1; x += 2) {
                const ColorType c00 = Get(index, x + 0, y + 0, i - 1);
                const ColorType c10 = Get(index, x + 1, y + 0, i - 1);
                const ColorType c11 = Get(index, x + 1, y + 1, i - 1);
                const ColorType c01 = Get(index, x + 0, y + 1, i - 1);

                ColorType res;
                for (int i = 0; i < N; ++i) {
                    res.v[i] = (c00.v[i] + c10.v[i] + c11.v[i] + c01.v[i]) / 4;
                }

                const int tilex = (x / 2) / TileSize, in_tilex = (x / 2) % TileSize;

                out_pixels[(tiley * p.res_in_tiles[i][0] + tilex) * TileSize * TileSize + in_tiley * TileSize +
                           in_tilex] = res;
            }
        }

        // write additional row to the right
        for (int y = 0; y < p.res[i][1]; ++y) {
            const int tiley = y / TileSize, in_tiley = y % TileSize;
            const int tilex = (p.res[i][0] - 1) / TileSize, in_tilex = (p.res[i][0] - 1) % TileSize;
            out_pixels[(tiley * p.res_in_tiles[i][0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
                Get(index, 0, y, i);
        }

        { // write additional line at the bottom
            const int tiley = (p.res[i][1] - 1) / TileSize, in_tiley = (p.res[i][1] - 1) % TileSize;

            for (int x = 0; x < p.res[i][0]; ++x) {
                const int tilex = x / TileSize, in_tilex = x % TileSize;

                out_pixels[(tiley * p.res_in_tiles[i][0] + tilex) * TileSize * TileSize + in_tiley * TileSize +
                           in_tilex] = Get(index, x, 0, i);
            }
        }

        { // write additional corner pixel
            const int tiley = (p.res[i][1] - 1) / TileSize, in_tiley = (p.res[i][1] - 1) % TileSize;
            const int tilex = (p.res[i][0] - 1) / TileSize, in_tilex = (p.res[i][0] - 1) % TileSize;

            out_pixels[(tiley * p.res_in_tiles[i][0] + tilex) * TileSize * TileSize + in_tiley * TileSize + in_tilex] =
                Get(index, 0, 0, i);
        }
    }

    return index;
}

template <typename T, int N> bool Ray::Ref::TexStorageTiled<T, N>::Free(const int index) {
    if (index < 0 || index > images_.size()) {
        return false;
    }

#ifndef NDEBUG
    memset(images_[index].res, 0, sizeof(images_[index].res));
    memset(images_[index].res_in_tiles, 0, sizeof(images_[index].res_in_tiles));
    memset(images_[index].lod_offsets, 0, sizeof(images_[index].lod_offsets));
#endif

    images_[index].pixels.reset();
    free_slots_.push_back(index);

    return true;
}

template class Ray::Ref::TexStorageTiled<uint8_t, 4>;
template class Ray::Ref::TexStorageTiled<uint8_t, 3>;
template class Ray::Ref::TexStorageTiled<uint8_t, 2>;
template class Ray::Ref::TexStorageTiled<uint8_t, 1>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
int Ray::Ref::TexStorageSwizzled<T, N>::Allocate(const ColorType *data, const int _res[2], bool mips) {
    int index = -1;
    if (!free_slots_.empty()) {
        index = free_slots_.back();
        free_slots_.pop_back();
    } else {
        index = int(images_.size());
        images_.resize(images_.size() + 1);
    }

    ImgData &p = images_[index];

    p.lod_offsets[0] = 0;
    p.res[0][0] = _res[0] + 1;
    p.res[0][1] = _res[1] + 1;
    p.tile_y_stride[0] = swizzle_x_tile(OuterTileW * ((p.res[0][0] + OuterTileW - 1) / OuterTileW));

    int total_size = p.tile_y_stride[0] * ((p.res[0][1] + OuterTileH - 1) / OuterTileH);

    mips = false;

    for (int i = 1; i < NUM_MIP_LEVELS; ++i) {
        if (mips && (p.res[i - 1][0] > 1 || p.res[i - 1][1] > 1)) {
            p.lod_offsets[i] = total_size;

            p.res[i][0] = (p.res[i - 1][0] / 2) + 1;
            p.res[i][1] = (p.res[i - 1][1] / 2) + 1;

            p.tile_y_stride[i] = swizzle_x_tile(OuterTileW * ((p.res[i][0] + OuterTileW - 1) / OuterTileW));

            total_size += p.tile_y_stride[i] * ((p.res[i][1] + OuterTileH - 1) / OuterTileH);
        } else {
            p.lod_offsets[i] = p.lod_offsets[i - 1];
            p.res[i][0] = p.res[i - 1][0];
            p.res[i][1] = p.res[i - 1][1];
            p.tile_y_stride[i] = p.tile_y_stride[i - 1];
        }
    }

    p.pixels.reset(new ColorType[total_size]);

    for (int y = 0; y < _res[1]; ++y) {
        const uint32_t y_off = (y / OuterTileH) * p.tile_y_stride[0] + swizzle_y(y);

        for (int x = 0; x < _res[0]; ++x) {
            const uint32_t x_off = swizzle_x_tile(x);

            p.pixels[y_off + x_off] = data[y * _res[0] + x];
        }

        { // write additional row to the right
            const uint32_t x_off = swizzle_x_tile(_res[0]);
            p.pixels[y_off + x_off] = data[y * _res[0] + _res[0]];
        }
    }

    { // write additional line at the bottom
        const uint32_t y_off = (_res[1] / OuterTileH) * p.tile_y_stride[0] + swizzle_y(_res[1]);

        for (int x = 0; x < _res[0]; ++x) {
            const uint32_t x_off = swizzle_x_tile(x);

            p.pixels[y_off + x_off] = data[x];
        }
    }

    { // write additional corner pixel
        const uint32_t y_off = (_res[1] / OuterTileH) * p.tile_y_stride[0] + swizzle_y(_res[1]);
        const uint32_t x_off = swizzle_x_tile(_res[0]);

        p.pixels[y_off + x_off] = data[0];
    }

    for (int i = 1; i < NUM_MIP_LEVELS && mips; ++i) {
        ColorType *out_pixels = &p.pixels[p.lod_offsets[i]];

        for (int y = 0; y < p.res[i - 1][1] - 1; y += 2) {
            const uint32_t y_off = ((y / 2) / OuterTileH) * p.tile_y_stride[i] + swizzle_y(y / 2);

            for (int x = 0; x < p.res[i - 1][0] - 1; x += 2) {
                const ColorType c00 = Get(index, x + 0, y + 0, i - 1);
                const ColorType c10 = Get(index, x + 1, y + 0, i - 1);
                const ColorType c11 = Get(index, x + 1, y + 1, i - 1);
                const ColorType c01 = Get(index, x + 0, y + 1, i - 1);

                ColorType res;
                for (int i = 0; i < N; ++i) {
                    res.v[i] = (c00.v[i] + c10.v[i] + c11.v[i] + c01.v[i]) / 4;
                }

                const uint32_t x_off = swizzle_x_tile(x / 2);

                out_pixels[y_off + x_off] = res;
            }
        }

        // write additional row to the right
        for (int y = 0; y < p.res[i][1]; ++y) {
            const uint32_t y_off = (y / OuterTileH) * p.tile_y_stride[i] + swizzle_y(y);
            const uint32_t x_off = swizzle_x_tile(p.res[i][0] - 1);

            out_pixels[y_off + x_off] = Get(index, 0, y, i);
        }

        { // write additional line at the bottom
            const uint32_t y_off = ((p.res[i][1] - 1) / OuterTileH) * p.tile_y_stride[i] + swizzle_y(p.res[i][1] - 1);

            for (int x = 0; x < p.res[i][0]; ++x) {
                const uint32_t x_off = swizzle_x_tile(x);

                out_pixels[y_off + x_off] = Get(index, x, 0, i);
            }
        }

        { // write additional corner pixel
            const uint32_t y_off = ((p.res[i][1] - 1) / OuterTileH) * p.tile_y_stride[i] + swizzle_y((p.res[i][1] - 1));
            const uint32_t x_off = swizzle_x_tile(p.res[i][0] - 1);

            out_pixels[y_off + x_off] = Get(index, 0, 0, i);
        }
    }

    return index;
}

template <typename T, int N> bool Ray::Ref::TexStorageSwizzled<T, N>::Free(const int index) {
    if (index < 0 || index > images_.size()) {
        return false;
    }

#ifndef NDEBUG
    memset(images_[index].res, 0, sizeof(images_[index].res));
    memset(images_[index].tile_y_stride, 0, sizeof(images_[index].tile_y_stride));
    memset(images_[index].lod_offsets, 0, sizeof(images_[index].lod_offsets));
#endif

    images_[index].pixels.reset();
    free_slots_.push_back(index);

    return true;
}

template class Ray::Ref::TexStorageSwizzled<uint8_t, 4>;
template class Ray::Ref::TexStorageSwizzled<uint8_t, 3>;
template class Ray::Ref::TexStorageSwizzled<uint8_t, 2>;
template class Ray::Ref::TexStorageSwizzled<uint8_t, 1>;
