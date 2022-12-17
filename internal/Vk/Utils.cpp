#include "Utils.h"

#include <cstring>

#include "TextureParams.h"

std::unique_ptr<uint8_t[]> Ray::Vk::ReadTGAFile(const void *data, int &w, int &h, eTexFormat &format) {
    uint32_t img_size;
    ReadTGAFile(data, w, h, format, nullptr, img_size);

    std::unique_ptr<uint8_t[]> image_ret(new uint8_t[img_size]);
    ReadTGAFile(data, w, h, format, image_ret.get(), img_size);

    return image_ret;
}

bool Ray::Vk::ReadTGAFile(const void *data, int &w, int &h, eTexFormat &format, uint8_t *out_data, uint32_t &out_size) {
    const uint8_t tga_header[12] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto *tga_compare = (const uint8_t *)data;
    const uint8_t *img_header = (const uint8_t *)data + sizeof(tga_header);
    bool compressed = false;

    if (memcmp(tga_header, tga_compare, sizeof(tga_header)) != 0) {
        if (tga_compare[2] == 1) {
            //fprintf(stderr, "Image cannot be indexed color.");
            return false;
        }
        if (tga_compare[2] == 3) {
            //fprintf(stderr, "Image cannot be greyscale color.");
            return false;
        }
        if (tga_compare[2] == 9 || tga_compare[2] == 10) {
            compressed = true;
        }
    }

    w = int(img_header[1] * 256u + img_header[0]);
    h = int(img_header[3] * 256u + img_header[2]);

    if (w <= 0 || h <= 0 || (img_header[4] != 24 && img_header[4] != 32)) {
        if (w <= 0 || h <= 0) {
            //fprintf(stderr, "Image must have a width and height greater than 0");
        }
        if (img_header[4] != 24 && img_header[4] != 32) {
            //fprintf(stderr, "Image must be 24 or 32 bit");
        }
        return false;
    }

    const uint32_t bpp = img_header[4];
    const uint32_t bytes_per_pixel = bpp / 8;
    if (bpp == 32) {
        format = eTexFormat::RawRGBA8888;
    } else if (bpp == 24) {
        format = eTexFormat::RawRGB888;
    }

    if (out_data && out_size < w * h * bytes_per_pixel) {
        return false;
    }

    out_size = w * h * bytes_per_pixel;
    if (out_data) {
        const uint8_t *image_data = (const uint8_t *)data + 18;

        if (!compressed) {
            for (size_t i = 0; i < out_size; i += bytes_per_pixel) {
                out_data[i] = image_data[i + 2];
                out_data[i + 1] = image_data[i + 1];
                out_data[i + 2] = image_data[i];
                if (bytes_per_pixel == 4) {
                    out_data[i + 3] = image_data[i + 3];
                }
            }
        } else {
            for (size_t num = 0; num < out_size;) {
                uint8_t packet_header = *image_data++;
                if (packet_header & (1u << 7u)) {
                    uint8_t color[4];
                    unsigned size = (packet_header & ~(1u << 7u)) + 1;
                    size *= bytes_per_pixel;
                    for (unsigned i = 0; i < bytes_per_pixel; i++) {
                        color[i] = *image_data++;
                    }
                    for (unsigned i = 0; i < size; i += bytes_per_pixel, num += bytes_per_pixel) {
                        out_data[num] = color[2];
                        out_data[num + 1] = color[1];
                        out_data[num + 2] = color[0];
                        if (bytes_per_pixel == 4) {
                            out_data[num + 3] = color[3];
                        }
                    }
                } else {
                    unsigned size = (packet_header & ~(1u << 7u)) + 1;
                    size *= bytes_per_pixel;
                    for (unsigned i = 0; i < size; i += bytes_per_pixel, num += bytes_per_pixel) {
                        out_data[num] = image_data[i + 2];
                        out_data[num + 1] = image_data[i + 1];
                        out_data[num + 2] = image_data[i];
                        if (bytes_per_pixel == 4) {
                            out_data[num + 3] = image_data[i + 3];
                        }
                    }
                    image_data += size;
                }
            }
        }
    }

    return true;
}
