#include "utils.h"

#include <cassert>
#include <cstring>

#include <fstream>
#include <memory>

std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>> LoadBIN(const char file_name[]) {
    std::ifstream in_file(file_name, std::ios::binary);
    uint32_t num_attrs;
    in_file.read((char *)&num_attrs, 4);
    uint32_t num_indices;
    in_file.read((char *)&num_indices, 4);
    uint32_t num_groups;
    in_file.read((char *)&num_groups, 4);

    std::vector<float> attrs;
    attrs.resize(num_attrs);
    in_file.read((char *)&attrs[0], size_t(num_attrs) * sizeof(float));

    std::vector<uint32_t> indices;
    indices.resize(num_indices);
    in_file.read((char *)&indices[0], size_t(num_indices) * sizeof(uint32_t));

    std::vector<uint32_t> groups;
    groups.resize(num_groups);
    in_file.read((char *)&groups[0], size_t(num_groups) * sizeof(uint32_t));

    return std::make_tuple(std::move(attrs), std::move(indices), std::move(groups));
}

std::unique_ptr<uint8_t[]> ReadTGAFile(const void *data, int &w, int &h, int &bpp) {
    uint8_t tga_header[12] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t *tga_compare = (const uint8_t *)data;
    const uint8_t *img_header = (const uint8_t *)data + sizeof(tga_header);
    uint32_t img_size;
    bool compressed = false;

    if (memcmp(tga_header, tga_compare, sizeof(tga_header)) != 0) {
        if (tga_compare[2] == 1) {
            fprintf(stderr, "Image cannot be indexed color.");
        }
        if (tga_compare[2] == 3) {
            fprintf(stderr, "Image cannot be greyscale color.");
        }
        if (tga_compare[2] == 9 || tga_compare[2] == 10) {
            // fprintf(stderr, "Image cannot be compressed.");
            compressed = true;
        }
    }

    w = img_header[1] * 256u + img_header[0];
    h = img_header[3] * 256u + img_header[2];

    if (w <= 0 || h <= 0 || (img_header[4] != 24 && img_header[4] != 32)) {
        if (w <= 0 || h <= 0) {
            fprintf(stderr, "Image must have a width and height greater than 0");
        }
        if (img_header[4] != 24 && img_header[4] != 32) {
            fprintf(stderr, "Image must be 24 or 32 bit");
        }
        return nullptr;
    }

    bpp = img_header[4];
    uint32_t bytes_per_pixel = bpp / 8;
    img_size = w * h * bytes_per_pixel;
    const uint8_t *image_data = (const uint8_t *)data + 18;

    std::unique_ptr<uint8_t[]> image_ret(new uint8_t[img_size]);
    uint8_t *_image_ret = &image_ret[0];

    if (!compressed) {
        for (unsigned i = 0; i < img_size; i += bytes_per_pixel) {
            _image_ret[i] = image_data[i + 2];
            _image_ret[i + 1] = image_data[i + 1];
            _image_ret[i + 2] = image_data[i];
            if (bytes_per_pixel == 4) {
                _image_ret[i + 3] = image_data[i + 3];
            }
        }
    } else {
        for (unsigned num = 0; num < img_size;) {
            uint8_t packet_header = *image_data++;
            if (packet_header & (1 << 7)) {
                uint8_t color[4];
                unsigned size = (packet_header & ~(1 << 7)) + 1;
                size *= bytes_per_pixel;
                for (unsigned i = 0; i < bytes_per_pixel; i++) {
                    color[i] = *image_data++;
                }
                for (unsigned i = 0; i < size; i += bytes_per_pixel, num += bytes_per_pixel) {
                    _image_ret[num] = color[2];
                    _image_ret[num + 1] = color[1];
                    _image_ret[num + 2] = color[0];
                    if (bytes_per_pixel == 4) {
                        _image_ret[num + 3] = color[3];
                    }
                }
            } else {
                unsigned size = (packet_header & ~(1 << 7)) + 1;
                size *= bytes_per_pixel;
                for (unsigned i = 0; i < size; i += bytes_per_pixel, num += bytes_per_pixel) {
                    _image_ret[num] = image_data[i + 2];
                    _image_ret[num + 1] = image_data[i + 1];
                    _image_ret[num + 2] = image_data[i];
                    if (bytes_per_pixel == 4) {
                        _image_ret[num + 3] = image_data[i + 3];
                    }
                }
                image_data += size;
            }
        }
    }

    return image_ret;
}

std::vector<uint8_t> LoadTGA(const char file_name[], int &w, int &h) {
    std::vector<uint8_t> tex_data;

    std::ifstream in_file(file_name, std::ios::binary);
    if (!in_file) {
        return {};
    }

    in_file.seekg(0, std::ios::end);
    size_t in_file_size = (size_t)in_file.tellg();
    in_file.seekg(0, std::ios::beg);

    std::vector<char> in_file_data(in_file_size);
    in_file.read(&in_file_data[0], in_file_size);

    int bpp;
    auto pixels = ReadTGAFile(&in_file_data[0], w, h, bpp);

    if (bpp == 24) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                tex_data.push_back(pixels[3 * (y * w + x)]);
                tex_data.push_back(pixels[3 * (y * w + x) + 1]);
                tex_data.push_back(pixels[3 * (y * w + x) + 2]);
                tex_data.push_back(255);
            }
        }
    } else if (bpp == 32) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                tex_data.push_back(pixels[4 * (y * w + x)]);
                tex_data.push_back(pixels[4 * (y * w + x) + 1]);
                tex_data.push_back(pixels[4 * (y * w + x) + 2]);
                tex_data.push_back(pixels[4 * (y * w + x) + 3]);
            }
        }
    } else {
        assert(false);
    }

    return tex_data;
}

void WriteTGA(const uint8_t *data, int w, int h, int bpp, const char *name) {
    std::ofstream file(name, std::ios::binary);

    unsigned char header[18] = {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    header[12] = w & 0xFF;
    header[13] = (w >> 8) & 0xFF;
    header[14] = (h)&0xFF;
    header[15] = (h >> 8) & 0xFF;
    header[16] = bpp * 8;

    file.write((char *)&header[0], sizeof(unsigned char) * 18);

    auto out_data = std::unique_ptr<uint8_t[]>{new uint8_t[size_t(w) * h * bpp]};
    if (bpp == 3) {
        for (size_t i = 0; i < size_t(w) * h; ++i) {
            out_data[i * 3 + 0] = data[i * 3 + 2];
            out_data[i * 3 + 1] = data[i * 3 + 1];
            out_data[i * 3 + 2] = data[i * 3 + 0];
        }
    } else {
        for (size_t i = 0; i < size_t(w) * h; ++i) {
            out_data[i * 4 + 0] = data[i * 4 + 2];
            out_data[i * 4 + 1] = data[i * 4 + 1];
            out_data[i * 4 + 2] = data[i * 4 + 0];
            out_data[i * 4 + 3] = data[i * 4 + 3];
        }
    }

    file.write((const char *)&out_data[0], size_t(w) * h * bpp);

    static const char footer[26] = "\0\0\0\0"         // no extension area
                                   "\0\0\0\0"         // no developer directory
                                   "TRUEVISION-XFILE" // yep, this is a TGA file
                                   ".";
    file.write((const char *)&footer, sizeof(footer));
}

