#include "utils.h"

#include <cassert>
#include <cstring>

#include <fstream>
#include <memory>
#include <string>

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

std::vector<uint8_t> LoadHDR(const char name[], int &out_w, int &out_h) {
    std::ifstream in_file(name, std::ios::binary);

    std::string line;
    if (!std::getline(in_file, line) || line != "#?RADIANCE") {
        throw std::runtime_error("Is not HDR file!");
    }

    float exposure = 1.0f;
    std::string format;

    while (std::getline(in_file, line)) {
        if (line.empty())
            break;

        if (!line.compare(0, 6, "FORMAT")) {
            format = line.substr(7);
        } else if (!line.compare(0, 8, "EXPOSURE")) {
            exposure = (float)atof(line.substr(9).c_str());
        }
    }

    if (format != "32-bit_rle_rgbe") {
        throw std::runtime_error("Wrong format!");
    }

    int res_x = 0, res_y = 0;

    std::string resolution;
    if (!std::getline(in_file, resolution)) {
        throw std::runtime_error("Cannot read resolution!");
    }

    { // parse resolution
        const char *delims = " \r\n";
        const char *p = resolution.c_str();
        const char *q = strpbrk(p + 1, delims);

        if ((q - p) != 2 || p[0] != '-' || p[1] != 'Y') {
            throw std::runtime_error("Unsupported format!");
        }

        p = q + 1;
        q = strpbrk(p, delims);
        res_y = int(strtol(p, nullptr, 10));

        p = q + 1;
        q = strpbrk(p, delims);
        if ((q - p) != 2 || p[0] != '+' || p[1] != 'X') {
            throw std::runtime_error("Unsupported format!");
        }

        p = q + 1;
        q = strpbrk(p, delims);
        res_x = int(strtol(p, nullptr, 10));
    }

    if (!res_x || !res_y) {
        throw std::runtime_error("Unsupported format!");
    }

    out_w = res_x;
    out_h = res_y;

    std::vector<uint8_t> data(res_x * res_y * 4);
    int data_offset = 0;

    int scanlines_left = res_y;
    std::vector<uint8_t> scanline(res_x * 4);

    while (scanlines_left) {
        {
            uint8_t rgbe[4];

            if (!in_file.read((char *)&rgbe[0], 4)) {
                throw std::runtime_error("Cannot read file!");
            }

            if ((rgbe[0] != 2) || (rgbe[1] != 2) || ((rgbe[2] & 0x80) != 0)) {
                data[0] = rgbe[0];
                data[1] = rgbe[1];
                data[2] = rgbe[2];
                data[3] = rgbe[3];

                if (!in_file.read((char *)&data[4], (res_x * scanlines_left - 1) * 4)) {
                    throw std::runtime_error("Cannot read file!");
                }
                return data;
            }

            if ((((rgbe[2] & 0xFF) << 8) | (rgbe[3] & 0xFF)) != res_x) {
                throw std::runtime_error("Wrong scanline width!");
            }
        }

        int index = 0;
        for (int i = 0; i < 4; i++) {
            int index_end = (i + 1) * res_x;
            while (index < index_end) {
                uint8_t buf[2];
                if (!in_file.read((char *)&buf[0], 2)) {
                    throw std::runtime_error("Cannot read file!");
                }

                if (buf[0] > 128) {
                    int count = buf[0] - 128;
                    if ((count == 0) || (count > index_end - index)) {
                        throw std::runtime_error("Wrong data!");
                    }
                    while (count-- > 0) {
                        scanline[index++] = buf[1];
                    }
                } else {
                    int count = buf[0];
                    if ((count == 0) || (count > index_end - index)) {
                        throw std::runtime_error("Wrong data!");
                    }
                    scanline[index++] = buf[1];
                    if (--count > 0) {
                        if (!in_file.read((char *)&scanline[index], count)) {
                            throw std::runtime_error("Cannot read file!");
                        }
                        index += count;
                    }
                }
            }
        }

        for (int i = 0; i < res_x; i++) {
            data[data_offset + 0] = scanline[i + 0 * res_x];
            data[data_offset + 1] = scanline[i + 1 * res_x];
            data[data_offset + 2] = scanline[i + 2 * res_x];
            data[data_offset + 3] = scanline[i + 3 * res_x];
            data_offset += 4;
        }

        scanlines_left--;
    }

    return data;
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