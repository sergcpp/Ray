#include "utils.h"

#include <cassert>
#include <cstring>

#include <fstream>
#include <memory>
#include <string>

#include "../internal/Utils.h"

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

std::vector<uint8_t> LoadTGA(const char file_name[], bool flip_y, int &w, int &h) {
    std::ifstream in_file(file_name, std::ios::binary);
    if (!in_file) {
        return {};
    }

    in_file.seekg(0, std::ios::end);
    size_t in_file_size = (size_t)in_file.tellg();
    in_file.seekg(0, std::ios::beg);

    std::vector<char> in_file_data(in_file_size);
    in_file.read(&in_file_data[0], in_file_size);

    Ray::eTexFormat format;
    uint32_t img_size;
    Ray::ReadTGAFile(&in_file_data[0], int(in_file_size), w, h, format, nullptr, img_size);
    if (!img_size) {
        return {};
    }

    std::vector<uint8_t> tex_data(img_size);
    Ray::ReadTGAFile(&in_file_data[0], int(in_file_size), w, h, format, tex_data.data(), img_size);

    if (format == Ray::eTexFormat::RawRGB888) {
        std::vector<uint8_t> tex_data2(w * h * 4);
        for (int i = 0; i < w * h; ++i) {
            tex_data2[4 * i + 0] = tex_data[3 * i + 0];
            tex_data2[4 * i + 1] = tex_data[3 * i + 1];
            tex_data2[4 * i + 2] = tex_data[3 * i + 2];
            tex_data2[4 * i + 3] = 255;
        }
        tex_data = tex_data2;
    }

    for (int y = 0; y < (h / 2) && flip_y; y++) {
        for (int i = 0; i < w * 4; ++i) {
            std::swap(tex_data[(h - y - 1) * w * 4 + i], tex_data[y * w * 4 + i]);
        }
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

    (void)exposure;

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
                    if (count > index_end - index) {
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
