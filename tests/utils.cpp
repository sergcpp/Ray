#include "utils.h"

#include <cassert>
#include <cstring>

#include <fstream>
#include <memory>
#include <string>

#include "../internal/TextureUtils.h"

namespace {
//	the following constants were copied directly off the MSDN website

//	The dwFlags member of the original DDSURFACEDESC2 structure
//        can be set to one or more of the following values.
#define DDSD_CAPS 0x00000001
#define DDSD_HEIGHT 0x00000002
#define DDSD_WIDTH 0x00000004
#define DDSD_PITCH 0x00000008
#define DDSD_PIXELFORMAT 0x00001000
#define DDSD_MIPMAPCOUNT 0x00020000
#define DDSD_LINEARSIZE 0x00080000
#define DDSD_DEPTH 0x00800000

//	DirectDraw Pixel Format
#define DDPF_ALPHAPIXELS 0x00000001
#define DDPF_FOURCC 0x00000004
#define DDPF_RGB 0x00000040

//	The dwCaps1 member of the DDSCAPS2 structure can be
//        set to one or more of the following values.
#define DDSCAPS_COMPLEX 0x00000008
#define DDSCAPS_TEXTURE 0x00001000
#define DDSCAPS_MIPMAP 0x00400000

struct DDSHeader {
    uint32_t dwMagic;
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth;
    uint32_t dwMipMapCount;
    uint32_t dwReserved1[11];

    //  DDPIXELFORMAT
    struct {
        uint32_t dwSize;
        uint32_t dwFlags;
        uint32_t dwFourCC;
        uint32_t dwRGBBitCount;
        uint32_t dwRBitMask;
        uint32_t dwGBitMask;
        uint32_t dwBBitMask;
        uint32_t dwAlphaBitMask;
    } sPixelFormat;

    //  DDCAPS2
    struct {
        uint32_t dwCaps1;
        uint32_t dwCaps2;
        uint32_t dwDDSX;
        uint32_t dwReserved;
    } sCaps;
    uint32_t dwReserved2;
};
static_assert(sizeof(DDSHeader) == 128, "!");
} // namespace

std::tuple<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>> LoadBIN(const char file_name[]) {
    std::ifstream in_file(file_name, std::ios::binary);
    if (!in_file) {
        return {};
    }
    uint32_t num_attrs;
    in_file.read((char *)&num_attrs, 4);
    if (!in_file) {
        return {};
    }
    uint32_t num_indices;
    in_file.read((char *)&num_indices, 4);
    if (!in_file) {
        return {};
    }
    uint32_t num_groups;
    in_file.read((char *)&num_groups, 4);
    if (!in_file) {
        return {};
    }

    std::vector<float> attrs;
    attrs.resize(num_attrs);
    in_file.read((char *)&attrs[0], size_t(num_attrs) * sizeof(float));
    if (!in_file) {
        return {};
    }

    std::vector<uint32_t> indices;
    indices.resize(num_indices);
    in_file.read((char *)&indices[0], size_t(num_indices) * sizeof(uint32_t));
    if (!in_file) {
        return {};
    }
    std::vector<uint32_t> groups;
    groups.resize(num_groups);
    in_file.read((char *)&groups[0], size_t(num_groups) * sizeof(uint32_t));
    if (!in_file) {
        return {};
    }

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
    if (!in_file) {
        return {};
    }

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

std::vector<uint8_t> LoadDDS(const char file_name[], int &w, int &h, int &mips, int &channels) {
    std::ifstream in_file(file_name, std::ios::binary);
    if (!in_file) {
        return {};
    }

    in_file.seekg(0, std::ios::end);
    size_t in_file_size = (size_t)in_file.tellg();
    in_file.seekg(0, std::ios::beg);

    DDSHeader dds_header = {};
    in_file.read((char *)&dds_header, sizeof(DDSHeader));
    if (!in_file) {
        return {};
    }

    w = dds_header.dwWidth;
    h = dds_header.dwHeight;
    mips = dds_header.dwMipMapCount;

    if ((dds_header.dwFlags & DDPF_FOURCC) != 0) {
        if (dds_header.sPixelFormat.dwFourCC == (('D' << 0) | ('X' << 8) | ('T' << 16) | ('1' << 24))) {
            channels = 3;
        } else if (dds_header.sPixelFormat.dwFourCC == (('D' << 0) | ('X' << 8) | ('T' << 16) | ('5' << 24))) {
            channels = 4;
        } else if (dds_header.sPixelFormat.dwFourCC == (('B' << 0) | ('C' << 8) | ('4' << 16) | ('U' << 24)) ||
                   dds_header.sPixelFormat.dwFourCC == (('A' << 0) | ('T' << 8) | ('I' << 16) | ('1' << 24))) {
            channels = 1;
        } else if (dds_header.sPixelFormat.dwFourCC == (('A' << 0) | ('T' << 8) | ('I' << 16) | ('2' << 24))) {
            channels = 2;
        } else if (dds_header.sPixelFormat.dwFourCC == (('D' << 0u) | ('X' << 8u) | ('1' << 16u) | ('0' << 24u))) {
            assert(false);
        }
    }

    std::vector<uint8_t> ret(in_file_size - sizeof(DDSHeader));
    in_file.read((char *)&ret[0], in_file_size - sizeof(DDSHeader));
    if (!in_file) {
        return {};
    }

    return ret;
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

    unused(exposure);

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
