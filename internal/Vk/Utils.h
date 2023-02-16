#pragma once

#include <memory>
#include <vector>

namespace Ray {
namespace Vk {
enum class eTexFormat : uint8_t;
std::unique_ptr<uint8_t[]> ReadTGAFile(const void *data, int &w, int &h, eTexFormat &format);
bool ReadTGAFile(const void *data, int &w, int &h, eTexFormat &format, uint8_t *out_data, uint32_t &out_size);
} // namespace Vk
} // namespace Ray
