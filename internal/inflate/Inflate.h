#pragma once

#include "../../Span.h"

namespace Ray {
struct huff_table_t;
class Inflater {
    Span<const uint8_t> input_;
    const uint8_t *cursor_ = nullptr;

    int temp_bits_available_ = 0;
    uint64_t temp_bits_ = 0;

    uint32_t GetBits(int n);
    uint32_t Peek16Bits();
    void ConsumeBits(const int n);
    void Refill32Bits();
    bool SkipToByteBoundary();

    uint32_t CopyUncompressed(Span<uint8_t> output);
    uint32_t DecodeHuffman(bool dynamic, Span<uint8_t> output);

    int ReadRawLengths(const int len_bits, const int read_symbols, const int symbols, uint8_t *code_length);

    uint32_t ReadValue(const uint32_t *rev_symbol_table, huff_table_t &out_table);
    int ReadLength(const uint32_t *tables_rev_symbol_table, const int read_symbols, const int symbols,
                   uint8_t *code_length, huff_table_t &out_table);

  public:
    void Feed(Span<const uint8_t> input) {
        input_ = input;
        cursor_ = input_.data();
    }

    int Inflate(Span<uint8_t> output);
};

inline std::vector<uint8_t> Inflate(Span<const uint8_t> data, const int max_size = 256 * 1024) {
    Inflater inflater;
    inflater.Feed(data);

    std::vector<uint8_t> ret(max_size);
    const int decompressed_size = inflater.Inflate(ret);
    if (decompressed_size == -1) {
        ret.assign(data.begin(), data.end());
        return ret;
    }
    ret.resize(decompressed_size);
    return ret;
}

} // namespace Ray