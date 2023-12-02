#include "Inflate.h"

#include <cstring>

#include "Huffman.h"

// Taken from https://github.com/Artexety/inflatecpp

namespace Ray {
const int HuffMatchLenSyms = 29;
const int HuffOffsetSyms = 32;
const int HuffMinMatchSize = 3;

#define MATCHLEN_PAIR(__base, __dispbits) ((__base) | ((__dispbits) << 16) | 0x8000)
#define OFFSET_PAIR(__base, __dispbits) ((__base) | ((__dispbits) << 16))

const uint32_t HuffMatchLenCode[HuffMatchLenSyms] = {
    MATCHLEN_PAIR(HuffMinMatchSize + 0, 0),   MATCHLEN_PAIR(HuffMinMatchSize + 1, 0),
    MATCHLEN_PAIR(HuffMinMatchSize + 2, 0),   MATCHLEN_PAIR(HuffMinMatchSize + 3, 0),
    MATCHLEN_PAIR(HuffMinMatchSize + 4, 0),   MATCHLEN_PAIR(HuffMinMatchSize + 5, 0),
    MATCHLEN_PAIR(HuffMinMatchSize + 6, 0),   MATCHLEN_PAIR(HuffMinMatchSize + 7, 0),
    MATCHLEN_PAIR(HuffMinMatchSize + 8, 1),   MATCHLEN_PAIR(HuffMinMatchSize + 10, 1),
    MATCHLEN_PAIR(HuffMinMatchSize + 12, 1),  MATCHLEN_PAIR(HuffMinMatchSize + 14, 1),
    MATCHLEN_PAIR(HuffMinMatchSize + 16, 2),  MATCHLEN_PAIR(HuffMinMatchSize + 20, 2),
    MATCHLEN_PAIR(HuffMinMatchSize + 24, 2),  MATCHLEN_PAIR(HuffMinMatchSize + 28, 2),
    MATCHLEN_PAIR(HuffMinMatchSize + 32, 3),  MATCHLEN_PAIR(HuffMinMatchSize + 40, 3),
    MATCHLEN_PAIR(HuffMinMatchSize + 48, 3),  MATCHLEN_PAIR(HuffMinMatchSize + 56, 3),
    MATCHLEN_PAIR(HuffMinMatchSize + 64, 4),  MATCHLEN_PAIR(HuffMinMatchSize + 80, 4),
    MATCHLEN_PAIR(HuffMinMatchSize + 96, 4),  MATCHLEN_PAIR(HuffMinMatchSize + 112, 4),
    MATCHLEN_PAIR(HuffMinMatchSize + 128, 5), MATCHLEN_PAIR(HuffMinMatchSize + 160, 5),
    MATCHLEN_PAIR(HuffMinMatchSize + 192, 5), MATCHLEN_PAIR(HuffMinMatchSize + 224, 5),
    MATCHLEN_PAIR(HuffMinMatchSize + 255, 0),
};

const uint32_t HuffOffsetCode[HuffOffsetSyms] = {
    OFFSET_PAIR(1, 0),      OFFSET_PAIR(2, 0),      OFFSET_PAIR(3, 0),     OFFSET_PAIR(4, 0),
    OFFSET_PAIR(5, 1),      OFFSET_PAIR(7, 1),      OFFSET_PAIR(9, 2),     OFFSET_PAIR(13, 2),
    OFFSET_PAIR(17, 3),     OFFSET_PAIR(25, 3),     OFFSET_PAIR(33, 4),    OFFSET_PAIR(49, 4),
    OFFSET_PAIR(65, 5),     OFFSET_PAIR(97, 5),     OFFSET_PAIR(129, 6),   OFFSET_PAIR(193, 6),
    OFFSET_PAIR(257, 7),    OFFSET_PAIR(385, 7),    OFFSET_PAIR(513, 8),   OFFSET_PAIR(769, 8),
    OFFSET_PAIR(1025, 9),   OFFSET_PAIR(1537, 9),   OFFSET_PAIR(2049, 10), OFFSET_PAIR(3073, 10),
    OFFSET_PAIR(4097, 11),  OFFSET_PAIR(6145, 11),  OFFSET_PAIR(8193, 12), OFFSET_PAIR(12289, 12),
    OFFSET_PAIR(16385, 13), OFFSET_PAIR(24577, 13),
};

#undef MATCHLEN_PAIR
#undef OFFSET_PAIR

} // namespace Ray

uint32_t Ray::Inflater::GetBits(const int n) {
    if (temp_bits_available_ < n) {
        if (cursor_ < input_.end()) {
            temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
            temp_bits_available_ += 8;
            if (cursor_ < input_.end()) {
                temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
                temp_bits_available_ += 8;
            }
        } else {
            return 0xffffffff;
        }
    }

    const uint32_t ret = temp_bits_ & ((1 << n) - 1);
    temp_bits_ >>= n;
    temp_bits_available_ -= n;
    return ret;
}

uint32_t Ray::Inflater::Peek16Bits() {
    if (temp_bits_available_ < 16) {
        if (cursor_ < input_.end()) {
            temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
            temp_bits_available_ += 8;
            if (cursor_ < input_.end()) {
                temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
                temp_bits_available_ += 8;
            }
        } else {
            return 0xffffffff;
        }
    }

    return temp_bits_ & 0xffff;
}

void Ray::Inflater::ConsumeBits(const int n) {
    assert(temp_bits_available_ >= n);
    temp_bits_ >>= n;
    temp_bits_available_ -= n;
}

void Ray::Inflater::Refill32Bits() {
    if (temp_bits_available_ <= 32 && (cursor_ + 4) <= input_.end()) {
        temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
        temp_bits_available_ += 8;
        temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
        temp_bits_available_ += 8;
        temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
        temp_bits_available_ += 8;
        temp_bits_ |= uint64_t(*cursor_++) << temp_bits_available_;
        temp_bits_available_ += 8;
    }
}

bool Ray::Inflater::SkipToByteBoundary() {
    while (temp_bits_available_ >= 8) {
        temp_bits_available_ -= 8;
        cursor_--;
        if (cursor_ < input_.data()) {
            return false;
        }
    }

    temp_bits_ = 0;
    temp_bits_available_ = 0;

    return true;
}

int Ray::Inflater::Inflate(Span<uint8_t> output) {
    // https://www.ietf.org/rfc/rfc1950.txt

    if (cursor_ + 2 >= input_.end()) {
        return -1;
    }

    const uint8_t CMF = cursor_[0], FLG = cursor_[1];

    const uint8_t CM = CMF & 0b00001111;
    if (CM != 8) {
        // Unknown compression method
        return -1;
    }
    const uint8_t CINFO = (CMF >> 4);
    if (CINFO > 7 || ((CMF << 8) | FLG) % 31 != 0) {
        return -1;
    }

    cursor_ += 2;

    const uint8_t FDICT = (FLG & 0b00100000);
    if (FDICT) {
        cursor_ += 4;
    }

    uint32_t BFINAL = 0;
    int output_offset = 0;

    do {
        BFINAL = GetBits(1);
        const uint32_t BTYPE = GetBits(2);

        switch (BTYPE) {
        case 0b00:
            // no compression
            output_offset +=
                CopyUncompressed(Span<uint8_t>{output.data() + output_offset, output.size() - output_offset});
            break;
        case 0b01:
            // fixed Huffman
            output_offset +=
                DecodeHuffman(false, Span<uint8_t>{output.data() + output_offset, output.size() - output_offset});
            break;
        case 0b10:
            // dynamic Huffman
            output_offset +=
                DecodeHuffman(true, Span<uint8_t>{output.data() + output_offset, output.size() - output_offset});
            break;
        default:
            return -1;
        }
    } while (!BFINAL);

    return output_offset;
}

uint32_t Ray::Inflater::CopyUncompressed(Span<uint8_t> output) {
    if (!SkipToByteBoundary()) {
        return 0xffffffff;
    }

    if (cursor_ + 4 > input_.end()) {
        return 0xffffffff;
    }

    const uint16_t len = uint16_t(cursor_[0]) | (uint16_t(cursor_[1]) << 8);
    cursor_ += 2;
    const uint16_t nlen = uint16_t(cursor_[0]) | (uint16_t(cursor_[1]) << 8);
    cursor_ += 2;

    if (len != ((~nlen) & 0xffff) || len > output.size()) {
        return -1;
    }

    memcpy(output.data(), cursor_, len);
    cursor_ += len;

    return len;
}

uint32_t Ray::Inflater::DecodeHuffman(const bool dynamic, Span<uint8_t> output) {
    // https://www.rfc-editor.org/rfc/rfc1951.html

    const int MaxOffsetSyms = 32;
    const int MaxCodeLenSyms = 19;
    const int CodeLenBits = 3;
    const int EODMarkerSym = 256;
    const int MatchLenSymStart = 257;

    huff_table_t literals_decode, offset_decode;

    if (dynamic) {
        uint32_t literal_syms = GetBits(5);
        if (literal_syms == 0xffffffff) {
            return 0xffffffff;
        }
        literal_syms += 257;
        if (literal_syms > HuffMaxSymbols) {
            return 0xffffffff;
        }

        uint32_t offset_syms = GetBits(5);
        if (offset_syms == 0xffffffff) {
            return 0xffffffff;
        }
        offset_syms += 1;
        if (offset_syms > MaxOffsetSyms) {
            return 0xffffffff;
        }

        uint32_t code_len_syms = GetBits(4);
        if (code_len_syms == 0xffffffff) {
            return 0xffffffff;
        }
        code_len_syms += 4;
        if (code_len_syms > MaxCodeLenSyms) {
            return 0xffffffff;
        }

        uint8_t code_length[HuffMaxSymbols + HuffOffsetSyms];
        if (ReadRawLengths(CodeLenBits, code_len_syms, HuffCodeLenSyms, code_length) < 0) {
            return -1;
        }
        huff_table_t tables_decode;
        if (huff_prepare_table(HuffCodeLenSyms, HuffCodeLenSyms, code_length, tables_decode) < 0) {
            return -1;
        }
        if (huff_finalize_table(tables_decode) < 0) {
            return -1;
        }

        if (ReadLength(tables_decode.rev_sym_table, literal_syms + offset_syms, HuffMaxSymbols + HuffOffsetSyms,
                       code_length, tables_decode) < 0) {
            return -1;
        }
        if (huff_prepare_table(literal_syms, HuffMaxSymbols, code_length, literals_decode) < 0 ||
            huff_prepare_table(offset_syms, HuffOffsetSyms, code_length + literal_syms, offset_decode) < 0) {
            return -1;
        }
    } else {
        uint8_t fixed_literal_code_len[HuffMaxSymbols];
        uint8_t fixed_offset_code_len[HuffOffsetSyms];

        int i;
        for (i = 0; i < 144; i++) {
            fixed_literal_code_len[i] = 8;
        }
        for (; i < 256; i++) {
            fixed_literal_code_len[i] = 9;
        }
        for (; i < 280; i++) {
            fixed_literal_code_len[i] = 7;
        }
        for (; i < HuffMaxSymbols; i++) {
            fixed_literal_code_len[i] = 8;
        }

        for (i = 0; i < HuffOffsetSyms; i++) {
            fixed_offset_code_len[i] = 5;
        }

        if (huff_prepare_table(HuffMaxSymbols, HuffMaxSymbols, fixed_literal_code_len, literals_decode) < 0 ||
            huff_prepare_table(HuffOffsetSyms, HuffOffsetSyms, fixed_offset_code_len, offset_decode) < 0) {
            return -1;
        }
    }

    for (int i = 0; i < HuffOffsetSyms; i++) {
        const uint32_t n = offset_decode.rev_sym_table[i];
        if (n < HuffOffsetSyms) {
            offset_decode.rev_sym_table[i] = HuffOffsetCode[n];
        }
    }

    for (int i = 0; i < HuffMaxSymbols; i++) {
        const uint32_t n = literals_decode.rev_sym_table[i];
        if (n >= MatchLenSymStart && n < HuffMaxSymbols - 2) {
            literals_decode.rev_sym_table[i] = HuffMatchLenCode[n - MatchLenSymStart];
        }
    }

    if (huff_finalize_table(literals_decode) < 0 || huff_finalize_table(offset_decode) < 0) {
        return -1;
    }

    uint8_t *current_out = output.data();
    const uint8_t *out_end = current_out + output.size();
    const uint8_t *out_fast_end = out_end - 15;

    while (true) {
        Refill32Bits();

        const uint32_t literals_code_word = ReadValue(literals_decode.rev_sym_table, literals_decode);
        if (literals_code_word < 256) {
            if (current_out < out_end) {
                *current_out++ = literals_code_word;
            } else {
                return -1;
            }
        } else {
            if (literals_code_word == EODMarkerSym) {
                break;
            }
            if (literals_code_word == -1) {
                return -1;
            }

            uint32_t match_length = GetBits((literals_code_word >> 16) & 15);
            if (match_length == -1) {
                return -1;
            }

            match_length += (literals_code_word & 0x7fff);

            const uint32_t offset_code_word = ReadValue(offset_decode.rev_sym_table, offset_decode);
            if (offset_code_word == -1) {
                return -1;
            }

            uint32_t match_offset = GetBits((offset_code_word >> 16) & 15);
            if (match_offset == -1) {
                return -1;
            }

            match_offset += (offset_code_word & 0x7fff);

            const uint8_t *src = current_out - match_offset;
            if (/*src >= output.data()*/ true) {
                if (match_offset >= 16 && (current_out + match_length) <= out_fast_end) {
                    const uint8_t *copy_src = src;
                    uint8_t *copy_dst = current_out;
                    const uint8_t *copy_end_dst = current_out + match_length;

                    do {
                        memcpy(copy_dst, copy_src, 16);
                        copy_src += 16;
                        copy_dst += 16;
                    } while (copy_dst < copy_end_dst);

                    current_out += match_length;
                } else {
                    if ((current_out + match_length) > out_end) {
                        return -1;
                    }

                    while (match_length--) {
                        *current_out++ = *src++;
                    }
                }
            } else {
                return -1;
            }
        }
    }

    return uint32_t(current_out - output.data());
}

int Ray::Inflater::ReadRawLengths(const int len_bits, const int read_symbols, const int symbols, uint8_t *code_length) {
    static const uint8_t code_len_syms[HuffCodeLenSyms] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                                           11, 4,  12, 3, 13, 2, 14, 1, 15};
    if (read_symbols < 0 || read_symbols > HuffMaxSymbols || symbols < 0 || symbols > HuffMaxSymbols ||
        read_symbols > symbols) {
        return -1;
    }

    int i = 0;
    while (i < read_symbols) {
        const uint32_t length = GetBits(len_bits);
        if (length == -1) {
            return -1;
        }
        code_length[code_len_syms[i++]] = length;
    }

    while (i < symbols) {
        code_length[code_len_syms[i++]] = 0;
    }

    return 0;
}

uint32_t Ray::Inflater::ReadValue(const uint32_t *rev_symbol_table, huff_table_t &out_table) {
    uint32_t stream = Peek16Bits();
    const uint32_t fast_sym_bits = out_table.fast_symbol[stream & ((1 << HuffFastSymbolBits) - 1)];
    if (fast_sym_bits) {
        ConsumeBits(fast_sym_bits >> 24);
        return fast_sym_bits & 0xffffff;
    }
    const uint32_t *rev_code_length_table = rev_symbol_table + out_table.symbols;
    uint32_t code_word = 0;
    int bits = 1;

    do {
        code_word |= (stream & 1);

        const uint32_t table_index = out_table.start_index[bits] + code_word;
        if (table_index < out_table.symbols) {
            if (bits == rev_code_length_table[table_index]) {
                ConsumeBits(bits);
                return rev_symbol_table[table_index];
            }
        }
        code_word <<= 1;
        stream >>= 1;
        bits++;
    } while (bits < 16);

    return -1;
}

int Ray::Inflater::ReadLength(const uint32_t *tables_rev_symbol_table, const int read_symbols, const int symbols,
                              uint8_t *code_length, huff_table_t &out_table) {
    if (read_symbols < 0 || symbols < 0 || read_symbols > symbols) {
        return -1;
    }

    int i = 0;
    uint32_t previous_length = 0;

    while (i < read_symbols) {
        const uint32_t length = ReadValue(tables_rev_symbol_table, out_table);
        if (length == -1) {
            return -1;
        }

        if (length < 16) {
            previous_length = length;
            code_length[i++] = previous_length;
        } else {
            uint32_t run_length = 0;

            if (length == 16) {
                int extra_run_length = GetBits(2);
                if (extra_run_length == -1) {
                    return -1;
                }
                run_length = 3 + extra_run_length;
            } else if (length == 17) {
                int extra_run_length = GetBits(3);
                if (extra_run_length == -1) {
                    return -1;
                }
                previous_length = 0;
                run_length = 3 + extra_run_length;
            } else if (length == 18) {
                int extra_run_length = GetBits(7);
                if (extra_run_length == -1) {
                    return -1;
                }
                previous_length = 0;
                run_length = 11 + extra_run_length;
            }

            while (run_length && i < read_symbols) {
                code_length[i++] = previous_length;
                run_length--;
            }
        }
    }

    while (i < symbols) {
        code_length[i++] = 0;
    }
    return 0;
}
