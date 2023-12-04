#pragma once

#include <cstdint>

#include <vector>

#include "../../Span.h"

namespace Ray {
struct huff_node_t {
    uint32_t left, right;
    int freq;
    char c;
};

inline uint32_t huff_parent(const uint32_t i) { return (i - 1) / 2; }
inline uint32_t huff_left(const uint32_t i) { return i * 2 + 1; }
inline uint32_t huff_right(const uint32_t i) { return i * 2 + 2; }

void huff_insert(std::vector<uint32_t> &q, Span<const huff_node_t> nodes, uint32_t n);
void huff_heapify(std::vector<uint32_t> &q, Span<const huff_node_t> nodes, uint32_t i);

uint32_t huff_extract_min(std::vector<uint32_t> &q, Span<const huff_node_t> nodes);
uint32_t huff_build_tree(Span<const char> input, const uint32_t freq[256], std::vector<Ray::huff_node_t> &out_nodes);

static const int HuffMaxSymbols = 288;
static const int HuffCodeLenSyms = 19;
static const int HuffFastSymbolBits = 10;

struct huff_table_t {
    uint32_t fast_symbol[1 << HuffFastSymbolBits];
    uint32_t start_index[16];
    uint32_t symbols;
    uint32_t num_sorted, starting_pos[16];
    uint32_t rev_sym_table[HuffMaxSymbols * 2];
};

int huff_prepare_table(int read_symbols, int symbols, uint8_t *code_length, huff_table_t &out_table);
int huff_finalize_table(huff_table_t &out_table);

} // namespace Ray