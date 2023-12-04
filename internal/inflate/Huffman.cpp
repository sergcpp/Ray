#include "Huffman.h"

void Ray::huff_insert(std::vector<uint32_t> &q, Span<const huff_node_t> nodes, const uint32_t n) {
    uint32_t i = uint32_t(q.size());
    q.emplace_back();

    while ((i > 0) && (nodes[q[huff_parent(i)]].freq > nodes[n].freq)) {
        q[i] = q[huff_parent(i)];
        i = huff_parent(i);
    }

    q[i] = n;
}

void Ray::huff_heapify(std::vector<uint32_t> &q, Span<const huff_node_t> nodes, uint32_t i) {
    const uint32_t l = huff_left(i);
    const uint32_t r = huff_right(i);

    uint32_t smallest;
    if (l < q.size() && nodes[q[l]].freq < nodes[q[i]].freq) {
        smallest = l;
    } else {
        smallest = i;
    }
    if (r < q.size() && nodes[q[r]].freq < nodes[q[smallest]].freq) {
        smallest = r;
    }

    if (smallest != i) {
        std::swap(q[i], q[smallest]);
        huff_heapify(q, nodes, smallest);
    }
}

uint32_t Ray::huff_extract_min(std::vector<uint32_t> &q, Span<const huff_node_t> nodes) {
    assert(!q.empty());

    uint32_t ret = q[0];

    q[0] = q.back();
    q.pop_back();

    huff_heapify(q, nodes, 0);
    return ret;
}

uint32_t Ray::huff_build_tree(Span<const char> input, const uint32_t freq[256],
                              std::vector<Ray::huff_node_t> &out_nodes) {
    std::vector<uint32_t> queue;
    for (int i = 0; i < 256; ++i) {
        if (freq[i]) {
            out_nodes.emplace_back();
            huff_node_t &new_node = out_nodes.back();
            new_node.left = new_node.right = 0xffffffff;
            new_node.freq = freq[i];
            new_node.c = char(i);

            huff_insert(queue, out_nodes, uint32_t(out_nodes.size() - 1));
        }
    }

    const uint32_t n = uint32_t(queue.size() - 1);
    for (uint32_t i = 0; i < n; ++i) {
        out_nodes.emplace_back();
        huff_node_t &new_node = out_nodes.back();
        new_node.left = huff_extract_min(queue, out_nodes);
        new_node.right = huff_extract_min(queue, out_nodes);
        new_node.freq = out_nodes[new_node.left].freq + out_nodes[new_node.right].freq;

        huff_insert(queue, out_nodes, uint32_t(out_nodes.size() - 1));
    }

    return queue[0];
}

int Ray::huff_prepare_table(const int read_symbols, const int symbols, uint8_t *code_length, huff_table_t &out_table) {
    if (read_symbols < 0 || read_symbols > HuffMaxSymbols || symbols < 0 || symbols > HuffMaxSymbols ||
        read_symbols > symbols) {
        return -1;
    }
    out_table.symbols = symbols;

    int num_symbols_per_len[16] = {};
    for (int i = 0; i < read_symbols; i++) {
        if (code_length[i] >= 16) {
            return -1;
        }
        num_symbols_per_len[code_length[i]]++;
    }

    out_table.starting_pos[0] = 0;
    out_table.num_sorted = 0;
    for (int i = 1; i < 16; i++) {
        out_table.starting_pos[i] = out_table.num_sorted;
        out_table.num_sorted += num_symbols_per_len[i];
    }

    for (int i = 0; i < symbols; i++) {
        out_table.rev_sym_table[i] = -1;
    }

    for (int i = 0; i < read_symbols; i++) {
        if (code_length[i]) {
            out_table.rev_sym_table[out_table.starting_pos[code_length[i]]++] = i;
        }
    }

    return 0;
}

int Ray::huff_finalize_table(huff_table_t &out_table) {
    uint32_t canonical_code_word = 0;
    uint32_t *rev_code_length_table = out_table.rev_sym_table + out_table.symbols;
    int canonical_length = 1;

    uint32_t i;
    for (i = 0; i < (1 << HuffFastSymbolBits); i++) {
        out_table.fast_symbol[i] = 0;
    }
    for (i = 0; i < 16; i++) {
        out_table.start_index[i] = 0;
    }
    i = 0;
    while (i < out_table.num_sorted) {
        if (canonical_length >= 16) {
            return -1;
        }
        out_table.start_index[canonical_length] = i - canonical_code_word;

        while (i < out_table.starting_pos[canonical_length]) {
            if (i >= out_table.symbols) {
                return -1;
            }
            rev_code_length_table[i] = canonical_length;

            if (canonical_code_word >= (1U << canonical_length)) {
                return -1;
            }

            if (canonical_length <= HuffFastSymbolBits) {
                uint32_t rev_word;

                // Get upside down codeword (branchless method by Eric Biggers)
                rev_word = ((canonical_code_word & 0x5555) << 1) | ((canonical_code_word & 0xaaaa) >> 1);
                rev_word = ((rev_word & 0x3333) << 2) | ((rev_word & 0xcccc) >> 2);
                rev_word = ((rev_word & 0x0f0f) << 4) | ((rev_word & 0xf0f0) >> 4);
                rev_word = ((rev_word & 0x00ff) << 8) | ((rev_word & 0xff00) >> 8);
                rev_word = rev_word >> (16 - canonical_length);

                int slots = 1 << (HuffFastSymbolBits - canonical_length);
                while (slots) {
                    out_table.fast_symbol[rev_word] =
                        (out_table.rev_sym_table[i] & 0xffffff) | (canonical_length << 24);
                    rev_word += (1 << canonical_length);
                    slots--;
                }
            }

            i++;
            canonical_code_word++;
        }
        canonical_length++;
        canonical_code_word <<= 1;
    }

    while (i < out_table.symbols) {
        out_table.rev_sym_table[i] = -1;
        rev_code_length_table[i++] = 0;
    }

    return 0;
}