#include "test_common.h"

#include <cstring>

#include "../internal/inflate/Huffman.h"

struct huff_code_t {
    uint16_t start, len;
};

void traverse(Ray::Span<const Ray::huff_node_t> nodes, uint32_t i, int level, std::vector<char> &codes_heap,
              char code_so_far[], huff_code_t codes[]) {
    if (nodes[i].left == 0xffffffff && nodes[i].right == 0xffffffff) {
        codes[uint8_t(nodes[i].c)] = {uint16_t(codes_heap.size()), uint16_t(level)};
        codes_heap.insert(end(codes_heap), code_so_far, code_so_far + level);
    } else {
        code_so_far[level] = '0';
        traverse(nodes, nodes[i].left, level + 1, codes_heap, code_so_far, codes);
        code_so_far[level] = '1';
        traverse(nodes, nodes[i].right, level + 1, codes_heap, code_so_far, codes);
    }
}

void test_huffman() {
    using namespace Ray;

    printf("Test huffman            | ");

    const char test_str[] = "aaaaaaaaaaaaaaaaaaaaaaaaaabbdcccsdfasdfasdfwcsddddccccd";

    uint32_t freq[256] = {};
    for (int i = 0; i < sizeof(test_str) - 1; ++i) {
        freq[uint8_t(test_str[i])]++;
    }

    std::vector<huff_node_t> nodes;
    const uint32_t root = huff_build_tree(test_str, freq, nodes);
    require(nodes[root].freq == 55);

    char temp_code[8];
    huff_code_t final_codes[256] = {};
    std::vector<char> codes_heap;
    traverse(nodes, root, 0, codes_heap, temp_code, final_codes);

    auto check_code = [&](char c, const char expected[]) {
        const char *code = &codes_heap[final_codes[uint8_t(c)].start];
        if (final_codes[uint8_t(c)].len != strlen(expected)) {
            return false;
        }
        for (int i = 0; i < final_codes[uint8_t(c)].len; ++i) {
            if (code[i] != expected[i]) {
                return false;
            }
        }
        return true;
    };

    // https://www.csfieldguide.org.nz/en/interactives/huffman-tree/
    require(check_code('a', "1"));
    require(check_code('s', "000"));
    require(check_code('f', "0010"));
    require(check_code('b', "00111"));
    require(check_code('w', "00110"));
    require(check_code('d', "011"));
    require(check_code('c', "010"));

    // for (int i = 0; i < 256; ++i) {
    //    if (freq[i]) {
    //        printf("[%.2i] %c : %.*s\n", freq[i], char(i), final_codes[i].len, &codes_heap[final_codes[i].start]);
    //    }
    //}

    printf("OK\n");
}