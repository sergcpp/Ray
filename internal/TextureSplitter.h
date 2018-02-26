#pragma once

#include <math/math.hpp>

namespace ray {
class TextureSplitter {
    struct node_t {
        int parent = -1;
        int child[2] = { -1, -1 };
        math::ivec2 pos, size;
        bool is_free = true;

        bool has_children() const {
            return child[0] != 0 || child[1] != 0;
        }
    };

    std::vector<node_t> nodes_;

    int Insert_Recursive(int i, const math::ivec2 &res);
    int Find_Recursive(int i, const math::ivec2 &pos) const;
    void SafeErase(int i, int *indices, int num);
public:
    explicit TextureSplitter(const math::ivec2 &res) {
        nodes_.emplace_back();
        nodes_.back().size = res;
    }

    bool empty() const {
        return nodes_.size() == 1;
    }

    int Allocate(const math::ivec2 &res, math::ivec2 &pos);
    bool Free(const math::ivec2 &pos);
    bool Free(int i);

    int FindNode(const math::ivec2 &pos, math::ivec2 &size) const;
};

}