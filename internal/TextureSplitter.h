#pragma once

#include <vector>

namespace Ray {
class TextureSplitter {
    struct node_t {
        int parent = -1;
        int child[2] = { -1, -1 };
        int pos[2], size[2];
        bool is_free = true;

        bool has_children() const {
            return child[0] != 0 || child[1] != 0;
        }
    };

    std::vector<node_t> nodes_;

    int Insert_Recursive(int i, const int res[2]);
    int Find_Recursive(int i, const int pos[2]) const;
    void SafeErase(int i, int *indices, int num);
public:
    explicit TextureSplitter(const int res[2]) {
        nodes_.emplace_back();
        nodes_.back().size[0] = res[0];
        nodes_.back().size[1] = res[1];
    }

    int resx() const { return nodes_[0].size[0]; }
    int resy() const { return nodes_[0].size[1]; }

    bool empty() const {
        return nodes_.size() == 1;
    }

    int Allocate(const int res[2], int pos[2]);
    bool Free(const int pos[2]);
    bool Free(int i);

    int FindNode(const int pos[2], int size[2]) const;
};

}