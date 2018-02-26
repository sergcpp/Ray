#include "TextureSplitter.h"

int ray::TextureSplitter::Allocate(const math::ivec2 &res, math::ivec2 &pos) {
    int i = Insert_Recursive(0, res);
    if (i != -1) {
        pos = nodes_[i].pos;
    }
    return i;
}

bool ray::TextureSplitter::Free(const math::ivec2 &pos) {
    int i = Find_Recursive(0, pos);
    return Free(i);
}

bool ray::TextureSplitter::Free(int i) {
    if (i == -1 || nodes_[i].is_free) return false;

    nodes_[i].is_free = true;

    int par = nodes_[i].parent;
    while (par != -1) {
        int ch0 = nodes_[par].child[0], ch1 = nodes_[par].child[1];

        if (nodes_[ch0].has_children() && nodes_[ch0].is_free &&
                nodes_[ch1].has_children() && nodes_[ch1].is_free) {

            SafeErase(ch0, &par, 1);
            ch1 = nodes_[par].child[1];
            SafeErase(ch1, &par, 1);

            nodes_[par].child[0] = nodes_[par].child[1] = -1;

            par = nodes_[par].parent;
        }
    }

    return true;
}

int ray::TextureSplitter::FindNode(const math::ivec2 &pos, math::ivec2 &size) const {
    int i = Find_Recursive(0, pos);
    if (i != -1) {
        size = nodes_[i].size;
    }
    return i;
}

int ray::TextureSplitter::Insert_Recursive(int i, const math::ivec2 &res) {
    if (!nodes_[i].is_free || res.x > nodes_[i].size.x || res.y > nodes_[i].size.y) {
        return -1;
    }

    int ch0 = nodes_[i].child[0], ch1 = nodes_[i].child[1];

    if (ch0 != -1) {
        int new_node = Insert_Recursive(ch0, res);
        if (new_node != -1) return new_node;

        return Insert_Recursive(ch1, res);
    } else {
        if (res.x == nodes_[i].size.x && res.y == nodes_[i].size.y) {
            nodes_[i].is_free = false;
            return i;
        }

        nodes_[i].child[0] = ch0 = (int)nodes_.size();
        nodes_.emplace_back();
        nodes_[i].child[1] = ch1 = (int)nodes_.size();
        nodes_.emplace_back();

        auto &n = nodes_[i];

        int dw = n.size.x - res.x;
        int dh = n.size.y - res.y;

        if (dw > dh) {
            nodes_[ch0].pos = n.pos;
            nodes_[ch0].size = { res.x, n.size.y };
            nodes_[ch1].pos = { n.pos.x + res.x, n.pos.y };
            nodes_[ch1].size = { n.size.x - res.x, n.size.y };
        } else {
            nodes_[ch0].pos = n.pos;
            nodes_[ch0].size = { n.size.x, res.y };
            nodes_[ch1].pos = { n.pos.x, n.pos.y + res.y };
            nodes_[ch1].size = { n.size.x, n.size.y - res.y };
        }

        nodes_[ch0].parent = nodes_[ch1].parent = i;

        return Insert_Recursive(ch0, res);
    }
}

int ray::TextureSplitter::Find_Recursive(int i, const math::ivec2 &pos) const {
    if (nodes_[i].is_free ||
            pos.x < nodes_[i].pos.x || pos.x >(nodes_[i].pos.x + nodes_[i].size.x) ||
            pos.y < nodes_[i].pos.y || pos.y >(nodes_[i].pos.y + nodes_[i].size.y)) {
        return -1;
    }

    int ch0 = nodes_[i].child[0], ch1 = nodes_[i].child[1];

    if (ch0 != -1) {
        int i = Find_Recursive(ch0, pos);
        if (i != -1) return i;
        return Find_Recursive(ch1, pos);
    } else {
        if (pos.x == nodes_[i].pos.x && pos.y == nodes_[i].pos.y) {
            return i;
        } else {
            return -1;
        }
    }
}

void ray::TextureSplitter::SafeErase(int i, int *indices, int num) {
    int last = (int)nodes_.size() - 1;

    if (last != i) {
        int ch0 = nodes_[last].child[0],
            ch1 = nodes_[last].child[1];

        if (ch0 != -1 && nodes_[i].parent != last) {
            nodes_[ch0].parent = nodes_[ch1].parent = i;
        }

        int par = nodes_[last].parent;

        if (nodes_[par].child[0] == last) {
            nodes_[par].child[0] = i;
        } else if (nodes_[par].child[1] == last) {
            nodes_[par].child[1] = i;
        }

        nodes_[i] = nodes_[last];
    }

    nodes_.erase(nodes_.begin() + last);

    for (int j = 0; j < num && indices; j++) {
        if (indices[j] == last) {
            indices[j] = i;
        }
    }
}