#include "TextureSplitter.h"

int Ray::TextureSplitter::Allocate(const int res[2], int pos[2]) {
    const int i = Insert_Recursive(0, res);
    if (i != -1) {
        pos[0] = nodes_[i].pos[0];
        pos[1] = nodes_[i].pos[1];
    }
    return i;
}

bool Ray::TextureSplitter::Free(const int pos[2]) {
    const int i = Find_Recursive(0, pos);
    return Free(i);
}

bool Ray::TextureSplitter::Free(const int i) {
    if (i == -1 || nodes_[i].is_free) {
        return false;
    }

    nodes_[i].is_free = true;

    int par = nodes_[i].parent;
    while (par != -1) {
        int ch0 = nodes_[par].child[0], ch1 = nodes_[par].child[1];
        if (!nodes_[ch0].has_children() && nodes_[ch0].is_free && !nodes_[ch1].has_children() && nodes_[ch1].is_free) {

            SafeErase(ch0, &par, 1);
            ch1 = nodes_[par].child[1];
            SafeErase(ch1, &par, 1);

            nodes_[par].child[0] = nodes_[par].child[1] = -1;

            par = nodes_[par].parent;
        } else {
            par = -1;
        }
    }

    return true;
}

int Ray::TextureSplitter::FindNode(const int pos[2], int size[2]) const {
    const int i = Find_Recursive(0, pos);
    if (i != -1) {
        size[0] = nodes_[i].size[0];
        size[1] = nodes_[i].size[1];
    }
    return i;
}

int Ray::TextureSplitter::Insert_Recursive(int i, const int res[2]) {
    if (!nodes_[i].is_free || res[0] > nodes_[i].size[0] || res[1] > nodes_[i].size[1]) {
        return -1;
    }

    int ch0 = nodes_[i].child[0], ch1 = nodes_[i].child[1];

    if (ch0 != -1) {
        const int new_node = Insert_Recursive(ch0, res);
        if (new_node != -1) {
            return new_node;
        }

        return Insert_Recursive(ch1, res);
    } else {
        if (res[0] == nodes_[i].size[0] && res[1] == nodes_[i].size[1]) {
            nodes_[i].is_free = false;
            return i;
        }

        nodes_[i].child[0] = ch0 = (int)nodes_.size();
        nodes_.emplace_back();
        nodes_[i].child[1] = ch1 = (int)nodes_.size();
        nodes_.emplace_back();

        node_t &n = nodes_[i];

        const int dw = n.size[0] - res[0];
        const int dh = n.size[1] - res[1];

        if (dw > dh) {
            nodes_[ch0].pos[0] = n.pos[0];
            nodes_[ch0].pos[1] = n.pos[1];
            nodes_[ch0].size[0] = res[0];
            nodes_[ch0].size[1] = n.size[1];
            nodes_[ch1].pos[0] = n.pos[0] + res[0];
            nodes_[ch1].pos[1] = n.pos[1];
            nodes_[ch1].size[0] = n.size[0] - res[0];
            nodes_[ch1].size[1] = n.size[1];
        } else {
            nodes_[ch0].pos[0] = n.pos[0];
            nodes_[ch0].pos[1] = n.pos[1];
            nodes_[ch0].size[0] = n.size[0];
            nodes_[ch0].size[1] = res[1];
            nodes_[ch1].pos[0] = n.pos[0];
            nodes_[ch1].pos[1] = n.pos[1] + res[1];
            nodes_[ch1].size[0] = n.size[0];
            nodes_[ch1].size[1] = n.size[1] - res[1];
        }

        nodes_[ch0].parent = nodes_[ch1].parent = i;

        return Insert_Recursive(ch0, res);
    }
}

int Ray::TextureSplitter::Find_Recursive(const int i, const int pos[2]) const {
    if (nodes_[i].is_free || pos[0] < nodes_[i].pos[0] || pos[0] > (nodes_[i].pos[0] + nodes_[i].size[0]) ||
        pos[1] < nodes_[i].pos[1] || pos[1] > (nodes_[i].pos[1] + nodes_[i].size[1])) {
        return -1;
    }

    const int ch0 = nodes_[i].child[0], ch1 = nodes_[i].child[1];

    if (ch0 != -1) {
        const int ndx = Find_Recursive(ch0, pos);
        if (ndx != -1) {
            return ndx;
        }
        return Find_Recursive(ch1, pos);
    } else {
        if (pos[0] == nodes_[i].pos[0] && pos[1] == nodes_[i].pos[1]) {
            return i;
        } else {
            return -1;
        }
    }
}

void Ray::TextureSplitter::SafeErase(const int i, int *indices, const int num) {
    const int last = int(nodes_.size()) - 1;

    if (last != i) {
        const int ch0 = nodes_[last].child[0], ch1 = nodes_[last].child[1];

        if (ch0 != -1 && nodes_[i].parent != last) {
            nodes_[ch0].parent = nodes_[ch1].parent = i;
        }

        const int par = nodes_[last].parent;

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