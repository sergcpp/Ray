#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional> // for std::equal_to
#include <string>
#include <utility>

#include "simd/aligned_allocator.h"

namespace Ray {
inline uint32_t _lua_hash(void const *v, const uint32_t l) {
    uint32_t i, step = (l >> 5u) + 1;
    uint32_t h = l + (l >= 4 ? *(uint32_t *)v : 0);
    for (i = l; i >= step; i -= step) {
        h = h ^ ((h << 5u) + (h >> 2u) + ((unsigned char *)v)[i - 1]);
    }
    return h;
}

inline uint32_t _str_hash(const char *s) {
    const uint32_t A = 54059;
    const uint32_t B = 76963;
    // const uint32_t C = 86969;
    const uint32_t FIRSTH = 37;

    uint32_t h = FIRSTH;
    while (*s) {
        h = (h * A) ^ (s[0] * B);
        s++;
    }
    return h;
}

inline uint32_t _str_hash_len(const char *s, size_t len) {
    const uint32_t A = 54059;
    const uint32_t B = 76963;
    // const uint32_t C = 86969;
    const uint32_t FIRSTH = 37;

    uint32_t h = FIRSTH;
    while (len) {
        h = (h * A) ^ (s[0] * B);
        s++;
        len--;
    }
    return h;
}

template <typename K> class Hash {
  public:
    uint32_t operator()(const K &k) const { return _lua_hash(&k, sizeof(K)); }
};

template <> class Hash<char *> {
  public:
    uint32_t operator()(const char *s) const { return _str_hash(s); }
};

template <> class Hash<const char *> {
  public:
    uint32_t operator()(const char *s) const { return _str_hash(s); }
};

template <> class Hash<std::string> {
  public:
    uint32_t operator()(const std::string &s) const { return _str_hash_len(s.c_str(), s.length()); }
    uint32_t operator()(const char *s) const { return _str_hash(s); }
};

template <typename K> class Equal : std::equal_to<K> {
  public:
    bool operator()(const K &k1, const K &k2) const { return std::equal_to<K>::operator()(k1, k2); }
};

template <> class Equal<char *> {
  public:
    bool operator()(const char *k1, const char *k2) const { return strcmp(k1, k2) == 0; }
};

template <> class Equal<const char *> {
  public:
    bool operator()(const char *k1, const char *k2) const { return strcmp(k1, k2) == 0; }
};

template <> class Equal<std::string> {
  public:
    template <typename K2> bool operator()(const std::string &k1, const K2 &k2) const { return k1 == k2; }
    bool operator()(const std::string &k1, const char *k2) const { return k1 == k2; }
};

template <typename K, typename HashFunc = Hash<K>, typename KeyEqual = Equal<K>,
          typename Allocator = aligned_allocator<uint8_t, alignof(K)>>
class HashSet32 : HashFunc, KeyEqual, Allocator {
    static const uint8_t OccupiedBit = 0b10000000;
    static const uint8_t HashMask = 0b01111111;

  public:
    struct Node {
        uint32_t hash;
        K key;
    };

  private:
    uint8_t *ctrl_;
    Node *nodes_;
    uint32_t capacity_, size_;

    static uint32_t ctrl_size(const uint32_t cap) {
        return alignof(Node) * ((cap + alignof(Node) - 1) / alignof(Node));
    }
    static uint32_t mem_size(const uint32_t cap) { return ctrl_size(cap) + sizeof(Node) * cap; }

  public:
    explicit HashSet32(const HashFunc &hash_func = HashFunc(), const KeyEqual &key_equal = KeyEqual(),
                       const Allocator &alloc = Allocator()) noexcept
        : HashFunc(hash_func), KeyEqual(key_equal), Allocator(alloc), ctrl_(nullptr), nodes_(nullptr), capacity_(0),
          size_(0) {}

    explicit HashSet32(const uint32_t capacity, const HashFunc &hash_func = HashFunc(),
                       const KeyEqual &key_equal = KeyEqual(), const Allocator &alloc = Allocator())
        : HashFunc(hash_func), KeyEqual(key_equal), Allocator(alloc), ctrl_(nullptr), nodes_(nullptr), capacity_(0),
          size_(0) {
        ReserveRealloc(capacity);
    }

    explicit HashSet32(std::initializer_list<K> l, const HashFunc &hash_func = HashFunc(),
                       const KeyEqual &key_equal = KeyEqual(), const Allocator &alloc = Allocator()) noexcept
        : HashFunc(hash_func), KeyEqual(key_equal), Allocator(alloc), ctrl_(nullptr), nodes_(nullptr), capacity_(0),
          size_(0) {
        ReserveRealloc(uint32_t(l.size()));
        for (auto it = l.begin(); it != l.end(); ++it) {
            Insert(*it);
        }
    }

    HashSet32(const HashSet32 &rhs) = delete;
    HashSet32 &operator=(const HashSet32 &rhs) = delete;

    HashSet32(HashSet32 &&rhs) noexcept { (*this) = std::move(rhs); }
    HashSet32 &operator=(HashSet32 &&rhs) noexcept {
        if (this == &rhs) {
            return (*this);
        }
        Allocator::operator=(static_cast<Allocator &&>(rhs));
        HashFunc::operator=(static_cast<HashFunc &&>(rhs));
        KeyEqual::operator=(static_cast<KeyEqual &&>(rhs));
        ctrl_ = std::exchange(rhs.ctrl_, nullptr);
        nodes_ = std::exchange(rhs.nodes_, nullptr);
        capacity_ = std::exchange(rhs.capacity_, 0);
        size_ = std::exchange(rhs.size_, 0);
        return (*this);
    }

    ~HashSet32() {
        clear();
        this->deallocate(ctrl_, mem_size(capacity_));
    }

    uint32_t size() const { return size_; }
    uint32_t capacity() const { return capacity_; }

    bool empty() const { return size_ == 0; }

    void clear() {
        for (uint32_t i = 0; i < capacity_ && size_; i++) {
            if (ctrl_[i] & OccupiedBit) {
                --size_;
                nodes_[i].key.~K();
            }
        }
        memset(ctrl_, 0, capacity_);
        assert(size_ == 0);
    }

    void reserve(const uint32_t capacity) { ReserveRealloc(capacity); }

    bool Insert(const K &key) {
        const K *k = Find(key);
        if (k) {
            return false;
        }
        InsertNoCheck(key);
        return true;
    }

    bool Insert(K &&key) {
        const uint32_t hash = HashFunc::operator()(key);

        const K *k = Find(hash, key);
        if (k) {
            return false;
        }
        InsertInternal(hash, std::forward<K>(key));
        return true;
    }

    Node *InsertNoCheck(const K &key) {
        const uint32_t hash = HashFunc::operator()(key);
        return InsertInternal(hash, key);
    }

    bool Erase(const K &key) {
        const uint32_t hash = HashFunc::operator()(key);
        const uint8_t ctrl_to_find = OccupiedBit | (hash & HashMask);

        uint32_t i = hash & (capacity_ - 1);
        const uint32_t end = i;
        while (ctrl_[i]) {
            if (ctrl_[i] == ctrl_to_find && nodes_[i].hash == hash && KeyEqual::operator()(nodes_[i].key, key)) {
                --size_;
                ctrl_[i] = HashMask;
                nodes_[i].key.~K();

                return true;
            }
            i = (i + 1) & (capacity_ - 1);
            if (i == end) {
                break;
            }
        }

        return false;
    }

    template <typename K2> const K *Find(const K2 &key) const { return Find(HashFunc::operator()(key), key); }
    template <typename K2> K *Find(const K2 &key) { return Find(HashFunc::operator()(key), key); }

    template <typename K2> const K *Find(const uint32_t hash, const K2 &key) const {
        if (!capacity_) {
            return nullptr;
        }

        const uint8_t ctrl_to_find = OccupiedBit | (hash & HashMask);

        uint32_t i = hash & (capacity_ - 1);
        const uint32_t end = i;
        while (ctrl_[i]) {
            if (ctrl_[i] == ctrl_to_find && nodes_[i].hash == hash && KeyEqual::operator()(nodes_[i].key, key)) {
                return &nodes_[i].key;
            }
            i = (i + 1) & (capacity_ - 1);
            if (i == end) {
                break;
            }
        }

        return nullptr;
    }

    template <typename K2> K *Find(const uint32_t hash, const K2 &key) {
        return const_cast<K *>(const_cast<const HashSet32 *>(this)->Find(hash, key));
    }

    Node *GetOrNull(const uint32_t index) {
        if (index < capacity_ && (ctrl_[index / 8] & (1u << (index % 8)))) {
            return &nodes_[index];
        } else {
            return nullptr;
        }
    }

    const Node *GetOrNull(const uint32_t index) const {
        if (index < capacity_ && (ctrl_[index / 8] & (1u << (index % 8)))) {
            return &nodes_[index];
        } else {
            return nullptr;
        }
    }

    class HashSet32Iterator {
        friend class HashSet32<K, HashFunc, KeyEqual>;

        HashSet32<K, HashFunc, KeyEqual> *container_;
        uint32_t index_;

        HashSet32Iterator(HashSet32<K, HashFunc, KeyEqual> *container, uint32_t index)
            : container_(container), index_(index) {}

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = K;
        using difference_type = std::ptrdiff_t;
        using pointer = K *;
        using reference = K &;

        K &operator*() { return container_->at(index_).key; }
        K *operator->() { return &container_->at(index_).key; }
        HashSet32Iterator &operator++() {
            index_ = container_->NextOccupied(index_);
            return *this;
        }
        HashSet32Iterator operator++(int) {
            HashSet32Iterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return index_; }

        bool operator<(const HashSet32Iterator &rhs) { return index_ < rhs.index_; }
        bool operator<=(const HashSet32Iterator &rhs) { return index_ <= rhs.index_; }
        bool operator>(const HashSet32Iterator &rhs) { return index_ > rhs.index_; }
        bool operator>=(const HashSet32Iterator &rhs) { return index_ >= rhs.index_; }
        bool operator==(const HashSet32Iterator &rhs) { return index_ == rhs.index_; }
        bool operator!=(const HashSet32Iterator &rhs) { return index_ != rhs.index_; }
    };

    class HashSet32ConstIterator {
        friend class HashSet32<K, HashFunc, KeyEqual>;

        const HashSet32<K, HashFunc, KeyEqual> *container_;
        uint32_t index_;

        HashSet32ConstIterator(const HashSet32<K, HashFunc, KeyEqual> *container, uint32_t index)
            : container_(container), index_(index) {}

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = K;
        using difference_type = std::ptrdiff_t;
        using pointer = K *;
        using reference = K &;

        const K &operator*() { return container_->at(index_).key; }
        const K *operator->() { return &container_->at(index_).key; }
        HashSet32ConstIterator &operator++() {
            index_ = container_->NextOccupied(index_);
            return *this;
        }
        HashSet32ConstIterator operator++(int) {
            HashSet32ConstIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return index_; }

        bool operator<(const HashSet32ConstIterator &rhs) { return index_ < rhs.index_; }
        bool operator<=(const HashSet32ConstIterator &rhs) { return index_ <= rhs.index_; }
        bool operator>(const HashSet32ConstIterator &rhs) { return index_ > rhs.index_; }
        bool operator>=(const HashSet32ConstIterator &rhs) { return index_ >= rhs.index_; }
        bool operator==(const HashSet32ConstIterator &rhs) { return index_ == rhs.index_; }
        bool operator!=(const HashSet32ConstIterator &rhs) { return index_ != rhs.index_; }
    };

    using iterator = HashSet32Iterator;
    using const_iterator = HashSet32ConstIterator;

    iterator begin() {
        for (uint32_t i = 0; i < capacity_; i++) {
            if (ctrl_[i] & OccupiedBit) {
                return iterator(this, i);
            }
        }
        return end();
    }
    const_iterator begin() const { return cbegin(); }

    const_iterator cbegin() const {
        for (uint32_t i = 0; i < capacity_; i++) {
            if (ctrl_[i] & OccupiedBit) {
                return const_iterator(this, i);
            }
        }
        return cend();
    }

    iterator end() { return iterator(this, capacity_); }
    const_iterator end() const { return const_iterator(this, capacity_); }
    const_iterator cend() const { return const_iterator(this, capacity_); }

    iterator iter_at(const uint32_t i) { return iterator(this, i); }
    const_iterator citer_at(const uint32_t i) const { return const_iterator(this, i); }

    Node &at(const uint32_t index) {
        assert((ctrl_[index] & OccupiedBit) && "Invalid index!");
        return nodes_[index];
    }

    const Node &at(const uint32_t index) const {
        assert((ctrl_[index] & OccupiedBit) && "Invalid index!");
        return nodes_[index];
    }

    iterator erase(const iterator it) {
        const uint32_t next = NextOccupied(it.index_);

        --size_;
        ctrl_[it.index_] = HashMask;
        nodes_[it.index_].key.~K();

        return iter_at(next);
    }

  private:
    void CheckRealloc() {
        if ((size_ + 1) > uint32_t(0.8f * capacity_)) {
            ReserveRealloc(capacity_ * 2);
        }
    }

    void ReserveRealloc(const uint32_t desired_capacity) {
        if (!capacity_ || capacity_ < desired_capacity) {
            uint8_t *old_ctrl = ctrl_;
            Node *old_nodes = nodes_;
            uint32_t old_capacity = capacity_;

            if (!capacity_) {
                capacity_ = 8;
            }
            while (capacity_ < desired_capacity) {
                capacity_ *= 2;
            }
            size_ = 0;

            ctrl_ = this->allocate(mem_size(capacity_));
            if (!ctrl_) {
                return;
            }
            nodes_ = reinterpret_cast<Node *>(&ctrl_[ctrl_size(capacity_)]);
            memset(ctrl_, 0, capacity_);

            for (uint32_t i = 0; i < old_capacity; ++i) {
                if (old_ctrl[i] & OccupiedBit) {
                    InsertInternal(old_nodes[i].hash, std::move(old_nodes[i].key));
                }
            }

            this->deallocate(old_ctrl, mem_size(old_capacity));
        }
    }

    Node *InsertInternal(const uint32_t hash, const K &key) {
        CheckRealloc();

        uint32_t i = hash & (capacity_ - 1);
        while (ctrl_[i] & OccupiedBit) {
            i = (i + 1) & (capacity_ - 1);
        }

        ++size_;
        ctrl_[i] = OccupiedBit | (hash & HashMask);

        Node *ret = &nodes_[i];
        ret->hash = hash;
        new (&ret->key) K(key);

        return ret;
    }

    void InsertInternal(const uint32_t hash, K &&key) {
        CheckRealloc();

        uint32_t i = hash & (capacity_ - 1);
        while (ctrl_[i] & OccupiedBit) {
            i = (i + 1) & (capacity_ - 1);
        }

        ++size_;
        ctrl_[i] = OccupiedBit | (hash & HashMask);

        Node &ret = nodes_[i];
        ret.hash = hash;
        new (&ret.key) K(std::forward<K>(key));
    }

    uint32_t NextOccupied(uint32_t index) const {
        assert((ctrl_[index] & OccupiedBit) && "Invalid index!");
        for (uint32_t i = index + 1; i < capacity_; ++i) {
            if (ctrl_[i] & OccupiedBit) {
                return i;
            }
        }
        return capacity_;
    }
};
} // namespace Ray
