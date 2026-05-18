#pragma once

#include "HashSet32.h"

namespace Ray {
template <int A, int B> constexpr int max_value = (A > B) ? A : B;

template <typename K, typename V, typename HashFunc = Hash<K>, typename KeyEqual = Equal<K>,
          typename Allocator = aligned_allocator<uint8_t, max_value<alignof(K), alignof(V)>>>
class HashMap32 : HashFunc, KeyEqual, Allocator {
    static const uint8_t OccupiedBit = 0b10000000;
    static const uint8_t HashMask = 0b01111111;

  public:
    struct Node {
        uint32_t hash;
        K key;
        V val;
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
    explicit HashMap32(const HashFunc &hash_func = HashFunc(), const KeyEqual &key_equal = KeyEqual(),
                       const Allocator &alloc = Allocator()) noexcept
        : HashFunc(hash_func), KeyEqual(key_equal), Allocator(alloc), ctrl_(nullptr), nodes_(nullptr), capacity_(0),
          size_(0) {}

    explicit HashMap32(const uint32_t capacity, const HashFunc &hash_func = HashFunc(),
                       const KeyEqual &key_equal = KeyEqual(), const Allocator &alloc = Allocator())
        : HashFunc(hash_func), KeyEqual(key_equal), Allocator(alloc), ctrl_(nullptr), nodes_(nullptr), capacity_(0),
          size_(0) {
        ReserveRealloc(capacity);
    }

    explicit HashMap32(std::initializer_list<std::pair<K, V>> l, const HashFunc &hash_func = HashFunc(),
                       const KeyEqual &key_equal = KeyEqual(), const Allocator &alloc = Allocator()) noexcept
        : HashFunc(hash_func), KeyEqual(key_equal), Allocator(alloc), ctrl_(nullptr), nodes_(nullptr), capacity_(0),
          size_(0) {
        ReserveRealloc(uint32_t(l.size()));
        for (auto it = l.begin(); it != l.end(); ++it) {
            Insert(it->first, it->second);
        }
    }

    HashMap32(const HashMap32 &rhs) = delete;
    HashMap32 &operator=(const HashMap32 &rhs) = delete;

    HashMap32(HashMap32 &&rhs) noexcept : ctrl_(nullptr), nodes_(nullptr), capacity_(0), size_(0) {
        (*this) = std::move(rhs);
    }
    HashMap32 &operator=(HashMap32 &&rhs) noexcept {
        if (this == &rhs) {
            return (*this);
        }
        if (ctrl_) {
            clear();
            this->deallocate(ctrl_, mem_size(capacity_));
        }
        if constexpr (std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
            Allocator::operator=(static_cast<Allocator &&>(rhs));
        }
        HashFunc::operator=(static_cast<HashFunc &&>(rhs));
        KeyEqual::operator=(static_cast<KeyEqual &&>(rhs));
        ctrl_ = std::exchange(rhs.ctrl_, nullptr);
        nodes_ = std::exchange(rhs.nodes_, nullptr);
        capacity_ = std::exchange(rhs.capacity_, 0);
        size_ = std::exchange(rhs.size_, 0);
        return (*this);
    }

    ~HashMap32() {
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
                nodes_[i].val.~V();
            }
        }
        memset(ctrl_, 0, capacity_);
        assert(size_ == 0);
    }

    void reserve(const uint32_t capacity) { ReserveRealloc(capacity); }

    V &operator[](const K &key) {
        V *v = Find(key);
        if (!v) {
            v = InsertNoCheck(key);
        }
        return *v;
    }

    bool Insert(const K &key, const V &val) {
        const V *v = Find(key);
        if (v) {
            return false;
        }
        *InsertNoCheck(key) = val;
        return true;
    }

    bool Insert(K &&key, V &&val) {
        const uint32_t hash = HashFunc::operator()(key);

        const V *v = Find(hash, key);
        if (v) {
            return false;
        }
        InsertInternal(hash, std::forward<K>(key), std::forward<V>(val));
        return true;
    }

    void Set(const K &key, const V &val) {
        V *v = Find(key);
        if (!v) {
            v = InsertNoCheck(key);
        }
        (*v) = val;
    }

    V *InsertNoCheck(const K &key) {
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
                nodes_[i].val.~V();

                return true;
            }
            i = (i + 1) & (capacity_ - 1);
            if (i == end) {
                break;
            }
        }

        return false;
    }

    template <typename K2> const V *Find(const K2 &key) const { return Find(HashFunc::operator()(key), key); }

    template <typename K2> V *Find(const K2 &key) { return Find(HashFunc::operator()(key), key); }

    template <typename K2> const V *Find(const uint32_t hash, const K2 &key) const {
        if (!capacity_) {
            return nullptr;
        }

        const uint8_t ctrl_to_find = OccupiedBit | (hash & HashMask);

        uint32_t i = hash & (capacity_ - 1);
        const uint32_t end = i;
        while (ctrl_[i]) {
            if (ctrl_[i] == ctrl_to_find && nodes_[i].hash == hash && KeyEqual::operator()(nodes_[i].key, key)) {
                return &nodes_[i].val;
            }
            i = (i + 1) & (capacity_ - 1);
            if (i == end) {
                break;
            }
        }

        return nullptr;
    }

    template <typename K2> V *Find(const uint32_t hash, const K2 &key) {
        return const_cast<V *>(std::as_const(*this).Find(hash, key));
    }

    Node *GetOrNull(const uint32_t index) {
        if (index < capacity_ && (ctrl_[index] & OccupiedBit)) {
            return &nodes_[index];
        } else {
            return nullptr;
        }
    }

    const Node *GetOrNull(const uint32_t index) const {
        if (index < capacity_ && (ctrl_[index] & OccupiedBit)) {
            return &nodes_[index];
        } else {
            return nullptr;
        }
    }

    class HashMap32Iterator {
        friend class HashMap32<K, V, HashFunc, KeyEqual>;

        HashMap32<K, V, HashFunc, KeyEqual> *container_;
        uint32_t index_;

        HashMap32Iterator(HashMap32<K, V, HashFunc, KeyEqual> *container, uint32_t index)
            : container_(container), index_(index) {}

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Node;
        using difference_type = std::ptrdiff_t;
        using pointer = Node *;
        using reference = Node &;

        Node &operator*() { return container_->at(index_); }
        Node *operator->() { return &container_->at(index_); }
        HashMap32Iterator &operator++() {
            index_ = container_->NextOccupied(index_);
            return *this;
        }
        HashMap32Iterator operator++(int) {
            HashMap32Iterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return index_; }

        bool operator<(const HashMap32Iterator &rhs) { return index_ < rhs.index_; }
        bool operator<=(const HashMap32Iterator &rhs) { return index_ <= rhs.index_; }
        bool operator>(const HashMap32Iterator &rhs) { return index_ > rhs.index_; }
        bool operator>=(const HashMap32Iterator &rhs) { return index_ >= rhs.index_; }
        bool operator==(const HashMap32Iterator &rhs) { return index_ == rhs.index_; }
        bool operator!=(const HashMap32Iterator &rhs) { return index_ != rhs.index_; }
    };

    class HashMap32ConstIterator {
        friend class HashMap32<K, V, HashFunc, KeyEqual>;

        const HashMap32<K, V, HashFunc, KeyEqual> *container_;
        uint32_t index_;

        HashMap32ConstIterator(const HashMap32<K, V, HashFunc, KeyEqual> *container, uint32_t index)
            : container_(container), index_(index) {}

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Node;
        using difference_type = std::ptrdiff_t;
        using pointer = Node *;
        using reference = Node &;

        const Node &operator*() { return container_->at(index_); }
        const Node *operator->() { return &container_->at(index_); }
        HashMap32ConstIterator &operator++() {
            index_ = container_->NextOccupied(index_);
            return *this;
        }
        HashMap32ConstIterator operator++(int) {
            HashMap32ConstIterator tmp(*this);
            ++(*this);
            return tmp;
        }

        uint32_t index() const { return index_; }

        bool operator<(const HashMap32ConstIterator &rhs) { return index_ < rhs.index_; }
        bool operator<=(const HashMap32ConstIterator &rhs) { return index_ <= rhs.index_; }
        bool operator>(const HashMap32ConstIterator &rhs) { return index_ > rhs.index_; }
        bool operator>=(const HashMap32ConstIterator &rhs) { return index_ >= rhs.index_; }
        bool operator==(const HashMap32ConstIterator &rhs) { return index_ == rhs.index_; }
        bool operator!=(const HashMap32ConstIterator &rhs) { return index_ != rhs.index_; }
    };

    using iterator = HashMap32Iterator;
    using const_iterator = HashMap32ConstIterator;

    iterator begin() {
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (ctrl_[i] & OccupiedBit) {
                return iterator(this, i);
            }
        }
        return end();
    }
    const_iterator begin() const { return cbegin(); }

    const_iterator cbegin() const {
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (ctrl_[i] & OccupiedBit) {
                return const_iterator(this, i);
            }
        }
        return cend();
    }

    iterator end() { return iterator(this, capacity_); }
    const_iterator end() const { return const_iterator(this, capacity_); }
    const_iterator cend() const { return const_iterator(this, capacity_); }

    iterator iter_at(uint32_t i) { return iterator(this, i); }
    const_iterator citer_at(uint32_t i) const { return const_iterator(this, i); }

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
        nodes_[it.index_].val.~V();

        return iter_at(next);
    }

    uint32_t FindOccupiedInRange(const uint32_t start, const uint32_t end) {
        for (uint32_t i = start; i < end; ++i) {
            if (ctrl_[i] & OccupiedBit) {
                return i;
            }
        }
        return end;
    }

  private:
    void CheckRealloc() {
        if ((size_ + 1) > uint32_t(0.8f * capacity_)) {
            ReserveRealloc(capacity_ * 2);
        }
    }

    void ReserveRealloc(uint32_t desired_capacity) {
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
                    InsertInternal(old_nodes[i].hash, std::move(old_nodes[i].key), std::move(old_nodes[i].val));
                }
            }

            this->deallocate(old_ctrl, mem_size(old_capacity));
        }
    }

    V *InsertInternal(uint32_t hash, const K &key) {
        CheckRealloc();

        uint32_t i = hash & (capacity_ - 1);
        while (ctrl_[i] & OccupiedBit) {
            i = (i + 1) & (capacity_ - 1);
        }

        ++size_;
        ctrl_[i] = OccupiedBit | (hash & HashMask);
        nodes_[i].hash = hash;
        new (&nodes_[i].key) K(key);
        new (&nodes_[i].val) V;

        return &nodes_[i].val;
    }

    void InsertInternal(uint32_t hash, const K &key, const V &val) {
        CheckRealloc();

        uint32_t i = hash & (capacity_ - 1);
        while (ctrl_[i] & OccupiedBit) {
            i = (i + 1) & (capacity_ - 1);
        }

        ++size_;
        ctrl_[i] = OccupiedBit | (hash & HashMask);
        nodes_[i].hash = hash;
        new (&nodes_[i].key) K(key);
        new (&nodes_[i].val) V(val);
    }

    void InsertInternal(uint32_t hash, K &&key, V &&val) {
        CheckRealloc();

        uint32_t i = hash & (capacity_ - 1);
        while (ctrl_[i] & OccupiedBit) {
            i = (i + 1) & (capacity_ - 1);
        }

        ++size_;
        ctrl_[i] = OccupiedBit | (hash & HashMask);
        nodes_[i].hash = hash;
        new (&nodes_[i].key) K(std::forward<K>(key));
        new (&nodes_[i].val) V(std::forward<V>(val));
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
