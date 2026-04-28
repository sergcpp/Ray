#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include <new>
#include <stdexcept>

namespace Ray {
inline void *aligned_malloc(size_t size, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1)));
    while (alignment < sizeof(void *)) {
        alignment *= 2;
    }
    if (size > static_cast<size_t>(PTRDIFF_MAX) - (alignment - 1) - sizeof(void *)) {
        return nullptr;
    }
    size_t space = size + (alignment - 1) + sizeof(void *);

    void *ptr = malloc(space);
    if (!ptr) {
        return nullptr;
    }
    void *original_ptr = ptr;

    char *ptr_bytes = static_cast<char *>(ptr);
    ptr_bytes += sizeof(void *);

    auto off = static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr_bytes) % alignment);
    if (off) {
        off = alignment - off;
    }
    ptr_bytes += off;

    ptr = static_cast<void *>(ptr_bytes);
    ptr_bytes -= sizeof(void *);

    memcpy(ptr_bytes, &original_ptr, sizeof(void *));

    return ptr;
}

inline void aligned_free(void *p) {
    if (p) {
        free(static_cast<void **>(p)[-1]);
    }
}

template <typename T, size_t Alignment> class aligned_allocator {
  public:
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    size_t max_size() const {
        return (static_cast<size_t>(0) - static_cast<size_t>(1)) / sizeof(T);
    }

    template <typename U> struct rebind {
        typedef aligned_allocator<U, Alignment> other;
    };

    bool operator!=(const aligned_allocator &other) const { return !(*this == other); }

    template <class U, class... Args> void construct(U *p, Args &&...args) {
        ::new ((void *)p) U(std::forward<Args>(args)...);
    }

    template <class U> void destroy(U *p) { p->~U(); }

    bool operator==(const aligned_allocator &other) const {
        (void)other;
        return true;
    }

    aligned_allocator() = default;

    aligned_allocator(const aligned_allocator &) = default;

    template <typename U> explicit aligned_allocator(const aligned_allocator<U, Alignment> &) {}

    ~aligned_allocator() = default;

    T *allocate(const size_t n) const {
        if (n == 0) {
            return nullptr;
        }
        if (n > max_size()) {
            throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
        }
        void *const pv = aligned_malloc(n * sizeof(T), Alignment);
        if (pv == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(pv);
    }

    void deallocate(T *const p, const size_t) const {
        aligned_free(p);
    }

    aligned_allocator &operator=(const aligned_allocator &) = delete;
};
} // namespace Ray
