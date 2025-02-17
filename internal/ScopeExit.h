#pragma once

#include <utility>

namespace Ray {
template <typename F> class AtScopeExit {
    F func_;
    bool engaged_ = false;

  public:
    explicit AtScopeExit(F func) : func_(std::move(func)), engaged_(true) {}
    AtScopeExit(AtScopeExit &&rhs) : func_(std::move(rhs.func_)), engaged_(rhs.engaged_) { rhs.Dismiss(); }
    ~AtScopeExit() { Execute(); }

    AtScopeExit(const AtScopeExit &) = delete;
    AtScopeExit &operator=(const AtScopeExit &) = delete;
    AtScopeExit &operator=(AtScopeExit &&) = delete;

    inline void Dismiss() { engaged_ = false; }

    inline void Execute() {
        if (engaged_) {
            func_();
        }
        Dismiss();
    }
};
template <typename F> AtScopeExit<F> make_scope_exit(F &&f) { return AtScopeExit<F>(std::forward<F>(f)); }
} // namespace Ray

#define SCOPE_EXIT_INTERNAL2(aname, ...) auto aname = Ray::make_scope_exit([&]() { __VA_ARGS__; });

#define SCOPE_EXIT_CONCAT(x, y) SCOPE_EXIT_##x##y

#define SCOPE_EXIT_INTERNAL1(ctr, ...) SCOPE_EXIT_INTERNAL2(SCOPE_EXIT_CONCAT(instance_, ctr), __VA_ARGS__)

#define SCOPE_EXIT(...) SCOPE_EXIT_INTERNAL1(__COUNTER__, __VA_ARGS__)