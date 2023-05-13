#pragma once

namespace Ray {
template <class F> class AtScopeExit {
    F &func_;

  public:
    explicit AtScopeExit(F &func) : func_(func) {}
    ~AtScopeExit() { func_(); }
};
}

#define SCOPE_EXIT_INTERNAL2(lname, aname, ...) \
    auto lname = [&]() { __VA_ARGS__; }; \
    Ray::AtScopeExit<decltype(lname)> aname(lname);

#define SCOPE_EXIT_CONCAT(x, y) SCOPE_EXIT_ ## x ## y

#define SCOPE_EXIT_INTERNAL1(ctr, ...) \
    SCOPE_EXIT_INTERNAL2(SCOPE_EXIT_CONCAT(func_, ctr), \
                         SCOPE_EXIT_CONCAT(instance_, ctr), __VA_ARGS__)

#define SCOPE_EXIT(...) SCOPE_EXIT_INTERNAL1(__COUNTER__, __VA_ARGS__)