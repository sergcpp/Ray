#pragma once

#include <cstdarg>
#include <cstdio>

namespace Ray {
class ILog {
  public:
    virtual ~ILog() = default;

    virtual void Info(const char *fmt, ...) = 0;
    virtual void Warning(const char *fmt, ...) = 0;
    virtual void Error(const char *fmt, ...) = 0;
};

class LogNull final : public ILog {
  public:
    void Info(const char *fmt, ...) override {}
    void Warning(const char *fmt, ...) override {}
    void Error(const char *fmt, ...) override {}
};

class LogStdout final : public Ray::ILog {
  public:
    void Info(const char *fmt, ...) override {
        va_list vl;
        va_start(vl, fmt);
        vprintf(fmt, vl);
        va_end(vl);
        putc('\n', stdout);
    }
    void Warning(const char *fmt, ...) override {
        va_list vl;
        va_start(vl, fmt);
        vprintf(fmt, vl);
        va_end(vl);
        putc('\n', stdout);
    }
    void Error(const char *fmt, ...) override {
        va_list vl;
        va_start(vl, fmt);
        vprintf(fmt, vl);
        va_end(vl);
        putc('\n', stdout);
    }
};
} // namespace Ray