#pragma once

#include <cstdarg>
#include <cstdio>

#if defined(_MSC_VER)
#include <sal.h>
#define CHECK_FORMAT_STRING(format_string_index_one_based, vargs_index_one_based)
#else
#define _Printf_format_string_
#define CHECK_FORMAT_STRING(format_string_index_one_based, vargs_index_one_based)                                      \
    __attribute__((format(printf, format_string_index_one_based, vargs_index_one_based)))
#endif

namespace Ray {
class ILog {
  public:
    virtual ~ILog() = default;

    virtual void Info(_Printf_format_string_ const char *fmt, ...) CHECK_FORMAT_STRING(2, 3) = 0;
    virtual void Warning(_Printf_format_string_ const char *fmt, ...) CHECK_FORMAT_STRING(2, 3) = 0;
    virtual void Error(_Printf_format_string_ const char *fmt, ...) CHECK_FORMAT_STRING(2, 3) = 0;
};

class LogNull final : public ILog {
  public:
    void CHECK_FORMAT_STRING(2, 3) Info(_Printf_format_string_ const char *fmt, ...) override {}
    void CHECK_FORMAT_STRING(2, 3) Warning(_Printf_format_string_ const char *fmt, ...) override {}
    void CHECK_FORMAT_STRING(2, 3) Error(_Printf_format_string_ const char *fmt, ...) override {}
};

class LogStdout final : public Ray::ILog {
  public:
    void CHECK_FORMAT_STRING(2, 3) Info(_Printf_format_string_ const char *fmt, ...) override {
        va_list vl;
        va_start(vl, fmt);
        vprintf(fmt, vl);
        va_end(vl);
        putc('\n', stdout);
    }
    void CHECK_FORMAT_STRING(2, 3) Warning(_Printf_format_string_ const char *fmt, ...) override {
        va_list vl;
        va_start(vl, fmt);
        vprintf(fmt, vl);
        va_end(vl);
        putc('\n', stdout);
    }
    void CHECK_FORMAT_STRING(2, 3) Error(_Printf_format_string_ const char *fmt, ...) override {
        va_list vl;
        va_start(vl, fmt);
        vprintf(fmt, vl);
        va_end(vl);
        putc('\n', stdout);
    }
};
} // namespace Ray

#undef CHECK_FORMAT_STRING
#if !defined(_MSC_VER)
#undef _Printf_format_string_
#endif