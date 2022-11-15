#pragma once

namespace Ray {
class ILog {
  public:
    virtual ~ILog() = default;

    virtual void Info(const char *fmt, ...) = 0;
    virtual void Warning(const char *fmt, ...) = 0;
    virtual void Error(const char *fmt, ...) = 0;
};

class LogNull : public ILog {
  public:
    virtual void Info(const char *fmt, ...) override {}
    virtual void Warning(const char *fmt, ...) override {}
    virtual void Error(const char *fmt, ...) override {}
};
} // namespace Ray