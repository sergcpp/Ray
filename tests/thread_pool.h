#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

class ThreadPool {
  public:
    explicit ThreadPool(size_t threads_count);
    ~ThreadPool();

    template <class F, class... Args>
    std::future<typename std::result_of<F(Args...)>::type> Enqueue(F &&f, Args &&... args);

    template <class UnaryFunction> void ParallelFor(int from, int to, UnaryFunction &&f);

    size_t workers_count() const { return workers_.size(); }

  private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers_;
    // the task queue
    std::queue<std::function<void()>> tasks_;

    // synchronization
    std::mutex q_mtx_;
    std::condition_variable condition_;
    bool stop_;
};

// the constructor just launches some amount of workers_
inline ThreadPool::ThreadPool(const size_t threads_count) : stop_(false) {
    for (size_t i = 0; i < threads_count; ++i)
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(q_mtx_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                task();
            }
        });
}

// add new work item to the pool
template <class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::Enqueue(F &&f, Args &&... args) {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task =
        std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(q_mtx_);

        // don't allow enqueueing after stopping the pool
        if (stop_) {
            throw std::runtime_error("Enqueue on stopped ThreadPool");
        }

        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();

    return res;
}

template <class UnaryFunction> inline void ThreadPool::ParallelFor(const int from, const int to, UnaryFunction &&f) {
    std::vector<std::future<void>> futures(to - from);

    for (int i = from; i < to; ++i) {
        futures[i - from] = Enqueue(f, i);
    }

    for (int i = 0; i < (to - from); ++i) {
        futures[i].wait();
    }
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(q_mtx_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_) {
        worker.join();
    }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
