// A fixed set of worker threads that outlive individual draw calls.
//
// Spawning threads per draw is fine when a frame is one big mesh, but a scene
// made of many small draws pays the thread-creation cost over and over -- and
// on a scene with a few dozen draw calls that cost is larger than the rendering
// itself. The pool is created once and parked on a condition variable between
// dispatches.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace sr {

class ThreadPool {
public:
    // Spawns `workerCount` threads; the dispatching thread participates too, so
    // the effective width is workerCount + 1.
    explicit ThreadPool(int workerCount) {
        workers_.reserve(static_cast<std::size_t>(workerCount > 0 ? workerCount : 0));
        for (int i = 0; i < workerCount; ++i) workers_.emplace_back([this] { workerLoop(); });
    }

    ~ThreadPool() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        startCv_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Width available to a dispatch, counting the calling thread.
    [[nodiscard]] int width() const noexcept { return static_cast<int>(workers_.size()) + 1; }

    // Invokes fn(0) .. fn(count-1) exactly once each and returns once they have
    // all finished. Index 0 always runs on the calling thread, so a count of 1
    // never touches the workers at all.
    void run(int count, const std::function<void(int)>& fn) {
        if (count <= 1 || workers_.empty()) {
            for (int i = 0; i < count; ++i) fn(i);
            return;
        }

        {
            const std::lock_guard<std::mutex> lock(mutex_);
            task_ = &fn;
            taskCount_ = count;
            nextIndex_.store(1, std::memory_order_relaxed);
            // Every worker decrements exactly once per generation, even the
            // ones that wake up to find nothing left to claim.
            busy_ = static_cast<int>(workers_.size());
            ++generation_;
        }
        startCv_.notify_all();

        // The caller takes index 0, then helps with whatever is unclaimed.
        fn(0);
        claimIndices(fn, count);

        std::unique_lock<std::mutex> lock(mutex_);
        doneCv_.wait(lock, [this] { return busy_ == 0; });
        task_ = nullptr;
    }

private:
    void claimIndices(const std::function<void(int)>& fn, int count) {
        for (int i = nextIndex_.fetch_add(1, std::memory_order_relaxed); i < count;
             i = nextIndex_.fetch_add(1, std::memory_order_relaxed)) {
            fn(i);
        }
    }

    void workerLoop() {
        std::uint64_t seenGeneration = 0;
        for (;;) {
            const std::function<void(int)>* task = nullptr;
            int count = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                startCv_.wait(lock, [this, seenGeneration] {
                    return stop_ || generation_ != seenGeneration;
                });
                if (stop_) return;
                seenGeneration = generation_;
                task = task_;
                count = taskCount_;
            }

            claimIndices(*task, count);

            {
                const std::lock_guard<std::mutex> lock(mutex_);
                if (--busy_ == 0) doneCv_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable startCv_;
    std::condition_variable doneCv_;

    const std::function<void(int)>* task_ = nullptr;
    std::atomic<int> nextIndex_{0};
    int taskCount_ = 0;
    int busy_ = 0;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
};

}  // namespace sr
