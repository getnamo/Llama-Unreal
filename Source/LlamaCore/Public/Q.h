#pragma once

#include <deque>
#include <functional>
#include <mutex>

class Q {
    public:
        void Enqueue(std::function<void()> func);
        bool ProcessQ();

    private:
        std::deque<std::function<void()>> queue;
        std::mutex mutex_;
};
