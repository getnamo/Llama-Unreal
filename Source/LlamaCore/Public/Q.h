#pragma once

#include <deque>
#include <functional>
#include <mutex>

class Q {
public:
	void Enqueue(TFunction<void()>);
	bool ProcessQ();

private:
	TQueue<TFunction<void()>> queue;
	FCriticalSection mutex_;
};
