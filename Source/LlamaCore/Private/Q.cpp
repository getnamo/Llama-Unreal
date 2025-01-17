#include "Q.h"

void Q::Enqueue(std::function<void()> func) {
	std::lock_guard<std::mutex> lock(mutex_);
	queue.push_back(std::move(func));
}

bool Q::ProcessQ() {
	std::function<void()> func;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue.empty()) {
			return false;
		}
		func = std::move(queue.front());
		queue.pop_front();
	}
	func();
	return true;
}
