#include "Q.h"

void Q::Enqueue(TFunction<void()> v)
{
	FScopeLock l(&mutex_);
	queue.Enqueue(MoveTemp(v));
}

bool Q::ProcessQ()
{
	TFunction<void()> v;
	{
		FScopeLock l(&mutex_);
		if (!queue.Dequeue(v))
		{
			return false;
		}
	}
	v();
	return true;
}
