#include <iostream>
#include <windows.h>
#include "thread_pool.h"

int func(unsigned int n)
{
	Sleep(1000 * n);
	return n;
}

int main()
{
	ThreadPool thread_pool(4);

	auto r1 = thread_pool.commit(func, 1);
	auto r2 = thread_pool.commit(func, 2);
	auto r3 = thread_pool.commit(func, 3);
	auto r4 = thread_pool.commit(func, 4);
	auto r5 = thread_pool.commit(func, 5);
	auto r6 = thread_pool.commit(func, 10);
	auto r7 = thread_pool.commit(func, 20);
	auto r8 = thread_pool.commit(func, 30);

	std::cout << r1.get() << std::endl;
	std::cout << r2.get() << std::endl;
	std::cout << r3.get() << std::endl;
	std::cout << r4.get() << std::endl;
	std::cout << r5.get() << std::endl;
	std::cout << r6.get() << std::endl;
	std::cout << r7.get() << std::endl;
	std::cout << r8.get() << std::endl;
	std::cout << "END" << std::endl;

	return 0;
}