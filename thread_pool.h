#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>

class ThreadPool
{
public:
	using Task = std::function<void ()>;

	inline ThreadPool(unsigned int n_thread = 4) : run(true)
	{
		if (n_thread < 1) n_thread = 1;
		idle_num = n_thread;
		// 创建线程
		for (unsigned int i = 0; i < n_thread; i++)
		{
			// 匿名函数作为创建线程的参数
			pool.emplace_back([this]() {
				while(this->run)
				{
					Task task;
					{
						std::unique_lock<std::mutex> lock_temp(this->look);
						// wait 之前判断匿名函数的返回值，为true时不wait，为false时wait
						// wait 过程中（锁已经释放）需要 notify 才会退出 wait，重新加锁
						this->cv.wait(lock_temp, [this]() { return !this->tasks.empty() || !this->run; });
						// 禁用并且任务队列为空时，退出线程
						if (!this->run && this->tasks.empty()) return;

						task = std::move(this->tasks.front());
						this->tasks.pop();
					}
					// 执行
					this->idle_num--;
					task();
					this->idle_num++;
				}
			});
		}
	}

	inline ~ThreadPool()
	{
		this->run = false;
		// wake 所有正在wait的线程
		this->cv.notify_all();
		for (std::thread & t : this->pool) t.join();
	}

	inline int get_idle_num() { return idle_num; }

	template<class F, class... Args>
	auto commit(F &&f, Args &&... args)->std::future<decltype(f(args...))>
	{
		// 提交任务，返回 std::future
		if (!this->run) throw std::runtime_error("commit error : ThreadPool is stopped.");
		// using return_type = typename std::result_of<F(Args...)>::type;
		using return_type = decltype(f(args...));
		// packaged_task package the bind function and future
		auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

		{
			std::lock_guard<std::mutex> lock_temp(this->look);
			this->tasks.emplace([task_ptr]() { (*task_ptr)(); });
		}

		this->cv.notify_one();
		return task_ptr->get_future();
	}

private:
	// 线程池
	std::vector<std::thread> pool;
	// 任务队列
	std::queue<Task> tasks;
	// 同步
	std::mutex look;
	std::condition_variable cv;
	// 状态
	std::atomic_bool run;
	std::atomic_uint idle_num;
};