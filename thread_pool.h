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
		// �����߳�
		for (unsigned int i = 0; i < n_thread; i++)
		{
			// ����������Ϊ�����̵߳Ĳ���
			pool.emplace_back([this]() {
				while(this->run)
				{
					Task task;
					{
						std::unique_lock<std::mutex> lock_temp(this->look);
						// wait ֮ǰ�ж����������ķ���ֵ��Ϊtrueʱ��wait��Ϊfalseʱwait
						// wait �����У����Ѿ��ͷţ���Ҫ notify �Ż��˳� wait�����¼���
						this->cv.wait(lock_temp, [this]() { return !this->tasks.empty() || !this->run; });
						// ���ò����������Ϊ��ʱ���˳��߳�
						if (!this->run && this->tasks.empty()) return;

						task = std::move(this->tasks.front());
						this->tasks.pop();
					}
					// ִ��
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
		// wake ��������wait���߳�
		this->cv.notify_all();
		for (std::thread & t : this->pool) t.join();
	}

	inline int get_idle_num() { return idle_num; }

	template<class F, class... Args>
	auto commit(F &&f, Args &&... args)->std::future<decltype(f(args...))>
	{
		// �ύ���񣬷��� std::future
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
	// �̳߳�
	std::vector<std::thread> pool;
	// �������
	std::queue<Task> tasks;
	// ͬ��
	std::mutex look;
	std::condition_variable cv;
	// ״̬
	std::atomic_bool run;
	std::atomic_uint idle_num;
};