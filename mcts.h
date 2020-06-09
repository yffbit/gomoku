#pragma once

//#define SINGLE_THREAD

#ifndef SINGLE_THREAD
#include "thread_pool.h"
#include <thread>
#include <atomic>
#endif // !SINGLE_THREAD
#include <vector>
#include <iostream>
#include "gomoku.h"
#include "policy_value_net.h"

class TreeNode
{
public:
    friend class MCTS; // 友元类，可以访问TreeNode的私有成员

    TreeNode(uint32_t action_dim);
    TreeNode(TreeNode *parent, double P, uint32_t action_dim);
    TreeNode(const TreeNode &node);
    ~TreeNode();
    TreeNode &operator=(const TreeNode &node);
    void copy(const TreeNode &node, TreeNode *parent=nullptr);
    uint32_t select(double c_puct, double virtual_loss);
    double get_value(double c_puct, uint32_t sum_N) const;
    bool expand(const at::Tensor &prior, const std::vector<bool> &legal_action);
	void backup(double value, double virtual_loss, bool success);
    inline bool is_leaf() const { return this->leaf; }

private:
    TreeNode *parent;
    std::vector<TreeNode *> children;
    bool leaf;	// 是否为叶子节点
#ifndef SINGLE_THREAD
    std::mutex lock; // 扩展时加锁
    std::atomic<int> N;
    std::atomic<double> W;
    std::atomic<double> Q;
#else
	int N;
	double W;
	double Q;
#endif // !SINGLE_THREAD
    double P;
};

class MCTS : public Player
{
public:
    MCTS(PolicyValueNet *network, uint32_t n_thread, double c_puct, double temp,
        uint32_t n_simulate, double virtual_loss, uint32_t action_dim, bool add_noise);
    uint32_t get_action(Gomoku *gomoku, bool explore = false);
	uint32_t get_action(std::vector<double> action_probs, bool explore = false);
    std::vector<double> get_action_prob(Gomoku *gomoku);
	inline void init() { this->is_self_play = false; }
    void update_with_move(int last_move);
    inline void set_temp(double temp = 1e-3) { this->temp = temp; }
	int self_play(Gomoku *gomoku, std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values,
				double temp = 1, uint32_t n_round = 20, bool add_noise = true, bool show = false);
private:
    void simulate(Gomoku * gomoku);
    std::unique_ptr<TreeNode> root;
#ifndef SINGLE_THREAD
    std::unique_ptr<ThreadPool> thread_pool;
#endif // !SINGLE_THREAD
    PolicyValueNet *network;

    uint32_t action_dim;
    uint32_t n_simulate;
    double c_puct;
    double virtual_loss;
    uint32_t n_count;   // 落子计数
    double temp;    // 温度参数
	bool add_noise;	// 扩展时是否添加噪声
	bool is_self_play;	// 游戏模式
};
