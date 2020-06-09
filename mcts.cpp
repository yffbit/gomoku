#include <math.h>
#include <float.h>
#include <numeric>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "mcts.h"

TreeNode::TreeNode(uint32_t action_dim) : 
        parent(nullptr),
        children(action_dim, nullptr),
        leaf(true),
        N(0),
        W(0),
        Q(0),
        P(0) {}

TreeNode::TreeNode(TreeNode *parent, double P, uint32_t action_dim) : 
        parent(parent),
        children(action_dim, nullptr),
        leaf(true),
        N(0),
        W(0),
        Q(0),
        P(P) {}

TreeNode::TreeNode(const TreeNode &node)
{
    // 拷贝构造函数
    this->copy(node, node.parent);
}

TreeNode & TreeNode::operator=(const TreeNode &node)
{
    // 赋值构造函数
    if (this == &node) return *this;
    this->copy(node, node.parent);
    return *this;
}

void TreeNode::copy(const TreeNode &node, TreeNode *parent)
{
    this->parent = parent;
    this->leaf = node.leaf;
#ifndef SINGLE_THREAD
	this->N.store(node.N.load());
	this->W.store(node.W.load());
	this->Q.store(node.Q.load());
#else
	this->N = node.N;
	this->W = node.W;
	this->Q = node.Q;
#endif // !SINGLE_THREAD
    this->P = node.P;
    this->children = node.children; // 浅拷贝
    uint32_t action_dim = node.children.size();
    for (int i = 0; i < action_dim; i++)
    {
        if (node.children[i])
        {
            this->children[i] = new TreeNode(action_dim);
            this->children[i]->copy(*node.children[i], this);
        }
    }
}

TreeNode::~TreeNode()
{
    uint32_t i, size = this->children.size();
	TreeNode *node = nullptr;
    for (i = 0; i < size; i++)
    {
		node = this->children[i];
        if (node)
        {
            node->parent = nullptr;
            this->children[i] = nullptr;
            delete node;
			node = nullptr;
        }
    }
    if (this->parent)
    {
		size = this->parent->children.size();
        for (i = 0; i < size; i++)
        {
            if (this->parent->children[i] == this)
            {
                this->parent->children[i] = nullptr;
                break;
            }
        }
        this->parent = nullptr;
    }
}

uint32_t TreeNode::select(double c_puct, double virtual_loss)
{
    double best_value = -DBL_MAX, value;
    int best_act = -1;
#ifndef SINGLE_THREAD
    uint32_t sum_N = this->N.load();
#else
	uint32_t sum_N = this->N;
#endif // !SINGLE_THREAD
    for (uint32_t i = 0; i < this->children.size(); i++)
    {
        if (nullptr == this->children[i]) continue;
        value = this->children[i]->get_value(c_puct, sum_N);
        if (value > best_value)
        {
            best_value = value;
            best_act = i;
        }
    }
#ifndef SINGLE_THREAD
    // 添加虚拟损失
    if (best_act >= 0)
    {
		TreeNode * node = this->children[best_act];
        node->N += (int)virtual_loss;
        node->W = node->W - virtual_loss;
		node->Q = node->W / node->N;
    }
#endif // !SINGLE_THREAD
    return best_act;
}

double TreeNode::get_value(double c_puct, uint32_t sum_N) const
{
#ifndef SINGLE_THREAD
	return this->Q.load() + c_puct * this->P * sqrt(sum_N) / (1 + this->N.load());
#else
	return this->Q + c_puct * this->P * sqrt(sum_N) / (1 + this->N);
#endif // !SINGLE_THREAD
}

bool TreeNode::expand(const at::Tensor &prior, const std::vector<bool> &legal_action)
{
#ifndef SINGLE_THREAD
    std::lock_guard<std::mutex> temp_lock(this->lock);
#endif // !SINGLE_THREAD
    if (this->leaf)
    {
        uint32_t action_dim = this->children.size(), i = 0;
		for (i = 0; i < action_dim; i++)
		{
			if (!legal_action[i]) continue;
			this->children[i] = new TreeNode(this, prior[i].item().toDouble(), action_dim);
		}
        this->leaf = false;
		return true;
    }
	else return false;
}

void TreeNode::backup(double value, double virtual_loss, bool success)
{
    if (this->parent) this->parent->backup(-value, virtual_loss, success);
	else
	{
		// 根节点
		this->N += 1;
		return;
	}
	// 非根节点
#ifndef SINGLE_THREAD
	if (success)
	{
		// 移除虚拟损失  更新 W N Q
		this->N = this->N - (int)virtual_loss + 1;
		this->W = this->W + virtual_loss + value;
	}
	else
	{
		// 移除虚拟损失  恢复原值
		this->N = this->N - (int)virtual_loss;
		this->W = this->W + virtual_loss;
	}
#else
	this->N += 1;
	this->W += value;
#endif // !SINGLE_THREAD
    this->Q = this->W / this->N;
}

// MCTS
MCTS::MCTS(PolicyValueNet *network, uint32_t n_thread, double c_puct, double temp,
           uint32_t n_simulate, double virtual_loss, uint32_t action_dim, bool add_noise) :
        network(network),
#ifndef SINGLE_THREAD
        thread_pool(new ThreadPool(n_thread)),
#endif // !SINGLE_THREAD
        c_puct(c_puct),
		n_simulate(n_simulate),
        virtual_loss(virtual_loss),
        temp(temp),
		add_noise(add_noise),
        action_dim(action_dim),
		n_count(0),
		is_self_play(false),
        root(new TreeNode(nullptr, 1., action_dim))
{
	srand(time(0));
	torch::set_num_threads(n_thread);
}

void MCTS::update_with_move(int last_move)
{
	TreeNode *root = this->root.get();
	if (this->is_self_play && last_move >= 0 && last_move < root->children.size() && root->children[last_move] != nullptr)
	{
		// 利用子树 孩子节点作为根节点
		TreeNode *node = root->children[last_move];
		root->children[last_move] = nullptr;
        node->parent = nullptr;
        this->root.reset(node);
    }
    else this->root.reset(new TreeNode(nullptr, 1., this->action_dim));
    this->n_count++;
}

uint32_t binary_search(std::vector<double> &values, double target)
{
    uint32_t i = 0, j = values.size() - 1, m;
	// 左开右闭
	// (0,v[0]] (v[0],v[1]] (v[1],v[2]] ... (v[n-2],v[n-1]]  v[n-1]=1
	// 找到 target 落在哪个区间
	// 重复元素应该取第一个
    while (i < j)
    {
        m = (i + j) >> 1;
        if (values[m] >= target) j = m;
        else i = m + 1;
    }
    return i;
}

uint32_t MCTS::get_action(Gomoku *gomoku, bool explore)
{
	return this->get_action(this->get_action_prob(gomoku), explore);
}

uint32_t MCTS::get_action(std::vector<double> action_prob, bool explore)
{
    uint32_t n = action_prob.size(), i = 0;
	// srand(time(0));
    /*if (explore)
    {
		double sum = 0;
        // 添加狄利克雷噪声
		at::Tensor noise = this->network->dirichlet_noise(n, 0.3);
		for (i = 0; i < n; i++)
		{
			if (nullptr == this->root->children[i]) action_prob[i] = 0;
			else action_prob[i] = 0.25 * noise[i].item().toDouble() + 0.75 * action_prob[i];
			sum += action_prob[i];
		}
		std::for_each(action_prob.begin(), action_prob.end(), [sum](double &x) { x /= sum; });
    }*/
    // 按权重随机选择
    for (i = 1; i < n; i++) action_prob[i] += action_prob[i-1];
	double p;
	uint32_t count = 0;
	while (true)
	{
		p = (double)(rand() % 1000) / 1000;
		// 二分查找
		i = binary_search(action_prob, p);
		if ((++count) > 2) std::printf("binary search count : %d\n", count);
		if (this->root->children[i]) break;
	}
    return i;
}

std::vector<double> MCTS::get_action_prob(Gomoku *gomoku)
{
    uint32_t i;
	// 根节点还未扩展，先扩展根节点
	if (this->root->leaf) this->simulate(gomoku);
#ifndef SINGLE_THREAD
	std::vector<std::future<void>> futures;
    for (i = 0; i < this->n_simulate; i++)
    {
        // 提交模拟任务到线程池
        auto future = this->thread_pool->commit(std::bind(&MCTS::simulate, this, gomoku));
        futures.emplace_back(std::move(future));
    }
    // 等待模拟结束
    for (i = 0; i < this->n_simulate; i++) futures[i].wait();
#else
	for (i = 0; i < this->n_simulate; i++) this->simulate(gomoku);
#endif // !SINGLE_THREAD

    std::vector<double> action_prob(this->action_dim, 0);
    std::vector<TreeNode *> & children = this->root->children;
    double sum = 0;
    uint32_t n, max_n = 0, size = children.size();
    for (i = 0; i < size; i++)
    {
        if (children[i])
        {
#ifndef SINGLE_THREAD
            n = children[i]->N.load();
#else
			n = children[i]->N;
#endif // !SINGLE_THREAD
            action_prob[i] = n;
            sum += n;
            max_n = n > max_n ? n : max_n;
        }
    }
    if (this->temp > 0 && this->temp <= 1e-3 + FLT_EPSILON)
	{
		// 选取次数最多的
		sum = 0;
		for (i = 0; i < action_prob.size(); i++)
		{
			if (abs(action_prob[i] - max_n) <= FLT_EPSILON) action_prob[i] = 1;
			else action_prob[i] = 0;
			sum += action_prob[i];
		}
	}
	else if (abs(this->temp - 1) > FLT_EPSILON)
	{
		sum = 0;
		for (i = 0; i < action_prob.size(); i++)
		{
			action_prob[i] = pow(action_prob[i], 1 / this->temp);
            sum += action_prob[i];
        }
    }
	if (sum <= FLT_EPSILON) std::cout << sum << std::endl;
    // 归一化
    std::for_each(action_prob.begin(), action_prob.end(), [sum](double &x) { x /= sum; });
	return action_prob;
}

void MCTS::simulate(Gomoku *gomoku)
{
    // 单次模拟
	// 模拟是否成功
	bool success = false;
	TreeNode *node = nullptr, *root = this->root.get();
	uint32_t action = 0, count = 0;
	double value = 0;
	while (!success)
	{
		// 复制游戏状态
		Gomoku game = *gomoku;
		node = root;
		while (!node->leaf)
		{
			action = node->select(this->c_puct, this->virtual_loss);
			game.execute_move(action);
			node = node->children[action];
		}
		// 游戏是否结束
		std::vector<int> res = game.get_game_status();
		if (res[0] == 0)
		{
			// 未结束 扩展 神经网络评估
			at::Tensor s = game.curr_state(true, this->network->device);
			// 输出包含batch维度
			std::vector<at::Tensor> pred = this->network->predict(s);

			value = pred[1][0].item().toDouble();
			// std::cout << value << std::endl;
			// std::cout << pred[0][0] << std::endl;

			std::vector<bool> legal_move = game.get_legal_move();
			// 扩展
			at::Tensor prior = pred[0][0];
			if (this->add_noise)
			{
				// 添加狄利克雷噪声
				prior = 0.75 * prior + 0.25 * this->network->dirichlet_noise(game.get_action_dim(), 0.3);
			}
			success = node->expand(prior, legal_move);
		}
		else
		{
			// 游戏结束 实际价值（以当前玩家为视角）
			int winner = res[1];
			value = winner == 0 ? 0 : (winner == game.get_curr_player() ? 1 : -1);
			success = true;
		}
		// 当前状态的前一步动作为对手方落子，价值取反
		if (node != root) node->backup(-value, this->virtual_loss, success);
		if ((++count) > 1) std::printf("simulation count : %d\n", count);
	}
}

int MCTS::self_play(Gomoku *gomoku, std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values,
	double temp, uint32_t n_round, bool add_noise, bool show)
{
    std::vector<int> res(2, 0);
	uint32_t move;
    int idx;
    gomoku->reset();
	// 起始温度参数
	this->temp = temp;
	this->add_noise = add_noise;
	this->is_self_play = true;
	if (show)
	{
		std::cout << "New game." << std::endl;
		gomoku->display();
	}
	std::vector<double> action_prob;
	std::vector<int> players;
	at::Tensor state;
	uint32_t s0 = states.size();
	if (probs.size() != s0 || values.size() != s0)
	{
		states.clear();
		probs.clear();
		values.clear();
		s0 = 0;
	}
	while (0 == res[0])
	{
		idx = gomoku->get_curr_player();
		// 训练数据缓存不用CUDA
		state = gomoku->curr_state(false, this->network->device);
		action_prob = this->get_action_prob(gomoku);
		move = this->get_action(action_prob);
        if (show)
        {
			std::printf("Player '%c' : %d %d\n", gomoku->get_symbol(idx), move / gomoku->get_n(), move % gomoku->get_n());
        }
		if (gomoku->execute_move(move))
		{
			this->update_with_move(move);
			states.emplace_back(state);
			probs.emplace_back(torch::tensor(action_prob));
			players.emplace_back(idx);
			res = gomoku->get_game_status();
			if (show) gomoku->display();
			if ((this->n_count >> 1) >= n_round) this->temp = 1e-3;
		}
	}
	this->update_with_move(-1);
    if (show)
    {
        if (0 != res[1]) std::printf("Game end. Winner is Player '%c'.\n", gomoku->get_symbol(res[1]));
	    else std::cout << "Game end. Tie." << std::endl;
    }
	int i, z;
	for (i = s0; i < states.size(); i++)
	{
		z = res[1] == 0 ? 0 : (players[i] == res[1] ? 1 : -1);
		values.emplace_back(z);
	}
	return res[1];
}