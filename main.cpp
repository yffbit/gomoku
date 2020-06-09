#include <deque>
#include <ctime>
#include "mcts.h"

const char *model_path = "./model-checkpoint.pt";
const char *best_path = "./model-best.pt";

bool file_exists(const char * file)
{
	if (nullptr == file) return false;
	FILE *fp = fopen(file, "rb");
	if (nullptr != fp)
	{
		fclose(fp);
		fp = nullptr;
		return true;
	}
	return false;
}

class TimeCounter
{
public:
	inline void start() { this->s = clock(); }
	inline clock_t end() { this->e = clock(); return this->e - this->s; }
	inline double end_s() { return (double)this->end() / CLOCKS_PER_SEC; }
private:
	clock_t s, e;
};

class Train
{
public:
	Train(uint32_t size=8, uint32_t n_in_line=5, uint32_t state_c=5, uint32_t n_thread=6, double lr=1e-3, double c_lr=1, double temp=1, uint32_t n_simulate=500,
		uint32_t c_puct=5, double virtual_loss=3, uint32_t buffer_size=10000, uint32_t batch_size=256, uint32_t epochs=20, double kl_targ=0.02, uint32_t check_freq=10, uint32_t n_game=10000) :
		gomoku(size, n_in_line), network(best_path, true, state_c, size, size*size), mcts(&network, n_thread, c_puct, temp, n_simulate, virtual_loss, size*size, true),
		state_c(state_c), n_thread(n_thread), c_puct(c_puct), virtual_loss(virtual_loss), temp(temp), n_simulate(n_simulate),
		N(buffer_size), lr(lr), c_lr(c_lr), batch_size(batch_size), epochs(epochs), kl_targ(kl_targ), check_freq(check_freq), n_game(n_game),
		optimizer(network.model->parameters(), torch::optim::AdamOptions(1e-3).weight_decay(1e-4))
	{
		this->states = torch::zeros({ 0,state_c,size,size });
		this->probs = torch::zeros({ 0,size,size });
		this->values = torch::zeros({ 0,1 });
	}
	// 扩充数据
	void augment_data(std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values);
	void push(const at::Tensor &s, const at::Tensor &p, const at::Tensor &z);
	// 评估
	double evaluate(const char *best_path, uint32_t num);
	void run(const char *model_path, const char *best_path);
	std::vector<double> train_step(const std::vector<at::Tensor> &state, const std::vector<at::Tensor> &prob, const std::vector<at::Tensor> &value, const double &lr);
	std::vector<double> train_step(const at::Tensor &state, const at::Tensor &prob, const at::Tensor &value, const double &lr);
private:
	Gomoku gomoku;
	uint32_t state_c;
	uint32_t n_thread;
	uint32_t c_puct;
	double temp;
	double virtual_loss;
	uint32_t n_simulate;
	PolicyValueNet network;
	MCTS mcts;
	double lr;	// 初始学习速率
	double c_lr;// 学习速率乘数
	uint32_t batch_size;// 每步训练的数据量
	uint32_t epochs;	// 训练多少步
	double kl_targ;		// kl_loss 目标（控制训练速率）
	uint32_t check_freq;// 每隔多少局游戏进行评估
	uint32_t n_game;	// 自我对弈多少局游戏
	//std::deque<at::Tensor> states;
	//std::deque<at::Tensor> probs;
	//std::deque<at::Tensor> values;
	at::Tensor states;
	at::Tensor probs;
	at::Tensor values;
	uint32_t N;	// 容量
	torch::optim::Adam optimizer;
};

void Train::augment_data(std::vector<at::Tensor> &states, std::vector<at::Tensor> &probs, std::vector<float> &values)
{
	uint32_t i, j, action_dim, state_h, size = states.size();
	if (0 == size || probs.size() != size || values.size() != size) return;
	// [batch channels height width]
	state_h = states[0].size(2);
	action_dim = probs[0].size(0);
	// action_dim = state_h * state_h;
	//at::Tensor s, p, z;
	//for (i = 0; i < size; i++)
	//{
	//	s = states[i];
	//	p = probs[i].reshape({ state_h,state_h });
	//	z = torch::tensor({ {values[i]} });
	//	for (j = 0; j < 4; j++)
	//	{
	//		this->push(s, p.reshape({ 1,action_dim }), z);
	//		// 上下翻转
	//		this->push(s.flip(2), p.flip(0).reshape({ 1,action_dim }), z);
	//		if (j == 3) break;
	//		// 旋转90度
	//		s = s.rot90(1, { 2,3 });
	//		p = p.rot90(1, { 0,1 });
	//	}
	//}

	uint32_t size0 = this->states.size(0) + (size << 3);
	if (size0 > this->N)
	{
		this->states = this->states.slice(0, size0 - this->N);
		this->probs = this->probs.slice(0, size0 - this->N);
		this->values = this->values.slice(0, size0 - this->N);
	}
	at::Tensor state = torch::cat(states, 0);
	at::Tensor prob = torch::stack(probs, 0).reshape({ size,this->gomoku.get_n(),this->gomoku.get_n() });
	at::Tensor value = torch::tensor(values).reshape({ size,1 });
	at::Tensor state_flip = state.flip(2);
	at::Tensor prob_flip = prob.flip(1);
	this->states = torch::cat({ this->states,state,state.rot90(1,{2,3}),state.rot90(2,{2,3}),state.rot90(3,{2,3}),
								state_flip,state_flip.rot90(1,{2,3}),state_flip.rot90(2,{2,3}),state_flip.rot90(3,{2,3}) }, 0);
	this->probs = torch::cat({ this->probs,prob,prob.rot90(1,{1,2}),prob.rot90(2,{1,2}),prob.rot90(3,{1,2}),
							prob_flip,prob_flip.rot90(1,{1,2}),prob_flip.rot90(2,{1,2}),prob_flip.rot90(3,{1,2}) }, 0);
	this->values = torch::cat({ this->values,value,value,value,value,value,value,value,value }, 0);
}

void Train::push(const at::Tensor &s, const at::Tensor &p, const at::Tensor &z)
{
	//while (this->values.size() >= this->N && this->N > 0)
	//{
	//	this->states.pop_front();
	//	this->probs.pop_front();
	//	this->values.pop_front();
	//}
	//this->states.emplace_back(s);
	//std::cout << this->states[this->states.size()-1] << std::endl;
	//this->probs.emplace_back(p);
	//this->values.emplace_back(z);
}

double Train::evaluate(const char *best_path, uint32_t num=20)
{
	PolicyValueNet network(best_path, true, this->state_c, this->gomoku.get_n(), this->gomoku.get_action_dim());
	MCTS mcts(&network, this->n_thread, this->c_puct, this->temp, this->n_simulate, this->virtual_loss, this->gomoku.get_action_dim(), true);
	this->mcts.set_temp(1e-3);
	mcts.set_temp(1e-3);
	int winner;
	bool swap = false;
	uint32_t i, count1 = 0, count2 = 0;
	for (i = 0; i < num; i++)
	{
		winner = this->gomoku.start_play(&this->mcts, &mcts, swap, false);
		if (winner == 1) count1 += 1;
		else if (winner == -1) count2 += 1;
		swap = !swap;
	}
	double ratio = (count1 + (double)(num - count1 - count2) / 2) / num;
	if (ratio > 0.55) this->network.save_model(best_path);
	else this->network.load_model(best_path);
	return ratio;
}

void Train::run(const char *model_path, const char *best_path)
{
	uint32_t i, j, k, size, idx;
	if (!file_exists(best_path)) this->network.save_model(best_path);
	std::vector<double> res;
	double kl, best_ratio = 0, ratio;
	TimeCounter timer;
	for (i = 0; i < this->n_game; i++)
	{
		timer.start();
		std::vector<at::Tensor> states, probs, values_;
		std::vector<float> values;
		mcts.self_play(&this->gomoku, states, probs, values, this->temp, 20, true, false);
		this->augment_data(states, probs, values);
		size = this->states.size(0);
		std::printf("game %4d/%d : duration=%.3fs  episode=%d  buffer=%d\n", i, this->n_game, timer.end_s(), states.size(), size);
		states.clear(); probs.clear(); values.clear(); values_.clear();
		if (size < this->batch_size) continue;
		//for (k = 0; k < size; k++)
		//{
		//	states.push_back(this->states[k]);
		//	probs.push_back(this->probs[k]);
		//	values_.push_back(this->values[k]);
		//}
		for (j = 0; j < this->epochs; j++)
		{
			at::Tensor index = torch::randperm(size, torch::Dtype::Long);
			at::Tensor index1;
			k = 0;
			while (k < size)
			{
				timer.start();
				index1 = index.slice(0, k, k + this->batch_size);
				if (k + this->batch_size > size)
				{
					// 补齐batch
					index1 = torch::cat({ index1,index.slice(0, 0, k + this->batch_size - size) }, 0);
				}
				res = this->train_step(this->states.index(index1), this->probs.index(index1).reshape({index1.size(0),this->gomoku.get_action_dim()}), 
					this->values.index(index1), this->lr * this->c_lr);
				kl = res[2];
				std::printf("train %3d/%d : cross_entropy_loss=%.8f  mse_loss=%.8f  kl=%.8f  R2_old=%.8f  R2_new=%.8f  c_lr=%.5f  duration=%.3fs\n", 
					j, this->epochs, res[0], res[1], kl, res[3], res[4], this->c_lr, timer.end_s());
				k += this->batch_size;
			}
			//at::Tensor index = torch::randint(size, this->batch_size);
			//states.clear(); probs.clear(); values.clear(); values_.clear();
			//for (k = 0; k < this->batch_size; k++)
			//{
			//	idx = index[k].item().toInt();
			//	states.push_back(this->states[idx]);
			//	probs.push_back(this->probs[idx]);
			//	values_.push_back(this->values[idx]);
			//}
			//res = this->train_step(states, probs, values_, this->lr * this->c_lr);
			//kl = res[2];
			//if (kl > this->kl_targ * 2 && this->c_lr > 0.1) this->c_lr /= 1.5;
			//else if (kl < this->kl_targ / 2 && this->c_lr < 10) this->c_lr *= 1.5;
			//std::printf("train %3d/%d : cross_entropy_loss=%.8f  mse_loss=%.8f  kl=%.8f  R2_old=%.8f  R2_new=%.8f  c_lr=%.5f  duration=%.3fs\n", 
			//		j, this->epochs, res[0], res[1], kl, res[3], res[4], this->c_lr, timer.end_s());
		}
		this->network.save_model(model_path);
		if ((i + 1) % this->check_freq == 0)
		{
			timer.start();
			ratio = this->evaluate(best_path);
			if (ratio > best_ratio) best_ratio = ratio;
			std::printf("evaluate : ratio=%.8f  best_ratio=%.8f  duration=%.3fs\n", ratio, best_ratio, timer.end_s());
		}
	}
}

std::vector<double> Train::train_step(const std::vector<at::Tensor> &state, const std::vector<at::Tensor> &prob, const std::vector<at::Tensor> &value, const double &lr)
{
	at::Tensor s = torch::cat(state, 0);
	at::Tensor p = torch::cat(prob, 0);
	at::Tensor z = torch::cat(value, 0);
	return this->train_step(s, p, z, lr);
}

std::vector<double> Train::train_step(const at::Tensor &state, const at::Tensor &prob, const at::Tensor &value, const double &lr)
{
	at::Tensor s = state.to(this->network.device);
	at::Tensor p = prob.to(this->network.device);
	at::Tensor z = value.to(this->network.device);
	/*auto param_groups = this->optimizer.param_groups();
	uint32_t i, n = param_groups.size();
	for (i = 0; i < n; i++)
	{
		param_groups[i].set_options(std::make_unique<torch::optim::AdamOptions>(torch::optim::AdamOptions(lr)));
	}*/
	this->optimizer.zero_grad();
	std::vector<at::Tensor> res = this->network.model->forward(s);
	at::Tensor loss1 = torch::binary_cross_entropy(res[0], p);
	at::Tensor loss2 = torch::mse_loss(res[1], z);
	at::Tensor loss = loss1 + loss2;
	loss.backward();
	this->optimizer.step();
	std::vector<at::Tensor> res1 = this->network.model->forward(s);
	// 新旧预测值的KL散度
	at::Tensor kl = (res1[0] * ((res1[0] + 1e-10).log() - (res[0] + 1e-10).log())).sum(1).mean();
	at::Tensor z_var = torch::var(z, 0, true, false);
	at::Tensor R2_old = 1 - torch::var(z - res[1], 0, true, false) / z_var;
	at::Tensor R2_new = 1 - torch::var(z - res1[1], 0, true, false) / z_var;
	return { loss1.item().toDouble(),loss2.item().toDouble(),kl.item().toDouble(),R2_old.item().toDouble(),R2_new.item().toDouble() };
}

int main()
{
	Train train;
	train.run(model_path, best_path);
	return 0;
}