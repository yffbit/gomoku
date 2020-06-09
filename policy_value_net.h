#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <iostream>

struct ResidualBlockImpl : torch::nn::Module
{
	ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int kernel_size = 3) :
		conv1(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(1).padding((kernel_size - 1) >> 1)),
		conv2(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).stride(1).padding((kernel_size - 1) >> 1)),
		batch_norm1(torch::nn::BatchNorm2dOptions(out_channels)),
		batch_norm2(torch::nn::BatchNorm2dOptions(out_channels))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("batch_norm1", batch_norm1);
		register_module("batch_norm2", batch_norm2);
	}
	at::Tensor forward(at::Tensor x)
	{
		at::Tensor x_ = x;
		x = batch_norm1(conv1(x)).relu();
		x = torch::add(batch_norm2(conv2(x)), x_);
		return torch::relu(x);
	}
	torch::nn::Conv2d conv1, conv2;
	torch::nn::BatchNorm2d batch_norm1, batch_norm2;
};
TORCH_MODULE(ResidualBlock);

struct NetworkImpl : torch::nn::Module
{
	NetworkImpl(int64_t state_channels, int64_t state_h, int64_t action_dim, int64_t conv_channels, int kernel_size = 3) :
		conv(torch::nn::Conv2dOptions(state_channels, conv_channels, kernel_size).stride(1).padding((kernel_size - 1) >> 1)),
		bn(torch::nn::BatchNorm2dOptions(conv_channels)),
		residual(ResidualBlock(conv_channels, conv_channels, kernel_size)),
		conv_policy(torch::nn::Conv2dOptions(conv_channels, 2, 1).stride(1)),
		bn_policy(torch::nn::BatchNorm2dOptions(2)),
		fc_policy(2 * state_h * state_h, action_dim),
		conv_value(torch::nn::Conv2dOptions(conv_channels, 1, 1).stride(1)),
		bn_value(torch::nn::BatchNorm2dOptions(1)),
		fc1_value(state_h * state_h, 256),
		fc2_value(256, 1)
	{
		register_module("conv", conv);
		register_module("bn", bn);
		register_module("residual", residual);
		register_module("conv_policy", conv_policy);
		register_module("bn_policy", bn_policy);
		register_module("fc_policy", fc_policy);
		register_module("conv_value", conv_value);
		register_module("bn_value", bn_value);
		register_module("fc1_value", fc1_value);
		register_module("fc2_value", fc2_value);
	}

	std::vector<at::Tensor> forward(at::Tensor x)
	{
		at::Tensor action_prob, value;
		x = residual(bn(conv(x)).relu());
		action_prob = fc_policy(bn_policy(conv_policy(x)).relu().flatten(1)).softmax(1);
		x = fc1_value(bn_value(conv_value(x)).relu().flatten(1)).relu();
		value = fc2_value(x).tanh();
		return { action_prob, value };
	}

	torch::nn::Conv2d conv;
	torch::nn::BatchNorm2d bn;
	ResidualBlock residual;
	torch::nn::Conv2d conv_policy;
	torch::nn::BatchNorm2d bn_policy;
	torch::nn::Linear fc_policy;
	torch::nn::Conv2d conv_value;
	torch::nn::BatchNorm2d bn_value;
	torch::nn::Linear fc1_value;
	torch::nn::Linear fc2_value;
};
TORCH_MODULE(Network);

class PolicyValueNet
{
public:
	PolicyValueNet(const char *model_path, bool use_cuda, int32_t state_c, int32_t state_h, int32_t action_dim) :
		model(state_c, state_h, action_dim, 128, 3), device(torch::kCPU)
	{
		if (use_cuda && torch::cuda::is_available()) this->device = torch::Device(torch::kCUDA, 0);
		if (nullptr != model_path)
		{
			FILE *fp = fopen(model_path, "rb");
			if (nullptr != fp)
			{
				fclose(fp);
				fp = nullptr;
				this->load_model(model_path);
			}
		}
		this->model->to(this->device);
		this->model(torch::zeros({ 1,state_c,state_h,state_h }, this->device));
	}
	~PolicyValueNet() {};
	inline void save_model(const char * save_path) { torch::save(this->model, save_path); }
	inline void load_model(const char * model_path) { torch::load(this->model, model_path); }
	inline std::vector<at::Tensor> predict(const std::vector<at::Tensor> &x) { return this->model(torch::cat(x, 0).to(this->device)); }
	inline std::vector<at::Tensor> predict(const at::Tensor &x) { return this->model(x); }
	inline at::Tensor dirichlet_noise(uint32_t dim, float alpha)
	{
		std::vector<float> dirichlet(dim, alpha);
		return torch::_sample_dirichlet(torch::tensor(dirichlet, this->device));
	}

	Network model;
	torch::Device device;
};