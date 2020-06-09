#include "policy_value_net.h"
#include <ctime>

void compare(uint32_t batch)
{
	uint32_t n = 10, action_dim, state_c = 5;
	action_dim = n * n;
	PolicyValueNet network(nullptr, true, state_c, n, action_dim);
	at::Tensor x = torch::zeros({ batch,state_c,n,n }, torch::Dtype::Float);
	clock_t s = clock();
	std::vector<at::Tensor> pred = network.model->forward(x);
	clock_t e = clock();
	clock_t s1 = clock();
	x = x.to(network.device);
	network.model->to(network.device);
	clock_t e1 = clock();
	std::vector<at::Tensor> pred1 = network.model->forward(x);
	clock_t e2 = clock();
	std::printf("batch %4d\t%d\t%d\t%d\n", batch, e - s, e2 - s1, e2 - e1);
}

int main()
{
	uint32_t n = 100, action_dim, state_c = 100;
	action_dim = n * n;
	PolicyValueNet network(nullptr, true, state_c, n, action_dim);
	at::Tensor x = torch::zeros({ 2,state_c,n,n }, torch::Dtype::Float);
	x = x.to(network.device);
	network.model->to(network.device);
	std::vector<at::Tensor> pred = network.model->forward(x);

	for (uint32_t i = 1; i < 8192; i = i << 1) compare(i);
	for (uint32_t i = 0; i < 10; i++) compare(1);
	for (uint32_t i = 0; i < 10; i++) compare(2);

	//std::cout << x << std::endl;
	//x[0][4] = 1;
	//std::cout << x << std::endl;
	std::cout << x.sizes() << std::endl;
	x = network.model->conv(x);
	std::cout << x.sizes() << std::endl;
	x = network.model->bn(x).relu();
	std::cout << x.sizes() << std::endl;
	x = network.model->residual(x);
	std::cout << x.sizes() << std::endl;
	at::Tensor y = network.model->conv_policy(x);
	y = network.model->bn_policy(y).relu();
	std::cout << y.sizes() << std::endl;
	y = y.flatten(1);
	std::cout << y.sizes() << std::endl;
	y = network.model->fc_policy(y);
	std::cout << y.sizes() << std::endl;
	y = y.softmax(1);
	std::cout << y.sizes() << std::endl;

	at::Tensor z = network.model->conv_value(x);
	z = network.model->bn_value(z).relu();
	std::cout << z.sizes() << std::endl;
	z = z.flatten(1);
	std::cout << z.sizes() << std::endl;
	z = network.model->fc1_value(z).relu();
	std::cout << z.sizes() << std::endl;
	z = network.model->fc2_value(z);
	z = z.tanh();
	std::cout << z.sizes() << std::endl;

	return 0;
}