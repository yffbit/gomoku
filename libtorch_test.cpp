#include <vector>
#include <deque>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;

int main()
{
	at::Tensor a = torch::tensor({ {1,2},{3,4} }, torch::Dtype::Int);
	at::Tensor b = torch::tensor({ {1,0},{2,4} }, torch::Dtype::Int);
	cout << "a:" << endl << a << endl;
	cout << "b:" << endl << b << endl;

	// reshape û�и�������
	a.reshape({ 4 })[0] = 0;
	cout << "a.reshape({ 4 })[0] = 0" << endl << a << endl;
	a.reshape({ 4 })[2] = 10;
	cout << "a.reshape({ 4 })[2] = 10" << endl << a << endl;

	// ��ӦԪ�ز���
	cout << "a + b" << endl << a + b << endl;
	cout << "a * b" << endl << a * b << endl;
	cout << "a.mul(b)" << endl << a.mul(b) << endl;
	// ��������
	cout << "a.matmul(b)" << endl << a.matmul(b) << endl;
	cout << "b.matmul(a)" << endl << b.matmul(a) << endl;
	// �㲥����
	cout << "a + 1" << endl << a + 1 << endl;
	cout << "(a > 1)" << endl << (a > 1) << endl;
	cout << "a.gt(1)" << endl << a.gt(1) << endl;
	cout << "(a < 1)" << endl << (a < 1) << endl;
	cout << "a.lt(1)" << endl << a.lt(1) << endl;
	// ����
	at::Tensor c = a.select(0, 0); // 0 ����ȡ index=0 �����ݣ�Ҳ���ǵ�һ�����ݣ������ڴ棩
	at::Tensor d = a.select(1, 0); // 1 ����ȡ index=0 �����ݣ�Ҳ���ǵ�һ�����ݣ������ڴ棩
	cout << "a:" << endl << a << endl;
	cout << "c:" << endl << c << endl;
	cout << "d:" << endl << d << endl;
	a[0][0] = 0;
	cout << "a:" << endl << a << endl;
	cout << "c:" << endl << c << endl;
	cout << "d:" << endl << d << endl;

	at::Tensor e = a.index_select(0, torch::tensor({1,1,0})); // 0 ����ȡ index={1,1,0} �����ݣ�Ҳ�������εڶ������ݣ�һ�ε�һ�����ݣ��������ڴ棩
	at::Tensor f = a.index_select(1, torch::tensor({1,0}));	// 1 ����ȡ index={1,0} �����ݣ�Ҳ���ǵڶ������ݣ���һ�����ݣ��������ڴ棩
	cout << "a:" << endl << a << endl;
	cout << "e:" << endl << e << endl;
	cout << "f:" << endl << f << endl;
	a[0][0] = 1; e[1][0] = 8; f[1][0] = 9;
	cout << "a:" << endl << a << endl;
	cout << "e:" << endl << e << endl;
	cout << "f:" << endl << f << endl;

	at::Tensor g = torch::cat({ a,b }, 0);	// ������ά�ȣ�����ָ��ά�ȶѵ����������ڴ棩
	cout << "a:" << endl << a << endl;
	cout << "b:" << endl << b << endl;
	cout << "g:" << endl << g << endl;
	g[0][1] = 12; g[2][0] = 13;
	cout << "a:" << endl << a << endl;
	cout << "b:" << endl << b << endl;
	cout << "g:" << endl << g << endl;
	cout << " torch::cat({ a,b }, 1):" << endl << torch::cat({ a,b }, 1) << endl;

	at::Tensor h = torch::stack({ a,b }, 0);	// ����ά�ȣ�����ָ��ά�ȶѵ����������ڴ棩
	cout << "a:" << endl << a << endl;
	cout << "b:" << endl << b << endl;
	cout << "h:" << endl << h << endl;
	h[0][0][1] = 14; h[1][0][0] = 15;
	cout << "a:" << endl << a << endl;
	cout << "b:" << endl << b << endl;
	cout << "h:" << endl << h << endl;
	cout << " torch::stack({ a,b }, 1):" << endl << torch::stack({ a,b }, 1) << endl;// a �ĵ�һ�к� b �ĵ�һ�жѵ���a �ĵڶ��к� b �ĵڶ��жѵ�

	at::Tensor i = g.slice(0, 0, 100, 2);	// 0 ���ϴ� index=0 ��ʼȡ���� index=100 ���� index����2 ȡ��һ�У�������... �������ڴ棩
	cout << "g:" << endl << g << endl;
	cout << "i:" << endl << i << endl;
	i[1][1] = 20;
	cout << "g:" << endl << g << endl;
	cout << "i:" << endl << i << endl;

	vector<int> j(3, 0);
	j[1] = 10; j[2] = 20;
	at::Tensor k = torch::tensor(j, torch::Dtype::Int);	// �������ڴ�
	cout << "j:" << endl << j << endl;
	cout << "k:" << endl << k << endl;
	j[2] = 3; j[1] = 2; j[0] = 1;
	cout << "j:" << endl << j << endl;
	cout << "k:" << endl << k << endl;

	at::Tensor l = a;	// �����ڴ�
	cout << "a:" << endl << a << endl;
	cout << "l:" << endl << l << endl;
	a[0][0] = 11;
	cout << "a:" << endl << a << endl;
	cout << "l:" << endl << l << endl;

	deque<at::Tensor> m(2, torch::zeros({ 2,3 }, torch::Dtype::Int));
	m[0][0] = 1; m[1][1] = 2;	// ����Ԫ�ع����ڴ�
	cout << "m[0]:" << endl << m[0] << endl;
	cout << "m[1]:" << endl << m[1] << endl;
	m[1][1][0] = 3;
	cout << "m[0]:" << endl << m[0] << endl;
	cout << "m[1]:" << endl << m[1] << endl;

	vector<at::Tensor> n(2, torch::zeros({ 2,3 }, torch::Dtype::Int));
	n[0][0] = 3; n[1][1] = 4;	// ����Ԫ�ع����ڴ�
	cout << "n[0]:" << endl << n[0] << endl;
	cout << "n[1]:" << endl << n[1] << endl;
	n[1][1][0] = 5;
	cout << "n[0]:" << endl << n[0] << endl;
	cout << "n[1]:" << endl << n[1] << endl;
	n.clear();
	n.push_back(a);	// û�������ڴ�
	n.push_back(a);
	cout << "a:" << endl << a << endl;
	cout << "n[0]:" << endl << n[0] << endl;
	cout << "n[1]:" << endl << n[1] << endl;
	a[0][0] = 14;
	cout << "a:" << endl << a << endl;
	cout << "n[0]:" << endl << n[0] << endl;
	cout << "n[1]:" << endl << n[1] << endl;

	n.clear();
	n.emplace_back(a);	// û�������ڴ�
	n.emplace_back(a);
	cout << "a:" << endl << a << endl;
	cout << "n[0]:" << endl << n[0] << endl;
	cout << "n[1]:" << endl << n[1] << endl;
	a[0][0] = 18;
	cout << "a:" << endl << a << endl;
	cout << "n[0]:" << endl << n[0] << endl;
	cout << "n[1]:" << endl << n[1] << endl;

	at::Tensor o0 = torch::zeros({ 0,2,3,3 }, torch::Dtype::Int);
	cout << "torch::zeros({ 0,2,3,3 }, torch::Dtype::Int)" << endl << o0 << endl;

	at::Tensor o1 = torch::range(1, 18, torch::Dtype::Int).reshape({1,2,3,3});
	at::Tensor o2 = torch::range(19, 36, torch::Dtype::Int).reshape({1,2,3,3});
	at::Tensor o = torch::cat({ o0,o1,o2 }, 0);
	cout << "o1:" << endl << o1 << endl;
	cout << "o2:" << endl << o2 << endl;
	cout << "o:" << endl << o << endl;
	// {2,3}����ʱ����ת90��*1��Ҳ����ÿ��ͨ����ʱ����ת90��
	cout << "o.rot90(1, {2,3}):" << endl << o.rot90(1, {2,3}) << endl;
	// ��2���Ϸ�ת��Ҳ����ÿ��ͨ�����·�ת
	cout << "o.flip(2):" << endl << o.flip(2) << endl;
	cout << "o.reshape({2,2,9}):" << endl << o.reshape({2,2,9}) << endl;
	cout << "o.reshape({2,2,9}).reshape({2,2,3,3}):" << endl << o.reshape({2,2,9}).reshape({2,2,3,3}) << endl;

	cout << "torch::get_num_threads() : " << torch::get_num_threads() << endl;

	cout << torch::randn({2,3}) << endl;
	cout << torch::randperm(10) << endl;
	cout << torch::randperm(10) << endl;
	cout << torch::randperm(10) << endl;
	cout << torch::randperm(10).slice(0, 0, 2) << endl;
	cout << torch::randperm(10).slice(0,0,100) << endl;
	cout << torch::cat({ torch::randperm(10).slice(0,0,2),torch::randperm(10).slice(0,0,100) }, 0) << endl;

	at::Tensor idx = torch::tensor({ 1,0,1,0 });
	cout << idx << endl;
	at::Tensor p = a.index(idx);	// �������ڴ�
	cout << "a:" << endl << a << endl;
	cout << "p:" << endl << p << endl;
	a[0][1] = 16; a[1][0] = 17; p[1][1] = 19;
	cout << "a:" << endl << a << endl;
	cout << "p:" << endl << p << endl;
	cout << a.index(torch::tensor({1})) << endl;
	cout << a[1].index(torch::tensor({1})) << endl;

	at::Tensor q = torch::range(1, 30, torch::Dtype::Int).reshape({ 10,3 });
	cout << "q:" << endl << q << endl;
	cout << "idx:" << endl << idx << endl;
	cout << "q.index(idx):" << endl << q.index(idx) << endl;

	idx = torch::randperm(q.size(0), torch::Dtype::Long);	// ������ CPULongType
	//idx = torch::randperm(q.size(0), torch::Dtype::Int);	// CPUIntType ���г���
	cout << "q:" << endl << q << endl;
	cout << "idx:" << endl << idx << endl;
	cout << "q.index(idx):" << endl << q.index(idx) << endl;

	return 0;
}