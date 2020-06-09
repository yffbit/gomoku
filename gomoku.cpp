#include <math.h>
#include <iomanip>
#include "gomoku.h"

Gomoku::Gomoku(uint32_t n, uint32_t n_in_line) :
	n(n), n_in_line(n_in_line), curr_player(1), n_count(0), last_move1(-1), last_move2(-1), dim(n * n)
{
	this->board = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
}

void Gomoku::reset(int start_player)
{
	uint32_t i, j;
	this->board;
	this->curr_player = start_player;
	this->n_count = 0;
	this->last_move1 = -1;
	this->last_move2 = -1;
	for (i = 0; i < this->n; i++)
	{
		for (j = 0; j < this->n; j++)
		{
			this->board[i][j] = 0;
		}
	}
}

std::vector<bool> Gomoku::get_legal_move()
{
	uint32_t n = this->n, i, j;
	std::vector<bool> legal_moves(this->get_action_dim(), 0);

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (this->board[i][j] == 0) legal_moves[i * n + j] = 1;
		}
	}
	return legal_moves;
}

bool Gomoku::execute_move(int move)
{
	if (move < 0 || move >= this->dim) return false;
	uint32_t i = move / this->n;
	uint32_t j = move % this->n;

	// if (this->board[i][j] != 0) throw runtime_error("execute_move borad[i][j] != 0.");
	if (this->board[i][j] != 0) return false;

	this->board[i][j] = this->curr_player;
	this->last_move2 = this->last_move1;
	this->last_move1 = move;
	this->n_count++;
	this->curr_player = -this->curr_player;
	return true;
}

std::vector<int> Gomoku::get_game_status()
{
	uint32_t n = this->n, i, j, k;
	uint32_t n_in_line = this->n_in_line;
	if (this->n_count < 2 * n_in_line - 1) return { 0, 0 };
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (this->board[i][j] == 0) continue;
			// 向下
			if (j <= n - n_in_line)
			{
				k = 0;
				while (k < n_in_line && this->board[i][j + k] == this->board[i][j]) k++;
				if (k == n_in_line) return { 1, this->board[i][j] };
			}
			// 向右
			if (i <= n - n_in_line)
			{
				k = 0;
				while (k < n_in_line && this->board[i + k][j] == this->board[i][j]) k++;
				if (k == n_in_line) return { 1, this->board[i][j] };
			}
			// 右下
			if (i <= n - n_in_line && j <= n - n_in_line)
			{
				k = 0;
				while (k < n_in_line && this->board[i + k][j + k] == this->board[i][j]) k++;
				if (k == n_in_line) return { 1, this->board[i][j] };
			}
			// 左下
			if (i <= n - n_in_line && j >= n_in_line - 1)
			{
				k = 0;
				while (k < n_in_line && this->board[i + k][j - k] == this->board[i][j]) k++;
				if (k == n_in_line) return { 1, this->board[i][j] };
			}
		}
	}
	if (this->n_count < this->get_action_dim()) return { 0, 0 };
	else return { 1, 0 };
}

at::Tensor Gomoku::curr_state(bool to_device, torch::Device &device)
{
    // 获取状态作为神经网络的输入  [batch channels height width]
    // 当前玩家视角
    uint32_t i, j, m;
	at::Tensor s;
	if (to_device) s = torch::zeros({ 1,5,this->n,this->n }, device);
	else s = torch::zeros({ 1,5,this->n,this->n }, torch::Dtype::Float);
    int a = this->curr_player, b = 0, m1 = this->last_move1, m2 = this->last_move2;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            // 当前位置对应的棋子
            m = i * this->n + j;
            b = this->board[i][j];
            // 当前玩家的棋子  最近一次落子 m2
            s[0][0][i][j] = b == a ? 1 : 0;
            // 当前玩家的棋子  不包括最近一次落子(m2)
            s[0][1][i][j] = (b == a && m2 != m) ? 1 : 0;
            // 对手玩家的棋子  最近一次落子 m1
            s[0][2][i][j] = b == (-a) ? 1 : 0;
            // 对手玩家的棋子  不包括最近一次落子(m1)
            s[0][3][i][j] = (b == (-a) && m1 != m) ? 1 : 0;
            // 棋子颜色
            s[0][4][i][j] = a == 1 ? 1 : 0;
        }
    }
	return s;
}

void Gomoku::display()
{
	uint32_t i, j;
	std::printf("Player 'X', Player 'O'. Target %d. Current : '%c'\n   ", this->n_in_line, this->get_symbol(this->curr_player));
	for (j = 0; j < this->n; j++) std::cout << std::setw(3) << std::setfill(' ') << j;
	std::cout << std::endl;
	for (i = 0; i < this->n; i++)
	{
		std::cout << std::setw(3) << std::setfill(' ') << i;
		for (j = 0; j < this->n; j++)
		{
			std::cout << std::setw(3) << std::setfill(' ') << this->get_symbol(this->board[i][j]);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

char Gomoku::get_symbol(int player)
{
	if (player == 1) return 'X';
	else if (player == -1) return 'O';
	else return '_';
}

int Gomoku::start_play(Player *player1, Player *player2, bool swap, bool show)
{
	player1->set_player(1);
	player1->init();
	player2->set_player(-1);
	player2->init();
	Player * players[2] = { player1,player2 };
	uint32_t idx = 0, move;
	if (swap) idx = 1;	// 交换先后手
	this->reset(players[idx]->get_player());
	std::vector<int> res(2, 0);
	if (show)
	{
		std::cout << "New game." << std::endl;
		this->display();
	}
	while (0 == res[0])
	{
		if (show)
		{
			std::printf("Player '%c' (example: 0 0):", this->get_symbol(players[idx]->get_player()));
		}
		move = players[idx]->get_action(this);
		if (this->execute_move(move))
		{
			if (show)
			{
				std::printf("Player '%c' : %d %d\n", this->get_symbol(players[idx]->get_player()), move / this->get_n(), move % this->get_n());
			}
			players[idx]->update_with_move(move);
			res = this->get_game_status();
			idx = 1 - idx;
			if (show) this->display();
		}
	}
	// 玩家重置
	player1->update_with_move(-1);
	player2->update_with_move(-1);
	if (show)
	{
		if (0 != res[1]) std::printf("Game end. Winner is Player '%c'.\n", this->get_symbol(res[1]));
		else std::cout << "Game end. Tie." << std::endl;
	}
	return res[1];
}