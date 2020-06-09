#pragma once

#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

// ǰ������
class Player;

class Gomoku
{
public:
    Gomoku(uint32_t n, uint32_t n_in_line);
    void reset(int start_player = 1);
    std::vector<bool> get_legal_move();
    bool execute_move(int move);
    std::vector<int> get_game_status();
    at::Tensor curr_state(bool to_device, torch::Device &device);
    void display();
	char get_symbol(int player);
	inline uint32_t get_n() const { return this->n; }
    inline uint32_t get_action_dim() const { return this->dim; }
    inline std::vector<std::vector<int>> get_board() const { return this->board; }
    inline int get_curr_player() const { return this->curr_player; }
	int start_play(Player *player1, Player *player2, bool swap, bool show);

private:
    /* 3 * 3 ����
    0 1 2
    3 4 5
    6 7 8
    */
    std::vector<std::vector<int>> board;
    uint32_t n;
	uint32_t dim;
    uint32_t n_in_line; // ��ʤĿ��
    uint32_t n_count;
    int curr_player;    // ��ұ�ʶ 1,-1
    int last_move1;     // ���һ������
    int last_move2;     // �����ڶ�������
};

class Player
{
public:
	inline Player(int player = 1) :player(player) {}
	inline ~Player() {}
	inline void set_player(int player) { this->player = player; }
	virtual void init() {}
	virtual void update_with_move(int last_move) {}
	inline int get_player() const { return this->player; }
	virtual uint32_t get_action(Gomoku *gomoku, bool explore = false) = 0;
private:
	int player;
};

class Human : public Player
{
public:
	inline Human(int player = 1) :Player(player) {}
	inline ~Human() {}
	inline uint32_t get_action(Gomoku *gomoku, bool explore = false)
	{
		uint32_t n = gomoku->get_n(), i, j;
		while (true)
		{
			std::cin >> i >> j;
			std::cin.clear();
			if (i >= 0 && i < n && j >= 0 && j < n) break;
			else std::cout << "Illegal input. Reenter : ";
		}
		return i * n + j;
	}
};