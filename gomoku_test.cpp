#include <iostream>
#include <vector>
#include "gomoku.h"
#include "mcts.h"

int main()
{
	Gomoku game(10, 5);
	game.display();
	game.execute_move(0);
	game.execute_move(11);
	game.execute_move(22);
	game.execute_move(33);
	game.execute_move(44);
	game.execute_move(55);
	game.execute_move(66);
	game.execute_move(77);
	game.execute_move(88);
	std::vector<int> res = game.get_game_status();
	std::cout << res[0] << "," << res[1] << std::endl;
	game.reset(-1);
	game.display();
	game.execute_move(0);
	game.execute_move(11);
	game.execute_move(22);
	game.display();
	Gomoku game1(19, 5);
	game1.execute_move(0);
	game1.execute_move(20);
	game1.execute_move(40);
	game1.display();
	Human player1, player2;
	game.start_play(&player1, &player2, false, true);
	game1.start_play(&player1, &player2, true, true);

	Gomoku game2(10, 4);
	PolicyValueNet network(nullptr, true, 5, game2.get_n(), game2.get_action_dim());
	MCTS mcts(&network, 4, 5, 1, 50, 3, game2.get_action_dim(), true);
	std::vector<at::Tensor> states, probs;
	std::vector<float> values;
	mcts.self_play(&game2, states, probs, values, 1, 20, true, true);
	return 0;
}