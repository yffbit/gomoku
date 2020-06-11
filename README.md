# gomoku
Gomoku AI based on AlphaGo Zero algorithm

## Environment
* Cmake 3.17.3
* Visual Studio 2017
* libtorch-win-shared-with-deps-debug-1.5.0
* CUDA 10.1

## Usage
```
# add libtorch to environment variable
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 15 Win64" ..
# open build/gomoku.sln with Visual Studio 2017
```

## Reference
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. https://github.com/junxiaosong/AlphaZero_Gomoku
4. https://github.com/hijkzzz/alpha-zero-gomoku
5. https://github.com/chengstone/cchess-zero
6. https://blog.csdn.net/zdarks/article/details/46994607
7. https://github.com/progschj/ThreadPool