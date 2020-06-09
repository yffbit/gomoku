cmake_minimum_required(VERSION 2.8)

project(gomoku)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

add_executable(${PROJECT_NAME} "gomoku_test.cpp" "gomoku.cpp" "mcts.cpp")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")