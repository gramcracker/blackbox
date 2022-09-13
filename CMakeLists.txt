cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(blackbox VERSION 1.0)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_subdirectory(include/neural_network include/neural_network/)
add_executable(a ${SRC_FILES})
target_link_libraries(a PUBLIC neuralnetwork)