cmake_minimum_required(VERSION 3.15)
project(algorithm_cpp)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(algorithm_cpp main.cpp)

add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O3)
