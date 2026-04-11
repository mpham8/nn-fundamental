cmake_minimum_required(VERSION 3.14)
project(nn_torchless_cpp CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  rapidcsv
  GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
  GIT_TAG v8.92
)
FetchContent_MakeAvailable(rapidcsv)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

add_executable(nn-cpp main.cpp)
target_include_directories(
  nn-cpp PRIVATE "${rapidcsv_SOURCE_DIR}/src" "${eigen_SOURCE_DIR}")
