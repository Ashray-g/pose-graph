cmake_minimum_required(VERSION 3.23)
project(FactorGraphs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O1")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Ceres REQUIRED)

add_executable(FactorGraphs main.cpp types.h pose_graph_3d_error_term.h pose_graph_pixel_error_term.h Camera.h)
target_link_libraries (FactorGraphs Eigen3::Eigen Ceres::ceres)
