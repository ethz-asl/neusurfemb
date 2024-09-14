cmake_minimum_required(VERSION 3.1..3.23)
project(estimate_bbox)

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(estimate_bbox estimate_bbox.cpp)
target_link_libraries(estimate_bbox CGAL::CGAL Eigen3::Eigen)