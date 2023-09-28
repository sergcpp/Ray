cmake_minimum_required(VERSION 3.18)
project(pmj_gen)

include_directories(pmj-cpp)

add_executable(pmj_gen main.cpp
                       pmj-cpp/sample_generation/pj.cc
                       pmj-cpp/sample_generation/pmj.cc
                       pmj-cpp/sample_generation/pmj02.cc
                       pmj-cpp/sample_generation/pmj02_util.cc
                       pmj-cpp/sample_generation/select_subquad.cc
                       pmj-cpp/sample_generation/util.cc)

set_target_properties(pmj_gen PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")