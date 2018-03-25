cmake_minimum_required(VERSION 3.2)
project(test_ray)

add_executable(test_ray main.cpp
                        test_common.h
						test_data.cpp
                        test_simd.cpp
                        test_simd.ipp
						test_primary_ray_gen.cpp)

target_link_libraries(test_ray ray)