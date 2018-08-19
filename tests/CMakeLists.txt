cmake_minimum_required(VERSION 3.2)
project(test_Ray)

add_executable(test_Ray main.cpp
                        test_common.h
                        test_data.cpp
						test_img1.h
						test_mesh_lights.cpp
                        test_simd.cpp
                        test_simd.ipp
                        test_primary_ray_gen.cpp
						test_scene1.h
                        )

target_link_libraries(test_Ray Ray)