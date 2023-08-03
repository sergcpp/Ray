cmake_minimum_required(VERSION 3.18)
project(append_refl_data)

IF(WIN32)
    set(LIBS opengl32)
ELSEIF(UNIX AND NOT APPLE)
    set(LIBS pthread)
ELSE()
    set(LIBS )
ENDIF()

add_executable(append_refl_data main.cpp)

target_link_libraries(append_refl_data dxcompiler)
