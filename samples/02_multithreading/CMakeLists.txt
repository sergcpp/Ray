cmake_minimum_required(VERSION 3.18)
project(02_multithreading)

find_package(OpenMP QUIET)

if(OPENMP_FOUND AND NOT TARGET OpenMP::OpenMP_CXX)
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

if(OPENMP_FOUND)
    set(OPENMP_CXX OpenMP::OpenMP_CXX)
endif()

add_executable(02_multithreading main.cpp)

target_link_libraries(02_multithreading Ray ${OPENMP_CXX})
