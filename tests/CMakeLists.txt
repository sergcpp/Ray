cmake_minimum_required(VERSION 3.2)
project(test_Ray)

IF(WIN32)
    set(LIBS opengl32)
ELSEIF(UNIX AND NOT APPLE)
    set(LIBS pthread)
ELSE()
    set(LIBS )
ENDIF()

add_executable(test_Ray main.cpp
                        test_common.h
                        test_materials.cpp
                        test_simd.cpp
                        test_simd_avx.cpp
                        test_simd_avx2.cpp
                        test_simd_avx512.cpp
                        test_simd_sse41.cpp
                        test_simd.ipp
                        test_tex_storage.cpp
                        thread_pool.h
                        utils.h
                        utils.cpp)

if(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
    set_source_files_properties(test_simd_avx.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
    set_source_files_properties(test_simd_avx2.cpp PROPERTIES COMPILE_FLAGS /arch:AVX2)
    set_source_files_properties(test_simd_avx512.cpp PROPERTIES COMPILE_FLAGS /arch:AVX512)
endif(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")

target_link_libraries(test_Ray Ray ${LIBS})

set_target_properties(test_Ray PROPERTIES OUTPUT_NAME_DEBUG test_Ray-dbg)
set_target_properties(test_Ray PROPERTIES OUTPUT_NAME_RELWITHDEBINFO test_Ray-dev)
set_target_properties(test_Ray PROPERTIES OUTPUT_NAME_ASAN test_Ray-asan)
set_target_properties(test_Ray PROPERTIES OUTPUT_NAME_RELEASE test_Ray)

set_target_properties(test_Ray PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
