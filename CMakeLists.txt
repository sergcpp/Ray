cmake_minimum_required(VERSION 2.8.12)
project(Ray)

OPTION(ENABLE_UNITYBUILD "Enable compilation of one large cpp file" ON)

IF(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Ob1")
ELSE(MSVC)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse2 -mfma")
        IF(WIN32)
        ELSE(WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        ENDIF(WIN32)
    endif()
ENDIF(MSVC)

set(INTERNAL_SOURCE_FILES internal/BVHSplit.h
                          internal/BVHSplit.cpp
                          internal/Core.h
                          internal/Core.cpp
                          internal/CoreRef.h
                          internal/CoreRef.cpp
                          internal/CoreSIMD.h
                          internal/FramebufferRef.h
                          internal/FramebufferRef.cpp
                          internal/Halton.h
                          internal/RendererRef.h
                          internal/RendererRef.cpp
                          internal/RendererSIMD.h
                          internal/SceneRef.h
                          internal/SceneRef.cpp
                          internal/SmallVector.h
                          internal/SparseStorage.h
                          internal/Span.h
                          internal/TextureAtlasRef.h
                          internal/TextureAtlasRef.cpp
                          internal/TextureSplitter.h
                          internal/TextureSplitter.cpp
                          internal/TextureUtilsRef.h
                          internal/TextureUtilsRef.cpp
                          internal/Time_.h
                          internal/Time.cpp
                          internal/UniformIntDistribution.h)
                          
if(NOT CMAKE_SYSTEM_NAME MATCHES "Android" AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")

set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                          internal/RendererSSE2.h
                          internal/RendererSSE2.cpp
                          internal/RendererSSE41.h
                          internal/RendererSSE41.cpp
                          internal/RendererAVX.h
                          internal/RendererAVX.cpp
                          internal/RendererAVX2.h
                          internal/RendererAVX2.cpp)
endif()

if("${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                          internal/RendererNEON.h
                          internal/RendererNEON.cpp)
endif()

if(MSVC AND NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    if(NOT CMAKE_CL_64)
        set_source_files_properties(internal/RendererSSE2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
        set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
    endif()
    set_source_files_properties(internal/RendererAVX.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
    set_source_files_properties(internal/RendererAVX2.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
endif(MSVC AND NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")

set(SOURCE_FILES Log.h
                 RendererBase.h
                 RendererBase.cpp
                 RendererFactory.h
                 RendererFactory.cpp
                 SceneBase.h
                 SceneBase.cpp
                 Types.h)

set(SIMD_FILES internal/simd/aligned_allocator.h
               internal/simd/detect.h
               internal/simd/detect.cpp
               internal/simd/simd_vec.h
               internal/simd/simd_vec_sse.h
               internal/simd/simd_vec_avx.h
               internal/simd/simd_vec_avx512.h
               internal/simd/simd_vec_neon.h)

list(APPEND ALL_SOURCE_FILES ${INTERNAL_SOURCE_FILES})
source_group("src\\internal" FILES ${INTERNAL_SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${INTERNAL_DEPRECATED_SOURCE_FILES})
source_group("src\\internal\\deprecated" FILES ${INTERNAL_DEPRECATED_SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${SOURCE_FILES})
source_group("src" FILES ${SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${KERNEL_FILES})
source_group("src\\internal\\kernels" FILES ${KERNEL_FILES})

list(APPEND ALL_SOURCE_FILES ${SIMD_FILES})
source_group("src\\internal\\simd" FILES ${SIMD_FILES})

set_source_files_properties(${KERNEL_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# enable or disable unity build
if(ENABLE_UNITYBUILD)
    set_source_files_properties(${INTERNAL_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${INTERNAL_DEPRECATED_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

    list(APPEND ALL_SOURCE_FILES _Ray.cpp)
    source_group("src" FILES _Ray.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Sse2.cpp)
    source_group("src" FILES _Ray_Sse2.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Sse41.cpp)
    source_group("src" FILES _Ray_Sse41.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Avx.cpp)
    source_group("src" FILES _Ray_Avx.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Avx2.cpp)
    source_group("src" FILES _Ray_Avx2.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Avx512.cpp)
    source_group("src" FILES _Ray_Avx512.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Neon.cpp)
    source_group("src" FILES _Ray_Neon.cpp)

    if(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
        if(NOT CMAKE_CL_64)
            set_source_files_properties(_Ray_Sse2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
            set_source_files_properties(_Ray_Sse41.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
        endif()
        set_source_files_properties(_Ray_Avx.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
        set_source_files_properties(_Ray_Avx2.cpp PROPERTIES COMPILE_FLAGS /arch:AVX2)
        set_source_files_properties(_Ray_Avx512.cpp PROPERTIES COMPILE_FLAGS /arch:AVX512)
    endif(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
endif()

if(MSVC)
    list(APPEND ALL_SOURCE_FILES internal/_CustomTypes.natvis)
endif(MSVC)

add_library(Ray STATIC ${ALL_SOURCE_FILES})
add_subdirectory(tests)
