cmake_minimum_required(VERSION 2.8.12)
project(Ray)

OPTION(ENABLE_UNITYBUILD "Enable compilation of one large cpp file" ON)
OPTION(ENABLE_OPENCL "Enables OpenCL backend" OFF)

if(ENABLE_OPENCL)
    add_library(OpenCL STATIC IMPORTED)
    include_directories(ocl/include)
else()
    add_definitions(-DDISABLE_OCL)
endif()

IF(MSVC)
    if(ENABLE_OPENCL)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set_target_properties(OpenCL PROPERTIES
              IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/ocl/lib/x86_64/opencl.lib"
            )
        else(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set_target_properties(OpenCL PROPERTIES
              IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/ocl/lib/x86/opencl.lib"
            )
        endif()
    endif()
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
                          internal/deprecated/CoreSIMD.h
                          internal/FramebufferRef.h
                          internal/FramebufferRef.cpp
                          internal/Halton.h
                          internal/RendererRef.h
                          internal/RendererRef.cpp
                          internal/deprecated/RendererSIMD.h
                          internal/SceneRef.h
                          internal/SceneRef.cpp
                          internal/SparseStorage.h
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
                          internal/deprecated/RendererAVX.h
                          internal/deprecated/RendererAVX.cpp
                          internal/deprecated/RendererAVX2.h
                          internal/deprecated/RendererAVX2.cpp
                          internal/deprecated/RendererSSE2.h
                          internal/deprecated/RendererSSE2.cpp)

    if (ENABLE_OPENCL)
        set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                          internal/deprecated/CoreOCL.h
                          internal/deprecated/RendererOCL.h
                          internal/deprecated/RendererOCL.cpp
                          internal/deprecated/SceneOCL.h
                          internal/deprecated/SceneOCL.cpp
                          internal/deprecated/TextureAtlasOCL.h
                          internal/deprecated/TextureAtlasOCL.cpp
                          internal/deprecated/VectorOCL.h)
    endif()
endif()

if("${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                          internal/deprecated/RendererNEON.h
                          internal/deprecated/RendererNEON.cpp)
endif()

if(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
    if(NOT CMAKE_CL_64)
        set_source_files_properties(internal/RendererSSE.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
    endif()
    set_source_files_properties(internal/RendererAVX.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
endif(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")

set(SOURCE_FILES Log.h
                 RendererBase.h
                 RendererBase.cpp
                 RendererFactory.h
                 RendererFactory.cpp
                 SceneBase.h
                 SceneBase.cpp
                 Types.h)

if (ENABLE_OPENCL)
    set(KERNEL_FILES internal/kernels/intersect.cl
                     internal/kernels/postprocess.cl
                     internal/kernels/primary_ray_gen.cl
                     internal/kernels/sh.cl
                     internal/kernels/shade.cl
                     internal/kernels/sort.cl
                     internal/kernels/texture.cl
                     internal/kernels/trace.cl
                     internal/kernels/transform.cl
                     internal/kernels/traverse_bvh.cl
                     internal/kernels/types.cl)
endif()

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
    set_source_files_properties(${SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

    list(APPEND ALL_SOURCE_FILES _Ray.cpp)
    source_group("src" FILES _Ray.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Sse2.cpp)
    source_group("src" FILES _Ray_Sse2.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Avx.cpp)
    source_group("src" FILES _Ray_Avx.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Avx2.cpp)
    source_group("src" FILES _Ray_Avx2.cpp)

    list(APPEND ALL_SOURCE_FILES _Ray_Avx512.cpp)
    source_group("src" FILES _Ray_Avx512.cpp)

    if(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
        if(NOT CMAKE_CL_64)
            set_source_files_properties(_Ray_Sse2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
        endif()
        set_source_files_properties(_Ray_Avx.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
        set_source_files_properties(_Ray_Avx2.cpp PROPERTIES COMPILE_FLAGS /arch:AVX2)
        set_source_files_properties(_Ray_Avx512.cpp PROPERTIES COMPILE_FLAGS /arch:AVX512)
    endif(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
endif()

add_library(Ray STATIC ${ALL_SOURCE_FILES})
if(ENABLE_OPENCL)
    target_link_libraries(Ray OpenCL)
endif()

add_subdirectory(tests)
