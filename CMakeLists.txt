cmake_minimum_required(VERSION 2.8.4)
project(ray)

IF(MSVC)

ELSE(MSVC)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse3")
	IF(WIN32)
    ELSE(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(WIN32)
ENDIF(MSVC)

set(INTERNAL_SOURCE_FILES internal/BVHSplit.h
                          internal/BVHSplit.cpp
                          internal/Core.h
                          internal/Core.cpp
                          internal/CoreOCL.h
                          internal/CoreRef.h
                          internal/CoreRef.cpp
                          internal/CoreSIMD.h
                          internal/FramebufferRef.h
                          internal/FramebufferRef.cpp
                          internal/Halton.h
                          internal/RendererAVX.h
                          internal/RendererOCL.h
                          internal/RendererOCL.cpp
                          internal/RendererRef.h
                          internal/RendererRef.cpp
                          internal/RendererRef2.h
                          internal/RendererSIMD.h
                          internal/RendererSSE.h
                          internal/SceneOCL.h
                          internal/SceneOCL.cpp
                          internal/SceneRef.h
                          internal/SceneRef.cpp
                          internal/TextureAtlasOCL.h
                          internal/TextureAtlasOCL.cpp
                          internal/TextureAtlasRef.h
                          internal/TextureAtlasRef.cpp
                          internal/TextureSplitter.h
                          internal/TextureSplitter.cpp
                          internal/TextureUtilsRef.h
                          internal/TextureUtilsRef.cpp
                          internal/VectorOCL.h)

set(SOURCE_FILES RendererBase.h
                 RendererFactory.h
                 RendererFactory.cpp
                 SceneBase.h
                 SceneBase.cpp
                 Types.h)

set(KERNEL_FILES internal/kernels/intersect.cl
                 internal/kernels/postprocess.cl
                 internal/kernels/primary_ray_gen.cl
                 internal/kernels/shade.cl
                 internal/kernels/texture.cl
                 internal/kernels/trace.cl
                 internal/kernels/transform.cl
                 internal/kernels/traverse_bvh.cl
                 internal/kernels/types.cl)

set(SIMD_FILES internal/simd/aligned_allocator.h
               internal/simd/detect.h
               internal/simd/simd_vec.h
               internal/simd/simd_vec_sse.h
               internal/simd/simd_vec_avx.h)

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
IF(1)
    set_source_files_properties(${INTERNAL_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

    list(APPEND ALL_SOURCE_FILES _ray.cpp)
    source_group("src" FILES _ray.cpp)
ENDIF()

add_library(ray STATIC ${ALL_SOURCE_FILES})
target_link_libraries(ray OpenCL)

add_subdirectory(tests)
