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
                          internal/CoreAVX.h
                          internal/CoreAVX.cpp
                          internal/CoreOCL.h
                          internal/CoreRef.h
                          internal/CoreRef.cpp
                          internal/CoreSSE.h
                          internal/CoreSSE.cpp
                          internal/FramebufferRef.h
                          internal/FramebufferRef.cpp
                          internal/Halton.h
                          internal/RendererAVX.h
                          internal/RendererAVX.cpp
                          internal/RendererOCL.h
                          internal/RendererOCL.cpp
                          internal/RendererRef.h
                          internal/RendererRef.cpp
                          internal/RendererSSE.h
                          internal/RendererSSE.cpp
                          internal/SceneCPU.h
                          internal/SceneCPU.cpp
                          internal/SceneOCL.h
                          internal/SceneOCL.cpp
                          internal/TextureAtlasOCL.h
                          internal/TextureAtlasOCL.cpp
                          internal/TextureSplitter.h
                          internal/TextureSplitter.cpp
                          internal/VectorOCL.h)

set(SOURCE_FILES RendererBase.h
                 RendererFactory.h
                 RendererFactory.cpp
                 SceneBase.h
                 SceneBase.cpp
                 Types.h)

set(KERNEL_FILES internal/kernels/intersect_boxes.cl
                 internal/kernels/intersect_cones.cl
                 internal/kernels/intersect_tris.cl
                 internal/kernels/postprocess.cl
                 internal/kernels/primary_ray_gen.cl
                 internal/kernels/shade.cl
                 internal/kernels/texture.cl
                 internal/kernels/trace.cl
                 internal/kernels/transform.cl
                 internal/kernels/traverse_bvh.cl
                 internal/kernels/types.cl)

list(APPEND ALL_SOURCE_FILES ${INTERNAL_SOURCE_FILES})
set_source_files_properties(${INTERNAL_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("src\\internal" FILES ${INTERNAL_SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${SOURCE_FILES})
set_source_files_properties(${SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("src" FILES ${SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${KERNEL_FILES})
set_source_files_properties(${KERNEL_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("src\\internal\\kernels" FILES ${KERNEL_FILES})

list(APPEND ALL_SOURCE_FILES _ray.cpp)
source_group("src" FILES _ray.cpp)

add_library(ray STATIC ${ALL_SOURCE_FILES})
target_link_libraries(ray OpenCL math)

add_subdirectory(tests)
