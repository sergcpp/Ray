cmake_minimum_required(VERSION 2.8.12)
project(Ray)

OPTION(ENABLE_REF_IMPL "Enable reference CPU implementation" ON)
OPTION(ENABLE_SIMD_IMPL "Enable SIMD CPU implementations" ON)
IF (NOT CMAKE_SYSTEM_NAME MATCHES "Android")
    OPTION(ENABLE_GPU_IMPL "Enable Vulkan GPU implementation" ON)
ELSE()
    OPTION(ENABLE_GPU_IMPL "Enable Vulkan GPU implementation" OFF)
ENDIF()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Asan" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /O1 /fp:except")
    if (NOT DEFINED CMAKE_CXX_FLAGS_ASAN)
        set(CMAKE_CXX_FLAGS_ASAN "/MTd /Zi /Od /RTC1 /fsanitize=address")
        set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
    endif()
    add_compile_definitions(VK_USE_PLATFORM_WIN32_KHR)
ELSE(MSVC)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Android" AND NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfma -fno-strict-aliasing")
        IF(WIN32)
        ELSE(WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        ENDIF(WIN32)
    endif()
    if(APPLE)
        add_compile_definitions(VK_USE_PLATFORM_MACOS_MVK)
    else(APPLE)
        add_compile_definitions(VK_USE_PLATFORM_XLIB_KHR)
    endif(APPLE)
ENDIF(MSVC)

set(INTERNAL_SOURCE_FILES internal/Bitmap.h
                          internal/BVHSplit.h
                          internal/BVHSplit.cpp
                          internal/Core.h
                          internal/Core.cpp
                          internal/CoreRef.h
                          internal/CoreRef.cpp
                          internal/CoreSIMD.h
                          internal/FramebufferRef.h
                          internal/FramebufferRef.cpp
                          internal/Halton.h
                          internal/RendererSIMD.h
                          internal/SceneRef.h
                          internal/SceneRef.cpp
                          internal/SmallVector.h
                          internal/SparseStorage.h
                          internal/Span.h
                          internal/TextureSplitter.h
                          internal/TextureSplitter.cpp
                          internal/TextureStorageRef.h
                          internal/TextureStorageRef.cpp
                          internal/TextureUtilsRef.h
                          internal/TextureUtilsRef.cpp
                          internal/Time_.h
                          internal/Time.cpp
                          internal/UniformIntDistribution.h
                          internal/Utils.h
                          internal/Utils.cpp)

if (ENABLE_REF_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/RendererRef.h
                              internal/RendererRef.cpp)
endif()

if (ENABLE_SIMD_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/RendererSSE2.h
                              internal/RendererSSE2.cpp
                              internal/RendererSSE41.h
                              internal/RendererSSE41.cpp
                              internal/RendererNEON.h
                              internal/RendererNEON.cpp
                              internal/RendererAVX.h
                              internal/RendererAVX.cpp
                              internal/RendererAVX2.h
                              internal/RendererAVX2.cpp
                              internal/RendererAVX512.h
                              internal/RendererAVX512.cpp
                              internal/UtilsSSE2.cpp)
endif(ENABLE_SIMD_IMPL)

if (ENABLE_GPU_IMPL)
    add_definitions(-DENABLE_OBJ_LABELS)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/CoreVK.h
                              internal/RendererVK.h
                              internal/RendererVK.cpp
                              internal/RendererVK_kernels.cpp
                              internal/SceneVK.h
                              internal/SceneVK.cpp
                              internal/SparseStorageVK.h
                              internal/VectorVK.h)
endif()

if(NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
    if(MSVC)
        if(NOT CMAKE_CL_64)
            set_source_files_properties(internal/RendererSSE2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
            set_source_files_properties(internal/UtilsSSE2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
            set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
        endif()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set_source_files_properties(internal/UtilsSSE2.cpp PROPERTIES COMPILE_FLAGS -mssse3)
            set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS -msse4.1)
        endif()
        set_source_files_properties(internal/RendererAVX.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
        set_source_files_properties(internal/RendererAVX2.cpp PROPERTIES COMPILE_FLAGS /arch:AVX2)
        set_source_files_properties(internal/RendererAVX512.cpp PROPERTIES COMPILE_FLAGS /arch:AVX512)
    else()
        set_source_files_properties(internal/RendererSSE2.cpp PROPERTIES COMPILE_FLAGS -msse2)
        set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS -msse4.1)
        set_source_files_properties(internal/RendererAVX.cpp PROPERTIES COMPILE_FLAGS -mavx)
        set_source_files_properties(internal/RendererAVX2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        set_source_files_properties(internal/RendererAVX512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq")
    endif()
endif(NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/Config.h)

set(SOURCE_FILES Config.h
                 Log.h
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

set(VK_FILES internal/Vk/AccStructure.h
             internal/Vk/AccStructure.cpp
             internal/Vk/Buffer.h
             internal/Vk/Buffer.cpp
             internal/Vk/Context.h
             internal/Vk/Context.cpp
             internal/Vk/DebugMarker.h
             internal/Vk/DescriptorPool.h
             internal/Vk/DescriptorPool.cpp
             internal/Vk/DrawCall.h
             internal/Vk/DrawCall.cpp
             internal/Vk/Fence.h
             internal/Vk/Fence.cpp
             internal/Vk/Fixed.h
             internal/Vk/LinearAlloc.h
             internal/Vk/LinearAlloc.cpp
             internal/Vk/MemoryAllocator.h
             internal/Vk/MemoryAllocator.cpp
             internal/Vk/Pipeline.h
             internal/Vk/Pipeline.cpp
             internal/Vk/Program.h
             internal/Vk/Program.cpp
             internal/Vk/RastState.h
             internal/Vk/RenderPass.h
             internal/Vk/RenderPass.cpp
             internal/Vk/Resource.h
             internal/Vk/Resource.cpp
             internal/Vk/Sampler.h
             internal/Vk/Sampler.cpp
             internal/Vk/SamplingParams.h
             internal/Vk/Shader.h
             internal/Vk/Shader.cpp
             internal/Vk/Texture.h
             internal/Vk/Texture.cpp
             internal/Vk/TextureAtlas.h
             internal/Vk/TextureAtlas.cpp
             internal/Vk/TextureParams.h
             internal/Vk/Utils.h
             internal/Vk/Utils.cpp
             internal/Vk/VertexInput.h
             internal/Vk/VertexInput.cpp
             internal/Vk/VK.h
             internal/Vk/VKExt.cpp
             third-party/SPIRV-Reflect/spirv_reflect.h
             third-party/SPIRV-Reflect/spirv_reflect.c)

set(SHADER_FILES    internal/shaders/_interface_common.h
                    internal/shaders/common.glsl
                    internal/shaders/debug_rt.comp.glsl
                    internal/shaders/envmap.glsl
                    internal/shaders/intersect.glsl
                    internal/shaders/intersect_area_lights.comp.glsl
                    internal/shaders/intersect_area_lights_interface.h
                    internal/shaders/intersect_scene.comp.glsl
                    internal/shaders/intersect_scene_interface.h
                    internal/shaders/intersect_scene_shadow.comp.glsl
                    internal/shaders/intersect_scene_shadow_interface.h
                    internal/shaders/mix_incremental.comp.glsl
                    internal/shaders/mix_incremental_interface.h
                    internal/shaders/postprocess.comp.glsl
                    internal/shaders/postprocess_interface.h
                    internal/shaders/prepare_indir_args.comp.glsl
                    internal/shaders/prepare_indir_args_interface.h
                    internal/shaders/primary_ray_gen.comp.glsl
                    internal/shaders/primary_ray_gen_interface.h
                    internal/shaders/shade.comp.glsl
                    internal/shaders/shade_interface.h
                    internal/shaders/texture.glsl
                    internal/shaders/traverse_bvh.glsl
                    internal/shaders/types.h)

list(APPEND ALL_SOURCE_FILES ${INTERNAL_SOURCE_FILES})
source_group("src\\internal" FILES ${INTERNAL_SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${SOURCE_FILES})
source_group("src" FILES ${SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${SHADER_FILES})
source_group("src\\internal\\shaders" FILES ${SHADER_FILES})

list(APPEND ALL_SOURCE_FILES ${SIMD_FILES})
source_group("src\\internal\\simd" FILES ${SIMD_FILES})

list(APPEND ALL_SOURCE_FILES ${VK_FILES})
source_group("src\\internal\\Vk" FILES ${VK_FILES})

set_source_files_properties(${SHADER_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

if(MSVC)
    list(APPEND ALL_SOURCE_FILES internal/_CustomTypes.natvis)
endif(MSVC)

add_library(Ray STATIC ${ALL_SOURCE_FILES})

add_subdirectory(tests)
