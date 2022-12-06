cmake_minimum_required(VERSION 2.8.12)
project(Ray)

OPTION(ENABLE_UNITYBUILD "Enable compilation of one large cpp file" ON)
OPTION(ENABLE_REF_IMPL "Enable reference CPU implementation" OFF)
OPTION(ENABLE_SIMD_IMPL "Enable SIMD CPU implementations" ON)
OPTION(ENABLE_GPU_IMPL "Enable Vulkan GPU implementation" ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Asan" CACHE STRING "" FORCE)

IF(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Ob1 /fp:except")
    set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_DEBUG} /fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
ELSE(MSVC)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Android" AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64" AND
       NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse2 -mfma")
        IF(WIN32)
        ELSE(WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        ENDIF(WIN32)
    endif()
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

if(NOT CMAKE_SYSTEM_NAME MATCHES "Android" AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64")
if (ENABLE_SIMD_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/RendererSSE2.h
                              internal/RendererSSE2.cpp
                              internal/RendererSSE41.h
                              internal/RendererSSE41.cpp
                              internal/RendererAVX.h
                              internal/RendererAVX.cpp
                              internal/RendererAVX2.h
                              internal/RendererAVX2.cpp
                              internal/UtilsSSE2.cpp)
endif()


elseif("${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")

if (ENABLE_SIMD_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/RendererNEON.h
                              internal/RendererNEON.cpp)
endif()


endif()

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

if(MSVC AND NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    if(NOT CMAKE_CL_64)
        set_source_files_properties(internal/RendererSSE2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
        set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
    endif()
    set_source_files_properties(internal/RendererAVX.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
    set_source_files_properties(internal/RendererAVX2.cpp PROPERTIES COMPILE_FLAGS /arch:AVX)
endif(MSVC AND NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")

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
                    internal/shaders/mix_incremental.comp.glsl
                    internal/shaders/mix_incremental_interface.h
                    internal/shaders/postprocess.comp.glsl
                    internal/shaders/postprocess_interface.h
                    internal/shaders/prepare_indir_args.comp.glsl
                    internal/shaders/prepare_indir_args_interface.h
                    internal/shaders/primary_ray_gen.comp.glsl
                    internal/shaders/primary_ray_gen_interface.h
                    internal/shaders/shade_hits.comp.glsl
                    internal/shaders/shade_hits_interface.h
                    internal/shaders/texture.glsl
                    internal/shaders/trace_rays.comp.glsl
                    internal/shaders/trace_rays_interface.h
                    internal/shaders/trace_shadow.comp.glsl
                    internal/shaders/trace_shadow_interface.h
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

# enable or disable unity build
if(ENABLE_UNITYBUILD)
    set_source_files_properties(${INTERNAL_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${INTERNAL_DEPRECATED_SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${SOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${VK_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

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

    list(APPEND ALL_SOURCE_FILES _Ray_Vk.cpp)
    source_group("src" FILES _Ray_Vk.cpp)

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
