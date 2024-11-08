cmake_minimum_required(VERSION 3.18)
project(Ray)

OPTION(ENABLE_REF_IMPL "Enable reference CPU implementation" ON)
OPTION(ENABLE_SIMD_IMPL "Enable SIMD CPU implementations" ON)
if(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
    OPTION(ENABLE_VK_IMPL "Enable Vulkan GPU implementation" ON)
    OPTION(ENABLE_DX_IMPL "Enable DirectX12 GPU implementation" ON)
else(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
    OPTION(ENABLE_VK_IMPL "Enable Vulkan GPU implementation" OFF)
    OPTION(ENABLE_DX_IMPL "Enable DirectX12 GPU implementation" OFF)
endif(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
OPTION(ENABLE_DEBUG_MARKERS "Enable GPU debug markers" OFF)
OPTION(ENABLE_PIX "Enable PIX debug/capture API" OFF)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Asan;Tsan" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /fp:except /WX")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations
                                                -Wno-unsupported-floating-point-opt
                                                -Wno-pragma-clang-attribute
                                                -Wno-microsoft-enum-forward-reference")
    endif()
    if (NOT DEFINED CMAKE_CXX_FLAGS_ASAN OR CMAKE_CXX_FLAGS_ASAN STREQUAL "")
        set(CMAKE_CXX_FLAGS_ASAN "/MTd /Zi /Od /RTC1 /fsanitize=address")
        set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
    endif()
    if (NOT DEFINED CMAKE_CXX_FLAGS_TSAN OR CMAKE_CXX_FLAGS_TSAN STREQUAL "")
        set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}") # no TSAN on windows
        set(CMAKE_EXE_LINKER_FLAGS_TSAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
    endif()
    add_compile_definitions(VK_USE_PLATFORM_WIN32_KHR)
else(MSVC)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Android" AND NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfma -fno-strict-aliasing -Wno-unused-command-line-argument")
        if (NOT DEFINED CMAKE_CXX_FLAGS_ASAN OR CMAKE_CXX_FLAGS_ASAN STREQUAL "")
            set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
            set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
        endif()
        if (NOT DEFINED CMAKE_CXX_FLAGS_TSAN OR CMAKE_CXX_FLAGS_TSAN STREQUAL "")
            set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fPIE -pie")
            set(CMAKE_EXE_LINKER_FLAGS_TSAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fPIE -pie")
        endif()
        if(WIN32)
        else(WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        endif(WIN32)
    endif()
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
        add_compile_definitions(VK_USE_PLATFORM_MACOS_MVK)
    else(APPLE)
        add_compile_definitions(VK_USE_PLATFORM_XLIB_KHR)
    endif(APPLE)
endif(MSVC)

set(INTERNAL_SOURCE_FILES internal/AtmosphereRef.h
                          internal/AtmosphereRef.cpp
                          internal/BVHSplit.h
                          internal/BVHSplit.cpp
                          internal/CDFUtils.h
                          internal/CDFUtils.cpp
                          internal/Constants.inl
                          internal/Convolution.h
                          internal/Core.h
                          internal/Core.cpp
                          internal/CoreDX.h
                          internal/CoreRef.h
                          internal/CoreRef.cpp
                          internal/CoreSIMD.h
                          internal/DenoiseRef.h
                          internal/DenoiseRef.cpp
                          internal/Fixed.h
                          internal/FreelistAlloc.h
                          internal/FreelistAlloc.cpp
                          internal/HashMap32.h
                          internal/RadCacheRef.h
                          internal/RadCacheRef.cpp
                          internal/RastState.h
                          internal/RendererCPU.h
                          internal/RendererGPU.h
                          internal/RendererGPU_kernels.h
                          internal/SamplingParams.h
                          internal/SamplingParams.cpp
                          internal/SceneCommon.h
                          internal/SceneCommon.cpp
                          internal/SceneCPU.h
                          internal/SceneCPU.cpp
                          internal/SceneGPU.h
                          internal/ScopeExit.h
                          internal/ShadeRef.h
                          internal/ShadeRef.cpp
                          internal/SmallVector.h
                          internal/SparseStorageCPU.h
                          internal/SparseStorageGPU.h
                          internal/TextureParams.h
                          internal/TextureParams.cpp
                          internal/TextureSplitter.h
                          internal/TextureSplitter.cpp
                          internal/TextureStorageCPU.h
                          internal/TextureStorageCPU.cpp
                          internal/TextureUtils.h
                          internal/TextureUtils.cpp
                          internal/TextureUtilsNEON.cpp
                          internal/TextureUtilsSSE2.cpp
                          internal/Time_.h
                          internal/Time.cpp
                          internal/TonemapRef.h
                          internal/TonemapRef.cpp
                          internal/UNetFilter.h
                          internal/UNetFilter.cpp
                          internal/VectorGPU.h)

if (ENABLE_REF_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/RendererRef.h
                              internal/RendererRef.cpp
    )
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
                              internal/RendererAVX512.cpp)

    set_source_files_properties(
        internal/RendererSSE2.cpp
        internal/RendererSSE41.cpp
        internal/RendererNEON.cpp
        internal/RendererAVX.cpp
        internal/RendererAVX2.cpp
        internal/RendererAVX512.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
    )
endif(ENABLE_SIMD_IMPL)

if (ENABLE_VK_IMPL)
    add_definitions(-DENABLE_OBJ_LABELS)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/CoreVK.h
                              internal/RendererVK.h
                              internal/RendererVK.cpp
                              internal/SceneVK.h
                              internal/SceneVK.cpp)
endif (ENABLE_VK_IMPL)

if(ENABLE_DX_IMPL AND WIN32)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                                internal/CoreDX.h
                                internal/RendererDX.h
                                internal/RendererDX.cpp
                                internal/SceneDX.h
                                internal/SceneDX.cpp)
    set_source_files_properties(
        internal/RendererDX.cpp
        internal/SceneDX.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
    )
endif(ENABLE_DX_IMPL AND WIN32)

if(NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
    if(MSVC)
        if(NOT CMAKE_CL_64)
            set_source_files_properties(internal/RendererSSE2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
            set_source_files_properties(internal/TextureUtilsSSE2.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
            set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS /arch:SSE2)
        endif()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set_source_files_properties(internal/TextureUtilsSSE2.cpp PROPERTIES COMPILE_FLAGS -mssse3)
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
        set_source_files_properties(internal/RendererAVX512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512dq -mavx512vl")
    endif()
endif(NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/Config.h)

set(SOURCE_FILES Bitmask.h
                 Config.h
                 Log.h
                 Ray.h
                 Ray.cpp
                 RendererBase.h
                 RendererBase.cpp
                 SceneBase.h
                 Span.h
                 Types.h
                 VulkanFunctions.h)

set(SIMD_FILES internal/simd/aligned_allocator.h
               internal/simd/detect.h
               internal/simd/detect.cpp
               internal/simd/simd.h
               internal/simd/simd_sse.h
               internal/simd/simd_avx.h
               internal/simd/simd_avx512.h
               internal/simd/simd_neon.h)

set(VK_SOURCE_FILES internal/Vk/AccStructureVK.h
                    internal/Vk/AccStructureVK.cpp
                    internal/Vk/Api.h
                    internal/Vk/Api.cpp
                    internal/Vk/BufferVK.h
                    internal/Vk/BufferVK.cpp
                    internal/Vk/ContextVK.h
                    internal/Vk/ContextVK.cpp
                    internal/Vk/DebugMarkerVK.h
                    internal/Vk/DescriptorPoolVK.h
                    internal/Vk/DescriptorPoolVK.cpp
                    internal/Vk/DrawCallVK.h
                    internal/Vk/DrawCallVK.cpp
                    internal/Vk/FenceVK.h
                    internal/Vk/FenceVK.cpp
                    internal/Vk/MemoryAllocatorVK.h
                    internal/Vk/MemoryAllocatorVK.cpp
                    internal/Vk/PipelineVK.h
                    internal/Vk/PipelineVK.cpp
                    internal/Vk/ProgramVK.h
                    internal/Vk/ProgramVK.cpp
                    internal/Vk/RenderPassVK.h
                    internal/Vk/RenderPassVK.cpp
                    internal/Vk/ResourceVK.h
                    internal/Vk/ResourceVK.cpp
                    internal/Vk/SamplerVK.h
                    internal/Vk/SamplerVK.cpp
                    internal/Vk/ShaderVK.h
                    internal/Vk/ShaderVK.cpp
                    internal/Vk/SparseStorageVK.h
                    internal/Vk/SPIRV.h
                    internal/Vk/SPIRV.cpp
                    internal/Vk/TextureVK.h
                    internal/Vk/TextureVK.cpp
                    internal/Vk/TextureAtlasVK.h
                    internal/Vk/TextureAtlasVK.cpp
                    internal/Vk/VectorVK.h
                    internal/Vk/VertexInputVK.h
                    internal/Vk/VertexInputVK.cpp)

if(WIN32)
    set(DX_SOURCE_FILES internal/Dx/AccStructureDX.h
                        internal/Dx/AccStructureDX.cpp
                        internal/Dx/BufferDX.h
                        internal/Dx/BufferDX.cpp
                        internal/Dx/ContextDX.h
                        internal/Dx/ContextDX.cpp
                        internal/Dx/DebugMarkerDX.h
                        internal/Dx/DebugMarkerDX.cpp
                        internal/Dx/DescriptorPoolDX.h
                        internal/Dx/DescriptorPoolDX.cpp
                        internal/Dx/DrawCallDX.h
                        internal/Dx/DrawCallDX.cpp
                        internal/Dx/FenceDX.h
                        internal/Dx/FenceDX.cpp
                        internal/Dx/MemoryAllocatorDX.h
                        internal/Dx/MemoryAllocatorDX.cpp
                        internal/Dx/PipelineDX.h
                        internal/Dx/PipelineDX.cpp
                        internal/Dx/ProgramDX.h
                        internal/Dx/ProgramDX.cpp
                        internal/Dx/RenderPassDX.h
                        internal/Dx/RenderPassDX.cpp
                        internal/Dx/ResourceDX.h
                        internal/Dx/ResourceDX.cpp
                        internal/Dx/SamplerDX.h
                        internal/Dx/SamplerDX.cpp
                        internal/Dx/ShaderDX.h
                        internal/Dx/ShaderDX.cpp
                        internal/Dx/SparseStorageDX.h
                        internal/Dx/TextureDX.h
                        internal/Dx/TextureDX.cpp
                        internal/Dx/TextureAtlasDX.h
                        internal/Dx/TextureAtlasDX.cpp
                        internal/Dx/VectorDX.h
                        internal/Dx/VertexInputDX.h
                        internal/Dx/VertexInputDX.cpp)
endif(WIN32)

set(SHADER_FILES    internal/shaders/_interface_common.h
                    internal/shaders/common.glsl
                    internal/shaders/debug_rt.comp.glsl
                    internal/shaders/debug_rt_interface.h
                    internal/shaders/envmap.glsl
                    internal/shaders/filter_variance.comp.glsl
                    internal/shaders/filter_variance_interface.h
                    internal/shaders/intersect.glsl
                    internal/shaders/intersect_area_lights.comp.glsl
                    internal/shaders/intersect_area_lights_interface.h
                    internal/shaders/intersect_scene.comp.glsl
                    internal/shaders/intersect_scene_interface.h
                    internal/shaders/intersect_scene_shadow.comp.glsl
                    internal/shaders/intersect_scene_shadow_interface.h
                    internal/shaders/mix_incremental.comp.glsl
                    internal/shaders/mix_incremental_interface.h
                    internal/shaders/nlm_filter.comp.glsl
                    internal/shaders/nlm_filter_interface.h
                    internal/shaders/postprocess.comp.glsl
                    internal/shaders/postprocess_interface.h
                    internal/shaders/prepare_indir_args.comp.glsl
                    internal/shaders/prepare_indir_args_interface.h
                    internal/shaders/primary_ray_gen.comp.glsl
                    internal/shaders/primary_ray_gen_interface.h
                    internal/shaders/shade.comp.glsl
                    internal/shaders/shade_interface.h
                    internal/shaders/sort_common.h
                    internal/shaders/sort_hash_rays.comp.glsl
                    internal/shaders/sort_hash_rays_interface.h
                    internal/shaders/sort_init_count_table.comp.glsl
                    internal/shaders/sort_init_count_table_interface.h
                    internal/shaders/sort_reduce.comp.glsl
                    internal/shaders/sort_reduce_interface.h
                    internal/shaders/sort_reorder_rays.comp.glsl
                    internal/shaders/sort_reorder_rays_interface.h
                    internal/shaders/sort_scan.comp.glsl
                    internal/shaders/sort_scan_interface.h
                    internal/shaders/sort_scatter.comp.glsl
                    internal/shaders/sort_scatter_interface.h
                    internal/shaders/texture.glsl
                    internal/shaders/traverse_bvh.glsl
                    internal/shaders/types.h)

set(INFLATE_SOURCE_FILES    internal/inflate/Huffman.h
                            internal/inflate/Huffman.cpp
                            internal/inflate/Inflate.h
                            internal/inflate/Inflate.cpp)

list(APPEND ALL_SOURCE_FILES ${INTERNAL_SOURCE_FILES})
source_group("src\\internal" FILES ${INTERNAL_SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${SOURCE_FILES})
source_group("src" FILES ${SOURCE_FILES})

list(APPEND ALL_SOURCE_FILES ${SHADER_FILES})
source_group("src\\internal\\shaders" FILES ${SHADER_FILES})

list(APPEND ALL_SOURCE_FILES ${SIMD_FILES})
source_group("src\\internal\\simd" FILES ${SIMD_FILES})

if(ENABLE_VK_IMPL)
    list(APPEND ALL_SOURCE_FILES ${VK_SOURCE_FILES})
    source_group("src\\internal\\Vk" FILES ${VK_SOURCE_FILES})
endif(ENABLE_VK_IMPL)

if(ENABLE_DX_IMPL AND WIN32)
    list(APPEND ALL_SOURCE_FILES ${DX_SOURCE_FILES})
    source_group("src\\internal\\Dx" FILES ${DX_SOURCE_FILES})
endif(ENABLE_DX_IMPL AND WIN32)

list(APPEND ALL_SOURCE_FILES ${INFLATE_SOURCE_FILES})
source_group("src\\internal\\inflate" FILES ${INFLATE_SOURCE_FILES})

set_source_files_properties(${SHADER_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

if(MSVC)
    list(APPEND ALL_SOURCE_FILES internal/_CustomTypes.natvis)
endif(MSVC)

if (ENABLE_PIX AND WIN32)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pix/include)
endif (ENABLE_PIX AND WIN32)

add_library(Ray STATIC ${ALL_SOURCE_FILES})
if(ENABLE_DX_IMPL AND WIN32)
    target_link_libraries(Ray DXGI D3D12)
    if(ENABLE_PIX)
        if(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
            target_link_libraries(Ray ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pix/ARM64/WinPixEventRuntime.lib)
        else(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
            target_link_libraries(Ray ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pix/x64/WinPixEventRuntime.lib)
        endif(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
    endif(ENABLE_PIX)
endif(ENABLE_DX_IMPL AND WIN32)

set_target_properties(Ray PROPERTIES UNITY_BUILD_MODE GROUP)

set_source_files_properties(${SOURCE_FILES}
                            PROPERTIES UNITY_GROUP "common"
                            )
set_source_files_properties(${INTERNAL_SOURCE_FILES}
                            PROPERTIES UNITY_GROUP "internal"
                            )
set_source_files_properties(${VK_SOURCE_FILES}
                            PROPERTIES UNITY_GROUP "vk"
                            )
set_source_files_properties(${DX_SOURCE_FILES}
                            PROPERTIES UNITY_GROUP "dx"
                            )

add_subdirectory(samples)
add_subdirectory(tests)
