cmake_minimum_required(VERSION 3.18)
project(Ray)

OPTION(RAY_ENABLE_REF_IMPL "Enable reference CPU implementation" ON)
OPTION(RAY_ENABLE_SIMD_IMPL "Enable SIMD CPU implementations" ON)
if(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
    OPTION(RAY_ENABLE_VK_IMPL "Enable Vulkan GPU implementation" ON)
    OPTION(RAY_ENABLE_DX_IMPL "Enable DirectX12 GPU implementation" ON)
else(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
    OPTION(RAY_ENABLE_VK_IMPL "Enable Vulkan GPU implementation" OFF)
    OPTION(RAY_ENABLE_DX_IMPL "Enable DirectX12 GPU implementation" OFF)
endif(NOT CMAKE_SYSTEM_NAME MATCHES "Android")
OPTION(RAY_ENABLE_GPU_DEBUG "Enable GPU debugging" OFF)
OPTION(RAY_ENABLE_PIX "Enable PIX debug/capture API" OFF)
OPTION(RAY_ENABLE_ASM_CONV "Use generated asm UNet conv kernels (auto-disabled if unsupported)" ON)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;Asan;Tsan" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 17)
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
        set(CMAKE_CXX_FLAGS_ASAN "/MTd /Zi /Od /RTC1 /fsanitize=address /analyze /WX /wd6993")
        set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
    endif()
    if (NOT DEFINED CMAKE_CXX_FLAGS_TSAN OR CMAKE_CXX_FLAGS_TSAN STREQUAL "")
        set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}") # no TSAN on windows
        set(CMAKE_EXE_LINKER_FLAGS_TSAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
    endif()
    add_compile_definitions(VK_USE_PLATFORM_WIN32_KHR)
else(MSVC)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Android" AND NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64" AND NOT CMAKE_APPLE_SILICON_PROCESSOR STREQUAL "arm64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mno-avx -fno-strict-aliasing -Wno-unused-command-line-argument")
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
                          internal/CompareOp.inl
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
                          internal/Filter.inl
                          internal/Fixed.h
                          internal/Format.inl
                          internal/FreelistAlloc.h
                          internal/FreelistAlloc.cpp
                          internal/HashMap32.h
                          internal/HashSet32.h
                          internal/ImageParams.h
                          internal/ImageParams.cpp
                          internal/ImageSplitter.h
                          internal/ImageSplitter.cpp
                          internal/PMJ.h
                          internal/PMJ.cpp
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
                          internal/Trig.h
                          internal/UNetFilter.h
                          internal/UNetFilter.cpp
                          internal/VectorGPU.h
                          internal/Wrap.inl)

if (RAY_ENABLE_REF_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/RendererRef.h
                              internal/RendererRef.cpp
    )
endif()

if (RAY_ENABLE_SIMD_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
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
        internal/RendererSSE41.cpp
        internal/RendererNEON.cpp
        internal/RendererAVX.cpp
        internal/RendererAVX2.cpp
        internal/RendererAVX512.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
    )
endif(RAY_ENABLE_SIMD_IMPL)

if (RAY_ENABLE_VK_IMPL)
    set(INTERNAL_SOURCE_FILES ${INTERNAL_SOURCE_FILES}
                              internal/CoreVK.h
                              internal/RendererVK.h
                              internal/RendererVK.cpp
                              internal/SceneVK.h
                              internal/SceneVK.cpp)
endif (RAY_ENABLE_VK_IMPL)

if(RAY_ENABLE_DX_IMPL AND WIN32)
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
endif(RAY_ENABLE_DX_IMPL AND WIN32)

if(NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64" AND NOT "${CMAKE_APPLE_SILICON_PROCESSOR}" STREQUAL "arm64")
    if(MSVC)
        if(NOT CMAKE_CL_64)
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
        set_source_files_properties(internal/simd/detect.cpp PROPERTIES COMPILE_FLAGS -mxsave)
        set_source_files_properties(internal/TextureUtilsSSE2.cpp PROPERTIES COMPILE_FLAGS -msse2)
        set_source_files_properties(internal/RendererSSE41.cpp PROPERTIES COMPILE_FLAGS -msse4.1)
        set_source_files_properties(internal/RendererAVX.cpp PROPERTIES COMPILE_FLAGS -mavx)
        set_source_files_properties(internal/RendererAVX2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        set_source_files_properties(internal/RendererAVX512.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl")
    endif()
endif(NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "ARM64" AND NOT "${CMAKE_APPLE_SILICON_PROCESSOR}" STREQUAL "arm64")

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
                    internal/Vk/ImageVK.h
                    internal/Vk/ImageVK.cpp
                    internal/Vk/ImageAtlasVK.h
                    internal/Vk/ImageAtlasVK.cpp
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
                        internal/Dx/ImageDX.h
                        internal/Dx/ImageDX.cpp
                        internal/Dx/ImageAtlasDX.h
                        internal/Dx/ImageAtlasDX.cpp
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
                    internal/shaders/types.h
                    internal/shaders/trig.glsl)

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

if(RAY_ENABLE_VK_IMPL)
    list(APPEND ALL_SOURCE_FILES ${VK_SOURCE_FILES})
    source_group("src\\internal\\Vk" FILES ${VK_SOURCE_FILES})
endif(RAY_ENABLE_VK_IMPL)

if(RAY_ENABLE_DX_IMPL AND WIN32)
    list(APPEND ALL_SOURCE_FILES ${DX_SOURCE_FILES})
    source_group("src\\internal\\Dx" FILES ${DX_SOURCE_FILES})
endif(RAY_ENABLE_DX_IMPL AND WIN32)

list(APPEND ALL_SOURCE_FILES ${INFLATE_SOURCE_FILES})
source_group("src\\internal\\inflate" FILES ${INFLATE_SOURCE_FILES})

set_source_files_properties(${SHADER_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

if(MSVC)
    list(APPEND ALL_SOURCE_FILES internal/_CustomTypes.natvis)
endif(MSVC)

if (RAY_ENABLE_PIX AND WIN32)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pix/include)
endif (RAY_ENABLE_PIX AND WIN32)

add_library(Ray STATIC ${ALL_SOURCE_FILES})

# Generated asm UNet conv kernels (scripts/gen_conv_asm.py) for denoise passes 0-9, 11-15:
# bit-identical to the C++ kernels and ~1.2-1.5x faster. Assembler per platform: MASM .asm
# on MSVC/ClangCL x64, armasm64 .asm on MSVC ARM64 (CMake's ASM_MASM language rejects ARM64,
# so armasm64.exe next to the target cl.exe is run via custom commands and the objects are
# linked into Ray), GAS .s through the C compiler driver on Linux/macOS - x86_64 gets
# avx1/avx2/sse41/avx512, aarch64/arm64 gets neon. .intel_syntax noprefix is accepted by
# GNU as and clang; the macOS files carry no CFI (Mach-O's CFI object writer rejects their
# label-difference advance_loc expressions; these hot kernels are never unwound through -
# same posture as the ml64/armasm64 objects), the Linux files keep CFI and .note.GNU-stack.
# macOS multi-arch (CMAKE_OSX_ARCHITECTURES with several arches) assembles one master TU per
# arch with explicit -arch flags and lipo-merges the objects into one fat object.
# RAY_ENABLE_ASM_CONV defaults to ON; if the platform has no kernel form or the assembler or
# the generated sources are missing, the kernels are disabled and the C++ Convolution3x3
# path is used instead of failing the build.
set(_ray_conv_mode "")
set(_ray_conv_reason "")
set(_ray_conv_missing FALSE)
set(_ray_conv_passes 0 1 2 3 4 5 6 7 8 9 11 12 13 14 15)
if(NOT RAY_ENABLE_ASM_CONV)
    set(_ray_conv_reason "RAY_ENABLE_ASM_CONV=OFF")
elseif(NOT RAY_ENABLE_SIMD_IMPL)
    set(_ray_conv_reason "RAY_ENABLE_SIMD_IMPL=OFF")
elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_ray_conv_reason "32-bit build")
elseif(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    # ml64.exe ships next to the target x64 cl.exe (and in the vcvars PATH); find it
    # explicitly - CMake's generic ASM_MASM probing may return the 32-bit ml.exe
    get_filename_component(_ray_x64_bin_dir ${CMAKE_CXX_COMPILER} DIRECTORY)
    find_program(RAY_ML64 NAMES ml64 ml64.exe HINTS ${_ray_x64_bin_dir})
    if(RAY_ML64)
        set(_ray_conv_mode "masm")
    else()
        set(_ray_conv_reason "MASM assembler (ml64) not found")
        set(_ray_conv_missing TRUE)
    endif()
elseif(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
    get_filename_component(_ray_arm64_bin_dir ${CMAKE_CXX_COMPILER} DIRECTORY)
    find_program(RAY_ARMASM64 NAMES armasm64 armasm64.exe HINTS ${_ray_arm64_bin_dir})
    if(RAY_ARMASM64)
        set(_ray_conv_mode "armasm")
    else()
        set(_ray_conv_reason "armasm64 not found next to ${CMAKE_CXX_COMPILER}")
        set(_ray_conv_missing TRUE)
    endif()
elseif(NOT MSVC AND (APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    if(APPLE AND CMAKE_OSX_ARCHITECTURES AND CMAKE_OSX_ARCHITECTURES MATCHES ";")
        # universal build: assemble one master TU per arch with explicit -arch flags, then
        # lipo-merge into a single fat object (the ASM language rule cannot be reused - the
        # driver would try to assemble every master for every requested arch)
        set(_ray_conv_univ_archs "")  # kernel file arch tag (aarch64 / x86_64)
        set(_ray_conv_univ_flags "")  # clang -arch name (arm64 / x86_64)
        set(_ray_conv_univ_ok TRUE)
        foreach(_ray_conv_osx_arch IN LISTS CMAKE_OSX_ARCHITECTURES)
            if(_ray_conv_osx_arch STREQUAL "arm64" AND NOT "arm64" IN_LIST _ray_conv_univ_flags)
                list(APPEND _ray_conv_univ_flags arm64)
                list(APPEND _ray_conv_univ_archs aarch64)
            elseif(_ray_conv_osx_arch STREQUAL "x86_64" AND NOT "x86_64" IN_LIST _ray_conv_univ_flags)
                list(APPEND _ray_conv_univ_flags x86_64)
                list(APPEND _ray_conv_univ_archs x86_64)
            elseif(_ray_conv_osx_arch IN_LIST _ray_conv_univ_flags)
                # duplicate arch entry, fine
            else()
                set(_ray_conv_univ_ok FALSE)
                set(_ray_conv_reason "no kernels generated for OSX arch '${_ray_conv_osx_arch}'")
            endif()
        endforeach()
        if(_ray_conv_univ_ok)
            set(_ray_conv_mode "gas_universal")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_ray_conv_mode "gas")
        set(_ray_conv_gas_isas "neon")
        set(_ray_conv_gas_arch "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(_ray_conv_mode "gas")
        set(_ray_conv_gas_isas "avx1;avx2;sse41;avx512")
        set(_ray_conv_gas_arch "x86_64")
    else()
        set(_ray_conv_reason "no kernels generated for CPU '${CMAKE_SYSTEM_PROCESSOR}'")
    endif()
    if(_ray_conv_mode STREQUAL "gas")
        if(APPLE)
            set(_ray_conv_gas_sfx "macos")
        else()
            set(_ray_conv_gas_sfx "linux")
        endif()
    endif()
else()
    set(_ray_conv_reason "no kernel form for this platform/compiler")
endif()

# disables the kernels when a required generated file is missing (paths relative to
# this directory or absolute); argn = variable names holding the file lists
macro(_ray_conv_probe_missing)
    foreach(_ray_conv_probe IN LISTS ${ARGN})
        if(NOT IS_ABSOLUTE "${_ray_conv_probe}")
            set(_ray_conv_probe "${CMAKE_CURRENT_SOURCE_DIR}/${_ray_conv_probe}")
        endif()
        if(NOT EXISTS "${_ray_conv_probe}")
            set(_ray_conv_reason "missing ${_ray_conv_probe} - run scripts/gen_conv_asm.py from src/Ray")
            set(_ray_conv_missing TRUE)
            break()
        endif()
    endforeach()
endmacro()

if(_ray_conv_mode STREQUAL "masm")
    # one master TU (INCLUDE) compiles to a single object; the per-kernel .asm files stay
    # separate (browse-only here), the master supplies the END directive
    set(_ray_conv_asm_srcs internal/generated/conv_all_win_x64.asm)
    set(_ray_conv_asm_kernels "")
    set(_ray_conv_asm_aux "")
    foreach(_ray_conv_pass IN LISTS _ray_conv_passes)
        list(APPEND _ray_conv_asm_kernels internal/generated/conv${_ray_conv_pass}_avx2.asm
            internal/generated/conv${_ray_conv_pass}_avx1.asm internal/generated/conv${_ray_conv_pass}_sse41.asm
            internal/generated/conv${_ray_conv_pass}_avx512.asm)
        list(APPEND _ray_conv_asm_aux internal/generated/conv${_ray_conv_pass}_avx2_linux.s
            internal/generated/conv${_ray_conv_pass}_avx2_macos.s
            internal/generated/conv${_ray_conv_pass}_avx1_linux.s
            internal/generated/conv${_ray_conv_pass}_avx1_macos.s
            internal/generated/conv${_ray_conv_pass}_sse41_linux.s
            internal/generated/conv${_ray_conv_pass}_sse41_macos.s
            internal/generated/conv${_ray_conv_pass}_avx512_linux.s
            internal/generated/conv${_ray_conv_pass}_avx512_macos.s)
    endforeach()
    list(APPEND _ray_conv_asm_aux internal/generated/conv_all_linux_x86_64.s internal/generated/conv_all_macos_x86_64.s
        internal/generated/conv_all_linux_aarch64.s internal/generated/conv_all_macos_aarch64.s)
    _ray_conv_probe_missing(_ray_conv_asm_srcs _ray_conv_asm_kernels)
    if(NOT _ray_conv_missing)
        # pin the assembler (also overrides stale generic detections in the cache)
        set(CMAKE_ASM_MASM_COMPILER "${RAY_ML64}" CACHE STRING "MASM assembler" FORCE)
        enable_language(ASM_MASM)
        # INCLUDE file names resolve via /I (ml64 does not look next to the including file)
        set_source_files_properties(${_ray_conv_asm_srcs} PROPERTIES COMPILE_OPTIONS
            "/I${CMAKE_CURRENT_SOURCE_DIR}/internal/generated")
        target_sources(Ray PRIVATE ${_ray_conv_asm_srcs} ${_ray_conv_asm_aux} ${_ray_conv_asm_kernels})
        # per-kernel asm / GAS sources are browse-only (assembled through the master TU)
        set_source_files_properties(${_ray_conv_asm_aux} ${_ray_conv_asm_kernels} PROPERTIES HEADER_FILE_ONLY TRUE)
        source_group("src\\internal\\generated" FILES ${_ray_conv_asm_srcs} ${_ray_conv_asm_aux} ${_ray_conv_asm_kernels})
    endif()
elseif(_ray_conv_mode STREQUAL "armasm")
    # armasm64 GET includes are per-unit (each kernel file ends with its own indented END, a
    # mid-stream END does not stop the rest, unlike ml64): the master TU assembles all kernels
    # into one object, GET names resolve via /I.
    set(_ray_conv_arm_master internal/generated/conv_all_win_arm64.asm)
    set(_ray_conv_arm_srcs "")
    set(_ray_conv_asm_aux "")
    foreach(_ray_conv_pass IN LISTS _ray_conv_passes)
        list(APPEND _ray_conv_arm_srcs internal/generated/conv${_ray_conv_pass}_neon.asm)
        list(APPEND _ray_conv_asm_aux internal/generated/conv${_ray_conv_pass}_neon_linux.s
            internal/generated/conv${_ray_conv_pass}_neon_macos.s)
    endforeach()
    list(APPEND _ray_conv_asm_aux internal/generated/conv_all_linux_aarch64.s internal/generated/conv_all_macos_aarch64.s)
    _ray_conv_probe_missing(_ray_conv_arm_master _ray_conv_arm_srcs)
    if(NOT _ray_conv_missing)
        set(_ray_conv_arm_obj ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_win_arm64.obj)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_asm)
        add_custom_command(OUTPUT ${_ray_conv_arm_obj}
                COMMAND ${RAY_ARMASM64} /nologo /I${CMAKE_CURRENT_SOURCE_DIR}/internal/generated
                        /o ${_ray_conv_arm_obj}
                        ${CMAKE_CURRENT_SOURCE_DIR}/internal/generated/conv_all_win_arm64.asm
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/internal/generated/conv_all_win_arm64.asm
                        ${_ray_conv_arm_srcs}
                COMMENT "Assembling conv_all_win_arm64.asm (armasm64, GET master)")
        target_sources(Ray PRIVATE ${_ray_conv_arm_obj})
        target_sources(Ray PRIVATE ${_ray_conv_arm_master} ${_ray_conv_arm_srcs} ${_ray_conv_asm_aux})
        # per-kernel sources are browse-only (assembled through the GET master)
        set_source_files_properties(${_ray_conv_arm_master} ${_ray_conv_arm_srcs} ${_ray_conv_asm_aux}
            PROPERTIES HEADER_FILE_ONLY TRUE)
        source_group("src\\internal\\generated" FILES ${_ray_conv_arm_master} ${_ray_conv_arm_srcs} ${_ray_conv_asm_aux})
    endif()
elseif(_ray_conv_mode STREQUAL "gas")
    # one master TU (.include) compiles to a single object; the per-kernel .s files stay
    # separate (browse-only)
    set(_ray_conv_gas_master internal/generated/conv_all_${_ray_conv_gas_sfx}_${_ray_conv_gas_arch}.s)
    set(_ray_conv_gas_srcs "")
    foreach(_ray_conv_gas_isa IN LISTS _ray_conv_gas_isas)
        foreach(_ray_conv_pass IN LISTS _ray_conv_passes)
            list(APPEND _ray_conv_gas_srcs internal/generated/conv${_ray_conv_pass}_${_ray_conv_gas_isa}_${_ray_conv_gas_sfx}.s)
        endforeach()
    endforeach()
    _ray_conv_probe_missing(_ray_conv_gas_master _ray_conv_gas_srcs)
    if(NOT _ray_conv_missing)
        enable_language(ASM)
        # .include names resolve via -I (GAS does not look next to the including file)
        set_source_files_properties(${_ray_conv_gas_master} PROPERTIES COMPILE_OPTIONS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/internal/generated")
        target_sources(Ray PRIVATE ${_ray_conv_gas_master} ${_ray_conv_gas_srcs})
        # per-kernel sources are browse-only (assembled through the master TU)
        set_source_files_properties(${_ray_conv_gas_srcs} PROPERTIES HEADER_FILE_ONLY TRUE)
        source_group("src\\internal\\generated" FILES ${_ray_conv_gas_master} ${_ray_conv_gas_srcs})
    endif()
elseif(_ray_conv_mode STREQUAL "gas_universal")
    # per-arch master TUs assembled with explicit -arch, lipo-merged into one fat object
    set(_ray_conv_gas_masters "")
    set(_ray_conv_gas_srcs "")
    foreach(_ray_conv_gas_arch IN LISTS _ray_conv_univ_archs)
        list(APPEND _ray_conv_gas_masters internal/generated/conv_all_macos_${_ray_conv_gas_arch}.s)
        if(_ray_conv_gas_arch STREQUAL "aarch64")
            set(_ray_conv_arch_isas "neon")
        else()
            set(_ray_conv_arch_isas "avx1;avx2;sse41;avx512")
        endif()
        set(_ray_conv_arch_srcs "")
        foreach(_ray_conv_gas_isa IN LISTS _ray_conv_arch_isas)
            foreach(_ray_conv_pass IN LISTS _ray_conv_passes)
                list(APPEND _ray_conv_arch_srcs internal/generated/conv${_ray_conv_pass}_${_ray_conv_gas_isa}_macos.s)
            endforeach()
        endforeach()
        set(_ray_conv_arch_srcs_${_ray_conv_gas_arch} ${_ray_conv_arch_srcs})
        list(APPEND _ray_conv_gas_srcs ${_ray_conv_arch_srcs})
    endforeach()
    _ray_conv_probe_missing(_ray_conv_gas_masters _ray_conv_gas_srcs)
    if(NOT _ray_conv_missing)
        find_program(RAY_LIPO NAMES lipo)
        if(NOT RAY_LIPO)
            set(_ray_conv_reason "lipo not found (required for multi-arch conv objects)")
            set(_ray_conv_missing TRUE)
        endif()
    endif()
    if(NOT _ray_conv_missing)
        enable_language(ASM)
        # clang -cc1as does not create the output directory
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_asm)
        if(CMAKE_OSX_DEPLOYMENT_TARGET)
            set(_ray_conv_minversion "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
        else()
            set(_ray_conv_minversion "")
        endif()
        set(_ray_conv_arch_objs "")
        list(LENGTH _ray_conv_univ_archs _ray_conv_univ_count)
        math(EXPR _ray_conv_univ_last "${_ray_conv_univ_count} - 1")
        foreach(_ray_conv_univ_i RANGE 0 ${_ray_conv_univ_last})
            list(GET _ray_conv_univ_archs ${_ray_conv_univ_i} _ray_conv_gas_arch)
            list(GET _ray_conv_univ_flags ${_ray_conv_univ_i} _ray_conv_arch_flag)
            add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_macos_${_ray_conv_gas_arch}.o
                    COMMAND ${CMAKE_ASM_COMPILER} -arch ${_ray_conv_arch_flag} -c
                            -I${CMAKE_CURRENT_SOURCE_DIR}/internal/generated ${_ray_conv_minversion}
                            ${CMAKE_CURRENT_SOURCE_DIR}/internal/generated/conv_all_macos_${_ray_conv_gas_arch}.s
                            -o ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_macos_${_ray_conv_gas_arch}.o
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/internal/generated/conv_all_macos_${_ray_conv_gas_arch}.s
                            ${_ray_conv_arch_srcs_${_ray_conv_gas_arch}}
                    COMMENT "Assembling conv_all_macos_${_ray_conv_gas_arch}.s (-arch ${_ray_conv_arch_flag})")
            list(APPEND _ray_conv_arch_objs ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_macos_${_ray_conv_gas_arch}.o)
        endforeach()
        if(_ray_conv_univ_count GREATER 1)
            add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_macos_universal.o
                    COMMAND ${RAY_LIPO} -create ${_ray_conv_arch_objs}
                            -output ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_macos_universal.o
                    DEPENDS ${_ray_conv_arch_objs}
                    COMMENT "Creating fat conv kernel object (lipo)")
            set(_ray_conv_obj_sources ${CMAKE_CURRENT_BINARY_DIR}/generated_asm/conv_all_macos_universal.o)
        else()
            set(_ray_conv_obj_sources ${_ray_conv_arch_objs})
        endif()
        target_sources(Ray PRIVATE ${_ray_conv_obj_sources})
        target_sources(Ray PRIVATE ${_ray_conv_gas_masters} ${_ray_conv_gas_srcs})
        # per-kernel sources are browse-only (assembled through the master TUs)
        set_source_files_properties(${_ray_conv_gas_masters} ${_ray_conv_gas_srcs} PROPERTIES HEADER_FILE_ONLY TRUE)
        source_group("src\\internal\\generated" FILES ${_ray_conv_gas_masters} ${_ray_conv_gas_srcs})
    endif()
endif()

if(_ray_conv_missing)
    message(WARNING "Ray: asm UNet conv kernels DISABLED, falling back to C++ kernels: ${_ray_conv_reason}")
elseif(NOT _ray_conv_mode)
    message(STATUS "Ray: asm UNet conv kernels disabled: ${_ray_conv_reason}")
else()
    message(STATUS "Ray: asm UNet conv kernels enabled (${_ray_conv_mode})")
    target_compile_definitions(Ray PRIVATE RAY_ENABLE_ASM_CONV)
endif()
if(RAY_ENABLE_DX_IMPL AND WIN32)
    target_link_libraries(Ray DXGI D3D12)
    if(RAY_ENABLE_PIX)
        if(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
            target_link_libraries(Ray ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pix/ARM64/WinPixEventRuntime.lib)
        else(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
            target_link_libraries(Ray ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pix/x64/WinPixEventRuntime.lib)
        endif(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
    endif(RAY_ENABLE_PIX)
endif(RAY_ENABLE_DX_IMPL AND WIN32)

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
