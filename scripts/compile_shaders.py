import binascii
import os
import sys
import subprocess
import threading
import zlib

ENABLE_MULTIPLE_THREADS = True
ENABLE_SPIRV_OPTIMIZATION = True
ENABLE_COMPRESSION = True
ENABLE_DXC_COMPILATION = (os.name == "nt")
ENABLE_GLSL_REWRITE = False
ENABLE_SPIRV_CROSS = False

mutex = threading.Lock()

def spirv_base_path():
    if sys.platform.startswith("linux"):
        return os.path.join("third-party", "spirv", "linux")
    elif sys.platform == "darwin":
        return os.path.join("third-party", "spirv", "macos")
    elif os.name == "nt":
        return os.path.join("third-party", "spirv", "win32")

def dxc_base_path():
    if sys.platform.startswith("linux"):
        return os.path.join("third-party", "dxc", "linux")
    elif os.name == "nt":
        return os.path.join("third-party", "dxc", "win32")

def glslx_base_path():
    if sys.platform.startswith("linux"):
        return os.path.join("first-party", "glslx", "linux")
    elif sys.platform == "darwin":
        return os.path.join("first-party", "glslx", "macos")
    elif os.name == "nt":
        return os.path.join("first-party", "glslx", "win32")

def make_sublist_group(lst: list, grp: int) -> list:
    return [lst[i:i+grp] for i in range(0, len(lst), grp)]

def do_conversion(content: bytes) -> str:
    hexstr = binascii.hexlify(content).decode("UTF-8")
    hexstr = hexstr.upper()
    array = ["0x" + hexstr[i:i + 2] + "" for i in range(0, len(hexstr), 2)]
    array = make_sublist_group(array, 1000000)
    ret = "\n    ".join([", ".join(e) + "," for e in array])
    return ret[0:len(ret) - 1]

def bin2header(data, file_name):
    file_name = file_name.replace(os.sep, '/')
    if ENABLE_COMPRESSION:
        data = zlib.compress(data, level=9)
    ret = "/* Contents of file " + file_name + " */\n"
    ret += "const int " + file_name.replace('/', '_').replace('.', '_') + "_size = " + str(len(data)) + ";\n"
    ret += "const unsigned char " + file_name.replace('/', '_').replace('.', '_') + "[" + str(len(data)) + "] = {\n    "
    ret += do_conversion(data)
    ret += "\n};\n"
    return ret

def compile_shader(src_name, spv_name=None, glsl_version=None, target_env="spirv1.3", defines="", hlsl_profile="cs_6_0"):
    if  spv_name is None:
        spv_name = src_name[0:-4] + "spv"

    hlsl_name = spv_name[0:-3] + "hlsl"
    cso_name = spv_name[0:-3] + "cso"

    rewrite_result = None
    if ENABLE_GLSL_REWRITE and hlsl_profile is not None:
        rewrite_name = spv_name[0:-3] + "glsl"
        rewrite_cmd = os.path.join(glslx_base_path(), "glslx -DVULKAN=1 -i internal/shaders/" + src_name + " " +
                                                  defines +" -o internal/shaders/output/" + rewrite_name)
        rewrite_result = subprocess.run(rewrite_cmd, shell=True, capture_output=True, text=True, check=False)

        compile_cmd = os.path.join(spirv_base_path(), "glslang -V --target-env " + target_env + " internal/shaders/output/" + rewrite_name + " " +
                                                    defines +" -o internal/shaders/output/" + spv_name)
    else:
        compile_cmd = os.path.join(spirv_base_path(), "glslang -V --target-env " + target_env + " internal/shaders/" + src_name + " " +
                                                    defines +" -o internal/shaders/output/" + spv_name)
    if glsl_version is not None:
        compile_cmd += " --glsl-version " + glsl_version
    compile_result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True, check=False)
    spirv_opt_result = None
    hlsl_cross_result = None
    dxc_result = None
    if ENABLE_SPIRV_OPTIMIZATION:
        if os.name == "nt":
            spirv_opt_result = subprocess.run(os.path.join(spirv_base_path(), "spirv-opt.bat internal/shaders/output/" + spv_name +
                                                                              " -o internal/shaders/output/" + spv_name),
                                              shell=True, capture_output=True, text=True, check=False)
        else:
            spirv_opt_result = subprocess.run(os.path.join(spirv_base_path(), "spirv-opt.sh internal/shaders/output/" + spv_name +
                                                                              " -o internal/shaders/output/" + spv_name),
                                              shell=True, capture_output=True, text=True, check=False)
    if ENABLE_DXC_COMPILATION and hlsl_profile is not None:
        if ENABLE_SPIRV_CROSS:
            hlsl_cross_flags = "--shader-model 62 --hlsl-enable-16bit-types" if hlsl_profile == "cs_6_2" else "--shader-model 60"
            hlsl_cross_result = subprocess.run(os.path.join(spirv_base_path(), "spirv-cross internal/shaders/output/" + spv_name +
                                                                                " --hlsl " + hlsl_cross_flags + " --output internal/shaders/output/" + hlsl_name),
                                                shell=True, capture_output=True, text=True, check=False)
        else:
            hlsl_cross_cmd = os.path.join(glslx_base_path(), "glslx -DVULKAN=1 -i internal/shaders/" + src_name + " " +
                                          defines +" -o internal/shaders/output/" + hlsl_name)
            hlsl_cross_result = subprocess.run(hlsl_cross_cmd, shell=True, capture_output=True, text=True, check=False)
        dxc_flags = "-enable-16bit-types" if hlsl_profile == "cs_6_2" else ""
        dxc_result = subprocess.run(os.path.join(dxc_base_path(), "dxc " + dxc_flags + " -T " + hlsl_profile + " internal/shaders/output/" + hlsl_name +
                                                                  " -Fo internal/shaders/output/" + cso_name),
                                    shell=True, capture_output=True, text=True, check=False)
        if dxc_result.returncode == 0:
            app_refl_result = subprocess.run(os.path.join(dxc_base_path(), "append_refl_data internal/shaders/output/", cso_name),
                                             shell=True, capture_output=True, text=True, check=False)
            if app_refl_result.returncode == 0:
                with open(os.path.join("internal", "shaders", "output", cso_name), 'rb') as f:
                    cso_data = f.read()
                out = bin2header(cso_data, os.path.join("internal", "shaders", "output", cso_name))
                with open(os.path.join("internal", "shaders", "output", cso_name + ".inl"), 'w') as f:
                    f.write(out)

    if compile_result.returncode == 0:
        with open(os.path.join("internal", "shaders", "output", spv_name), 'rb') as f:
            spv_data = f.read()
        out = bin2header(spv_data, os.path.join("internal", "shaders", "output", spv_name))
        with open(os.path.join("internal", "shaders", "output", spv_name + ".inl"), 'w') as f:
            f.write(out)

    mutex.acquire()
    try:
        print(compile_result.stdout)
        if rewrite_result is not None:
            print(rewrite_result.stdout)
            print(rewrite_result.stderr)
        #if spirv_opt_result is not None:
        #    print(spirv_opt_result.stdout)
        #    print(spirv_opt_result.stderr)
        if hlsl_cross_result is not None:
            print(hlsl_cross_result.stdout)
            print(hlsl_cross_result.stderr)
        if dxc_result is not None:
            print(dxc_result.stdout)
            print(dxc_result.stderr)
    finally:
        mutex.release()

def compile_shader_async(src_name, spv_name=None, glsl_version=None, target_env="spirv1.3", defines = "", hlsl_profile="cs_6_0"):
    if ENABLE_MULTIPLE_THREADS:
        threading.Thread(target=compile_shader, args=(src_name, spv_name, glsl_version, target_env, defines, hlsl_profile,)).start()
    else:
        compile_shader(src_name, spv_name, glsl_version, target_env, defines, hlsl_profile)

def main():
    for item in os.listdir("internal/shaders/output"):
        if item.endswith(".glsl") or item.endswith(".spv") or item.endswith(".spv.inl") or (ENABLE_DXC_COMPILATION and (item.endswith(".hlsl") or item.endswith(".cso") or item.endswith(".cso.inl"))):
            os.remove(os.path.join("internal/shaders/output", item))

    # Primary ray generation
    compile_shader_async(src_name="primary_ray_gen.comp.glsl", spv_name="primary_ray_gen_simple.comp.spv", defines="-DADAPTIVE=0")
    compile_shader_async(src_name="primary_ray_gen.comp.glsl", spv_name="primary_ray_gen_adaptive.comp.spv", defines="-DADAPTIVE=1")

    # Scene intersection (main, inline RT)
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_swrt_atlas.comp.spv",
                         defines="-DINDIRECT=0 -DHWRT=0 -DBINDLESS=0 -DSUBGROUP=0")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_swrt_bindless.comp.spv",
                         defines="-DINDIRECT=0 -DHWRT=0 -DBINDLESS=1 -DSUBGROUP=0")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_swrt_atlas_subgroup.comp.spv",
                         defines="-DINDIRECT=0 -DHWRT=0 -DBINDLESS=0 -DSUBGROUP=1")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_swrt_bindless_subgroup.comp.spv",
                         defines="-DINDIRECT=0 -DHWRT=0 -DBINDLESS=1 -DSUBGROUP=1")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_hwrt_atlas.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=0 -DHWRT=1 -DBINDLESS=0", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_hwrt_bindless.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=0 -DHWRT=1 -DBINDLESS=1", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_atlas.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=0 -DSUBGROUP=0")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_bindless.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=1 -DSUBGROUP=0")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_atlas_subgroup.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=0 -DSUBGROUP=1")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_bindless_subgroup.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=1 -DSUBGROUP=1")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_hwrt_atlas.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DHWRT=1 -DBINDLESS=0", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_hwrt_bindless.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DHWRT=1 -DBINDLESS=1", hlsl_profile="cs_6_5")
    # Scene intersection (main, pipeline RT)
    compile_shader_async(src_name="intersect_scene.rgen.glsl", spv_name="intersect_scene.rgen.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=0 -DBINDLESS=1", hlsl_profile=None)
    compile_shader_async(src_name="intersect_scene.rgen.glsl", spv_name="intersect_scene_indirect.rgen.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DBINDLESS=1", hlsl_profile=None)
    compile_shader_async(src_name="intersect_scene.rchit.glsl", spv_name="intersect_scene.rchit.spv",
                         glsl_version="460", target_env="spirv1.4", hlsl_profile=None)
    compile_shader_async(src_name="intersect_scene.rmiss.glsl", spv_name="intersect_scene.rmiss.spv",
                         glsl_version="460", target_env="spirv1.4", hlsl_profile=None)

    # Lights intersection
    compile_shader_async(src_name="intersect_area_lights.comp.glsl", defines="-DPRIMARY=0")

    # Shading
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_atlas.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_atlas_sky.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_atlas_cache_update.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=0 -DCACHE_UPDATE=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_atlas_cache_query.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=0 -DCACHE_QUERY=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_atlas_cache_query_sky.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=0 -DCACHE_QUERY=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_bindless.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_bindless_sky.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_bindless_cache_update.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=1 -DCACHE_UPDATE=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_bindless_cache_query.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=1 -DCACHE_QUERY=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_primary_bindless_cache_query_sky.comp.spv",
                         defines="-DPRIMARY=1 -DINDIRECT=1 -DBINDLESS=1 -DCACHE_QUERY=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_cache_update.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DCACHE_UPDATE=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_cache_query.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DCACHE_QUERY=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_cache_query_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DCACHE_QUERY=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_cache_update.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DCACHE_UPDATE=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_cache_query.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DCACHE_QUERY=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_cache_query_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DCACHE_QUERY=1 -DDETAILED_SKY=1")

    # Sky
    compile_shader_async(src_name="shade_sky.comp.glsl", spv_name="shade_sky.comp.spv", defines="-DBAKE=0")
    compile_shader_async(src_name="shade_sky.comp.glsl", spv_name="bake_sky.comp.spv", defines="-DBAKE=1")

    # Scene intersection (shadow)
    compile_shader_async(src_name="intersect_scene_shadow.comp.glsl", spv_name="intersect_scene_shadow_swrt_atlas.comp.spv",
                         defines="-DHWRT=0 -DBINDLESS=0 -DSUBGROUP=0")
    compile_shader_async(src_name="intersect_scene_shadow.comp.glsl", spv_name="intersect_scene_shadow_swrt_bindless.comp.spv",
                         defines="-DHWRT=0 -DBINDLESS=1 -DSUBGROUP=0")
    compile_shader_async(src_name="intersect_scene_shadow.comp.glsl", spv_name="intersect_scene_shadow_swrt_atlas_subgroup.comp.spv",
                         defines="-DHWRT=0 -DBINDLESS=0 -DSUBGROUP=1")
    compile_shader_async(src_name="intersect_scene_shadow.comp.glsl", spv_name="intersect_scene_shadow_swrt_bindless_subgroup.comp.spv",
                         defines="-DHWRT=0 -DBINDLESS=1 -DSUBGROUP=1")
    compile_shader_async(src_name="intersect_scene_shadow.comp.glsl", spv_name="intersect_scene_shadow_hwrt_atlas.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DHWRT=1 -DBINDLESS=0", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene_shadow.comp.glsl", spv_name="intersect_scene_shadow_hwrt_bindless.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DHWRT=1 -DBINDLESS=1", hlsl_profile="cs_6_5")

    # Spatial radiance caching
    compile_shader_async(src_name="spatial_cache_update.comp.glsl", spv_name="spatial_cache_update.comp.spv",
                         defines="-DNO_64BIT_ATOMICS=0", hlsl_profile="cs_6_6")
    compile_shader_async(src_name="spatial_cache_update.comp.glsl", spv_name="spatial_cache_update_compat.comp.spv",
                         defines="-DNO_64BIT_ATOMICS=1")
    compile_shader_async(src_name="spatial_cache_resolve.comp.glsl", spv_name="spatial_cache_resolve.comp.spv")

    # Postprocess
    compile_shader_async(src_name="mix_incremental.comp.glsl", spv_name="mix_incremental.comp.spv")
    compile_shader_async(src_name="postprocess.comp.glsl")

    # Denoise
    compile_shader_async(src_name="filter_variance.comp.glsl")
    compile_shader_async(src_name="nlm_filter.comp.glsl", spv_name="nlm_filter.comp.spv", defines="-DUSE_BASE_COLOR=1 -DUSE_DEPTH_NORMAL=1")

    # Sorting
    compile_shader_async(src_name="sort_hash_rays.comp.glsl")
    compile_shader_async(src_name="sort_init_count_table.comp.glsl")
    compile_shader_async(src_name="sort_reduce.comp.glsl")
    compile_shader_async(src_name="sort_scan.comp.glsl", spv_name="sort_scan.comp.spv", defines="-DADD_PARTIAL_SUMS=0")
    compile_shader_async(src_name="sort_scan.comp.glsl", spv_name="sort_scan_add.comp.spv", defines="-DADD_PARTIAL_SUMS=1")
    compile_shader_async(src_name="sort_scatter.comp.glsl")
    compile_shader_async(src_name="sort_reorder_rays.comp.glsl")

    # Other
    compile_shader_async(src_name="prepare_indir_args.comp.glsl")
    compile_shader_async(src_name="debug_rt.comp.glsl", target_env="spirv1.4", hlsl_profile="cs_6_5")

    # Convolution
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_fp32.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_fp16.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1 " \
                                 "-DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_coop.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1 " \
                                 "-DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1 -DUSE_FP16=1 " \
                                 "-DUSE_COOP_MATRIX=1",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112 -DUSE_FP16=1 " \
                                 "-DUSE_COOP_MATRIX=1",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96 -DUSE_FP16=1 " \
                                 "-DUSE_COOP_MATRIX=1",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64 -DUSE_FP16=1 " \
                                 "-DUSE_COOP_MATRIX=1",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_coop.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1 -DUSE_FP16=1 -DUSE_COOP_MATRIX=1",
                         hlsl_profile=None)


if __name__ == "__main__":
    main()
