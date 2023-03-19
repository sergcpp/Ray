import binascii
import os
import sys
import subprocess

ENABLE_OPTIMIZATION = True
ENABLE_HLSL_CROSS_COMPILATION = (os.name == "nt")

def base_path():
    if sys.platform.startswith("linux"):
        return os.path.join("third-party", "spirv", "linux")
    elif sys.platform == "darwin":
        return os.path.join("third-party", "spirv", "macos")
    elif os.name == "nt":
        return os.path.join("third-party", "spirv", "win32")

def make_sublist_group(lst: list, grp: int) -> list:
    return [lst[i:i+grp] for i in range(0, len(lst), grp)]

def do_conversion(content: bytes) -> str:
    hexstr = binascii.hexlify(content).decode("UTF-8")
    hexstr = hexstr.upper()
    array = ["0x" + hexstr[i:i + 2] + "" for i in range(0, len(hexstr), 2)]
    array = make_sublist_group(array, 16)
    ret = "\n    ".join([", ".join(e) + "," for e in array])
    return ret[0:len(ret) - 1]

def bin2header(data, file_name):
    ret = "/* Contents of file " + file_name + " */\n"
    ret += "const long int " + file_name.replace('/', '_').replace('.', '_') + "_size = " + str(len(data)) + ";\n"
    ret += "const unsigned char " + file_name.replace('/', '_').replace('.', '_') + "[" + str(len(data)) + "] = {\n    "
    ret += do_conversion(data)
    ret += "\n};\n"
    return ret

def compile_shader(src_name, spv_name=None, glsl_version=None, target_env="spirv1.3", defines = ""):
    if  spv_name == None:
        spv_name = src_name
    src_name = src_name + ".comp.glsl"
    hlsl_name = spv_name + ".comp.hlsl"
    header_name = spv_name + ".comp.inl"
    spv_name = spv_name + ".comp.spv"

    compile_cmd = os.path.join(base_path(), "glslangValidator -V --target-env " + target_env + " internal/shaders/" + src_name + " " + defines + " -o internal/shaders/output/" + spv_name)
    if (glsl_version != None):
        compile_cmd += " --glsl-version " + glsl_version
    os.system(compile_cmd)
    if ENABLE_OPTIMIZATION == True:
        if os.name == "nt":
            os.system(os.path.join(base_path(), "spirv-opt.bat internal/shaders/output/" + spv_name + " -o internal/shaders/output/" + spv_name))
        else:
            os.system(os.path.join(base_path(), "spirv-opt.sh internal/shaders/output/" + spv_name + " -o internal/shaders/output/" + spv_name))
    if ENABLE_HLSL_CROSS_COMPILATION == True:
        os.system(os.path.join(base_path(), "spirv-cross internal/shaders/output/" + spv_name + " --hlsl --shader-model 60 --output internal/shaders/output/" + hlsl_name))

    with open("internal/shaders/output/" + spv_name, 'rb') as f:
        data = f.read()
    out = bin2header(data, "internal/shaders/output/" + spv_name)
    with open("internal/shaders/output/" + header_name, 'w') as f:
        f.write(out)

def main():
    for item in os.listdir("internal/shaders/output"):
        if item.endswith(".spv") or item.endswith(".inl") or (ENABLE_HLSL_CROSS_COMPILATION and item.endswith(".hlsl")):
            os.remove(os.path.join("internal/shaders/output", item))

    # Primary ray generation
    compile_shader(src_name="primary_ray_gen")

    # Scene intersection (main)
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_primary_swrt_atlas", defines="-DPRIMARY=1 -DHWRT=0 -DBINDLESS=0")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_primary_swrt_bindless", defines="-DPRIMARY=1 -DHWRT=0 -DBINDLESS=1")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_primary_hwrt_atlas", glsl_version="460", target_env="spirv1.4", defines="-DPRIMARY=1 -DHWRT=1 -DBINDLESS=0")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_primary_hwrt_bindless", glsl_version="460", target_env="spirv1.4", defines="-DPRIMARY=1 -DHWRT=1 -DBINDLESS=1")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_secondary_swrt_atlas", defines="-DPRIMARY=0 -DHWRT=0 -DBINDLESS=0")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_secondary_swrt_bindless", defines="-DPRIMARY=0 -DHWRT=0 -DBINDLESS=1")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_secondary_hwrt_atlas", glsl_version="460", target_env="spirv1.4", defines="-DPRIMARY=0 -DHWRT=1 -DBINDLESS=0")
    compile_shader(src_name="intersect_scene", spv_name="intersect_scene_secondary_hwrt_bindless", glsl_version="460", target_env="spirv1.4", defines="-DPRIMARY=0 -DHWRT=1 -DBINDLESS=1")

    # Lights intersection
    compile_shader(src_name="intersect_area_lights", defines="-DPRIMARY=0")

    # Shading
    compile_shader(src_name="shade", spv_name="shade_primary_atlas", defines="-DPRIMARY=1 -DBINDLESS=0")
    compile_shader(src_name="shade", spv_name="shade_primary_bindless", defines="-DPRIMARY=1 -DBINDLESS=1")
    compile_shader(src_name="shade", spv_name="shade_secondary_atlas", defines="-DPRIMARY=0 -DBINDLESS=0")
    compile_shader(src_name="shade", spv_name="shade_secondary_bindless", defines="-DPRIMARY=0 -DBINDLESS=1")

    # Scene intersection (shadow)
    compile_shader(src_name="intersect_scene_shadow", spv_name="intersect_scene_shadow_swrt_atlas", defines="-DHWRT=0 -DBINDLESS=0")
    compile_shader(src_name="intersect_scene_shadow", spv_name="intersect_scene_shadow_swrt_bindless", defines="-DHWRT=0 -DBINDLESS=1")
    compile_shader(src_name="intersect_scene_shadow", spv_name="intersect_scene_shadow_hwrt_atlas", glsl_version="460", target_env="spirv1.4", defines="-DHWRT=1 -DBINDLESS=0")
    compile_shader(src_name="intersect_scene_shadow", spv_name="intersect_scene_shadow_hwrt_bindless", glsl_version="460", target_env="spirv1.4", defines="-DHWRT=1 -DBINDLESS=1")

    # Postprocess
    compile_shader(src_name="mix_incremental")
    compile_shader(src_name="postprocess")

    # Other
    compile_shader(src_name="prepare_indir_args")
    compile_shader(src_name="debug_rt", target_env="spirv1.4")

if __name__ == "__main__":
    main()
