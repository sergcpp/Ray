import binascii
import os
import re
import struct
import sys
import subprocess
import threading
import zlib

ENABLE_MULTIPLE_THREADS = True
ENABLE_SPIRV_OPTIMIZATION = True
ENABLE_COMPRESSION = True
ENABLE_DXC_COMPILATION = (os.name == "nt")
ENABLE_METAL_COMPILATION = (sys.platform == "darwin")  # metallib blobs emit on macOS hosts (xcrun metal), like CSO on Windows
ENABLE_GLSL_REWRITE = False
ENABLE_SPIRV_CROSS = False

MSL_VERSION = "30000"
# AIR ray queries need macOS 13; the *load* floor of these blobs is set by the Xcode used to build them
MTL_TARGET = "air64-apple-macos13.0"
MTL_STD = "-std=" + "metal" + MSL_VERSION[0] + "." + MSL_VERSION[1]
# Direct bindings for set 0, device argument buffer (Tier 2) for set 1 bindless textures.
MSL_CROSS_FLAGS = ("--msl-argument-buffers --msl-argument-buffer-tier 1"
                   " --msl-discrete-descriptor-set 0 --msl-device-argument-buffer 1")

mutex = threading.Lock()

g_threads = []
g_failures = []

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

# --- MSL (Metal) cross-compilation from the optimized SPIR-V ------------------
# spirv-cross auto-assigns MSL attribute indices; dump-resources gives us the
# SPIR-V (set, binding) per variable ID. Joined by variable ID into a static
# binding table so the Metal renderer needs no runtime reflection.

msl_sec_kinds = {
    "textures": "tex", "separate images": "tex", "separate samplers": "sampler",
    "images": "image", "ssbos": "buf", "push": "push", "acceleration structures": "acc",
}

# Must match MslBindingKind in the Metal renderer (RendererMtl.cpp / SceneMtl.h).
msl_k_sbuf_ro, msl_k_sbuf_rw, msl_k_tex, msl_k_sampler, msl_k_image_ro, \
msl_k_image_wo, msl_k_image_rw, msl_k_push, msl_k_acc, msl_k_descr_arr = range(10)

def parse_msl_dump(dump_text):
    # var id -> {set, binding, kind_hint, readonly, writeonly, unsized, size}
    resources = {}
    section = None
    for line in dump_text.splitlines():
        stripped = line.strip()
        if line in msl_sec_kinds:
            section = msl_sec_kinds[line]
            continue
        if stripped.startswith("=====") or not stripped:
            continue
        m = re.match(r"^\s*ID (\d+) : (\w+)(\[(\d+)?\])?(?: .*)?$", line)
        if not m or section is None:
            continue
        var_id = int(m.group(1))
        entry = {"kind": section, "unsized": False, "size": 0, "set": 0, "binding": 0,
                 "readonly": "readonly" in line, "writeonly": "writeonly" in line}
        sm = re.search(r"Set : (\d+)\) \(Binding : (\d+)", line)
        if sm:
            entry["set"], entry["binding"] = int(sm.group(1)), int(sm.group(2))
        if m.group(3) is not None:
            if m.group(4) is None:
                entry["unsized"] = True
            else:
                entry["size"] = int(m.group(4))
        resources[var_id] = entry
    return resources

def split_msl_args(sig):
    args, depth, cur = [], 0, ""
    for ch in sig:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur)
    return args

def parse_msl_signature(msl_text):
    m = re.search(r"kernel void main0\((.*?)\)\n\{", msl_text, re.S)
    if not m:
        raise RuntimeError("kernel signature not found")
    args = []
    for a in split_msl_args(m.group(1)):
        am = re.search(r"\[\[(buffer|texture|sampler)\((\d+)\)\]\]\s*$", a.strip())
        if not am:
            continue  # builtin
        var = re.search(r"(\w+)\s+\[\[", a)
        args.append({"text": a, "attr": am.group(1), "index": int(am.group(2)),
                     "var": var.group(1), "argbuf": "spvDescriptorSetBuffer" in a})
    return args

def parse_msl_argbuf_structs(msl_text):
    # struct name -> [(id_slot, var id), ...]
    struct_body = {}
    for m in re.finditer(r"struct (spvDescriptorSetBuffer\d+)\s*\{(.*?)\n\}", msl_text, re.S):
        fields = []
        for fm in re.finditer(r"(?:spvDescriptor<.+?>|sampler)\s+m_(\d+)\s*\[\[id\((\d+)\)\]\]", m.group(2)):
            fields.append((int(fm.group(2)), int(fm.group(1))))
        struct_body[m.group(1)] = fields
    return struct_body

def build_msl_binding_table(msl_text, dump_text):
    resources = parse_msl_dump(dump_text)
    args = parse_msl_signature(msl_text)
    structs = parse_msl_argbuf_structs(msl_text)
    push_id = None
    for line in dump_text.splitlines():
        pm = re.match(r"^ ID (\d+) : _\d+$", line)
        if pm:
            push_id = int(pm.group(1))
    table = []
    for a in args:
        if a["argbuf"]:  # set-N argument buffer (bindless descriptor arrays)
            tm = re.search(r"spvDescriptorSetBuffer\d+", a["text"])
            if not tm:
                continue
            for (id_slot, var_id) in structs.get(tm.group(0), []):
                res = resources.get(var_id)
                if res is None:
                    continue
                table.append((res["set"], res["binding"], a["index"], id_slot, res["size"], msl_k_descr_arr))
            continue
        if a["attr"] == "buffer":
            m = re.search(r"_(\d+)$", a["var"])
            if push_id is not None and m and int(m.group(1)) == push_id:
                table.append((0, 0, a["index"], 0, 1, msl_k_push))  # push constants: no SPIR-V set/binding
                continue
        m = re.search(r"_(\d+)(Smplr)?$", a["var"])
        if not m:
            continue
        res = resources.get(int(m.group(1)))
        if res is None:
            continue
        if m.group(2):  # combined-sampler helper arg, same (set, binding) as its texture
            kind = msl_k_sampler
        elif res["kind"] == "buf":
            kind = msl_k_sbuf_ro if res["readonly"] else msl_k_sbuf_rw
        elif res["kind"] == "tex":
            kind = msl_k_tex
        elif res["kind"] == "sampler":
            kind = msl_k_sampler
        elif res["kind"] == "image":
            kind = msl_k_image_ro if res["readonly"] else (msl_k_image_wo if res["writeonly"] else msl_k_image_rw)
        elif res["kind"] == "acc":
            kind = msl_k_acc
        else:
            continue
        table.append((res["set"], res["binding"], a["index"], 0, res["size"], kind))
    return sorted(set(table))

def parse_msl_workgroup(dump_text):
    m = re.search(r"LocalSize: \((\d+), (\d+), (\d+)\)", dump_text)
    if not m:
        raise RuntimeError("LocalSize execution mode not found")
    return tuple(int(x) for x in m.groups())

# Blob payload (mirrors DX append_refl_data): [metallib | refl chunk | 8-byte trailer].
# Refl chunk: 'MLR1', u32 version, u32 workgroup[3], u32 count, count*{6 x u32 MslBindingDesc}.
# Trailer: refl chunk size (u32), 'MLR1'. Reader slices from EOF; metallib goes to newLibraryWithData.
def pack_msl_refl(workgroup, table):
    refl = struct.pack("<4sI3II", b"MLR1", 1, workgroup[0], workgroup[1], workgroup[2], len(table))
    for e in table:
        refl += struct.pack("<6I", *e)
    return refl

def compile_msl(spv_name):
    msl_name = spv_name[0:-3] + "msl"
    metallib_name = spv_name[0:-3] + "metallib"
    base = "internal/shaders/output/"
    cross_cmd = os.path.join(spirv_base_path(), "spirv-cross " + base + spv_name +
                                            " --msl --msl-version " + MSL_VERSION + " " + MSL_CROSS_FLAGS +
                                            " --output " + base + msl_name)
    cross_result = subprocess.run(cross_cmd, shell=True, capture_output=True, text=True, check=False)
    if cross_result.returncode != 0:
        return cross_result, None
    dump_result = subprocess.run(os.path.join(spirv_base_path(), "spirv-cross " + base + spv_name +
                                                            " --msl --msl-version " + MSL_VERSION + " " + MSL_CROSS_FLAGS +
                                                            " --dump-resources"),
                                 shell=True, capture_output=True, text=True, check=False)
    with open(base + msl_name, 'r') as f:
        msl_text = f.read()
    table = build_msl_binding_table(msl_text, dump_result.stderr)
    workgroup = parse_msl_workgroup(dump_result.stderr)
    # default FMA contraction kept: matches MoltenVK's VK ff values bit-identically in 35/37 M4 goldens;
    # -ffp-contract=off drifted them ~5 ff from VK (max 69)
    metal_result = subprocess.run("xcrun -sdk macosx metal -x metal " + MTL_STD + " -target " + MTL_TARGET +
                                                  " -O2 " + base + msl_name + " -o " + base + metallib_name,
                                  shell=True, capture_output=True, text=True, check=False)
    if metal_result.returncode != 0:
        return metal_result, None
    with open(base + metallib_name, 'rb') as f:
        metallib_data = f.read()
    refl = pack_msl_refl(workgroup, table)
    payload = metallib_data + refl + struct.pack("<I4s", len(refl), b"MLR1")
    return metal_result, (metallib_name, payload, table, workgroup)

def compile_shader(src_name, spv_name=None, glsl_version=None, target_env="spirv1.3", defines="", hlsl_profile="cs_6_0", metal=True):
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

    msl_cross_result = None
    if ENABLE_METAL_COMPILATION and metal and hlsl_profile is not None:
        msl_cross_result, msl_out = compile_msl(spv_name)
        if msl_cross_result.returncode == 0 and msl_out is not None:
            metallib_name, payload, table, workgroup = msl_out
            mtl_path = os.path.join("internal", "shaders", "output", metallib_name)
            # refl (bindings + workgroup) travels inside the payload - no sidecar symbols, single blob per shader
            out = bin2header(payload, mtl_path)
            with open(mtl_path + ".inl", 'w') as f:
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
        if compile_result.returncode != 0:
            print("SPIR-V compilation FAILED: " + spv_name)
            print(compile_result.stderr)
            g_failures.append(spv_name)
        if dxc_result is not None and dxc_result.returncode != 0:
            g_failures.append(spv_name)
        if msl_cross_result is not None and msl_cross_result.returncode != 0:
            print("MSL cross-compilation FAILED: " + spv_name)
            print(msl_cross_result.stdout)
            print(msl_cross_result.stderr)
            g_failures.append(spv_name)
    finally:
        mutex.release()

def compile_shader_async(src_name, spv_name=None, glsl_version=None, target_env="spirv1.3", defines = "", hlsl_profile="cs_6_0", metal=True):
    if ENABLE_MULTIPLE_THREADS:
        g_threads.append(threading.Thread(target=compile_shader, args=(src_name, spv_name, glsl_version, target_env, defines, hlsl_profile, metal,)))
        g_threads[-1].start()
    else:
        compile_shader(src_name, spv_name, glsl_version, target_env, defines, hlsl_profile, metal)

def main():
    for item in os.listdir("internal/shaders/output"):
        if item.endswith(".glsl") or item.endswith(".spv") or item.endswith(".spv.inl") or \
           (ENABLE_DXC_COMPILATION and (item.endswith(".hlsl") or item.endswith(".cso") or item.endswith(".cso.inl"))) or \
           (ENABLE_METAL_COMPILATION and (item.endswith(".msl") or item.endswith(".msl.inl") or
                                          item.endswith(".metallib") or item.endswith(".metallib.inl"))):
            os.remove(os.path.join("internal/shaders/output", item))

    # Primary ray generation
    compile_shader_async(src_name="primary_ray_gen.comp.glsl", spv_name="primary_ray_gen_simple.comp.spv", defines="-DADAPTIVE=0")
    compile_shader_async(src_name="primary_ray_gen.comp.glsl", spv_name="primary_ray_gen_adaptive.comp.spv", defines="-DADAPTIVE=1")

    # Scene intersection (main, inline RT). Subgroups are required by the renderer, so only
    # _subgroup SWRT variants are built; PRIMARY=1 variants shade/trace primary rays without the sort perm.
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_atlas_subgroup.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=0 -DSUBGROUP=1 -DPRIMARY=0")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_bindless_subgroup.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=1 -DSUBGROUP=1 -DPRIMARY=0")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_atlas_subgroup_primary.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=0 -DSUBGROUP=1 -DPRIMARY=1")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_swrt_bindless_subgroup_primary.comp.spv",
                         defines="-DINDIRECT=1 -DHWRT=0 -DBINDLESS=1 -DSUBGROUP=1 -DPRIMARY=1")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_hwrt_atlas.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DHWRT=1 -DBINDLESS=0 -DPRIMARY=0", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_hwrt_bindless.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DHWRT=1 -DBINDLESS=1 -DPRIMARY=0", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_hwrt_atlas_primary.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DHWRT=1 -DBINDLESS=0 -DPRIMARY=1", hlsl_profile="cs_6_5")
    compile_shader_async(src_name="intersect_scene.comp.glsl", spv_name="intersect_scene_indirect_hwrt_bindless_primary.comp.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DHWRT=1 -DBINDLESS=1 -DPRIMARY=1", hlsl_profile="cs_6_5")
    # Scene intersection (main, pipeline RT)
    compile_shader_async(src_name="intersect_scene.rgen.glsl", spv_name="intersect_scene_indirect.rgen.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DBINDLESS=1 -DPRIMARY=0", hlsl_profile=None)
    compile_shader_async(src_name="intersect_scene.rgen.glsl", spv_name="intersect_scene_indirect_primary.rgen.spv",
                         glsl_version="460", target_env="spirv1.4", defines="-DINDIRECT=1 -DBINDLESS=1 -DPRIMARY=1", hlsl_profile=None)
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
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_cache_update.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DCACHE_UPDATE=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_cache_query.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DCACHE_QUERY=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_atlas_cache_query_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=0 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DCACHE_QUERY=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DDETAILED_SKY=1")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_cache_update.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DCACHE_UPDATE=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_cache_query.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DCACHE_QUERY=1 -DDETAILED_SKY=0")
    compile_shader_async(src_name="shade.comp.glsl", spv_name="shade_secondary_bindless_cache_query_sky.comp.spv",
                         defines="-DPRIMARY=0 -DINDIRECT=1 -DBINDLESS=1 -DOUTPUT_BASE_COLOR=1 -DOUTPUT_DEPTH_NORMALS=1 -DCACHE_QUERY=1 -DDETAILED_SKY=1")

    # Sky
    compile_shader_async(src_name="shade_sky.comp.glsl", spv_name="shade_sky.comp.spv", defines="-DBAKE=0 -DPRIMARY=0")
    compile_shader_async(src_name="shade_sky.comp.glsl", spv_name="shade_sky_primary.comp.spv", defines="-DBAKE=0 -DPRIMARY=1")
    compile_shader_async(src_name="shade_sky.comp.glsl", spv_name="bake_sky.comp.spv", defines="-DBAKE=1")

    # Scene intersection (shadow)
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
                         defines="-DNO_64BIT_ATOMICS=0", hlsl_profile="cs_6_6", metal=False)  # MSL backend rejects 64-bit atomics - Metal uses the _compat build
    compile_shader_async(src_name="spatial_cache_update.comp.glsl", spv_name="spatial_cache_update_compat.comp.spv",
                         defines="-DNO_64BIT_ATOMICS=1")
    compile_shader_async(src_name="spatial_cache_update.comp.glsl", spv_name="spatial_cache_update_perm.comp.spv",
                         defines="-DNO_64BIT_ATOMICS=0 -DPERM=1", hlsl_profile="cs_6_6", metal=False)
    compile_shader_async(src_name="spatial_cache_update.comp.glsl", spv_name="spatial_cache_update_perm_compat.comp.spv",
                         defines="-DNO_64BIT_ATOMICS=1 -DPERM=1")
    compile_shader_async(src_name="spatial_cache_resolve.comp.glsl", spv_name="spatial_cache_resolve.comp.spv")

    # Postprocess
    compile_shader_async(src_name="mix_incremental.comp.glsl", spv_name="mix_incremental.comp.spv")
    compile_shader_async(src_name="postprocess.comp.glsl")

    # Denoise
    compile_shader_async(src_name="filter_variance.comp.glsl")
    compile_shader_async(src_name="nlm_filter.comp.glsl", spv_name="nlm_filter.comp.spv", defines="-DUSE_BASE_COLOR=1 -DUSE_DEPTH_NORMAL=1")

    # Sorting
    compile_shader_async(src_name="sort_init_count_table.comp.glsl")
    compile_shader_async(src_name="sort_reduce.comp.glsl")
    compile_shader_async(src_name="sort_scan.comp.glsl", spv_name="sort_scan_add.comp.spv")
    compile_shader_async(src_name="sort_scatter.comp.glsl")

    # Other
    compile_shader_async(src_name="prepare_indir_args.comp.glsl")
    compile_shader_async(src_name="debug_rt.comp.glsl", target_env="spirv1.4", hlsl_profile="cs_6_5")

    # Convolution
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_fp32.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_fp16.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1 " \
                                 "-DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_coop_8x8x16_CF32.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1 " \
                                 "-DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_Img_9_32_coop_16x16x16_CF16.comp.spv",
                         defines="-DIMG_INPUT1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIN_CHANNELS1=9 -DOUT_CHANNELS=32 -DHDR_TRANSFER1=1 -DPOS_NORMALIZE3=1 " \
                                 "-DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_32_Downsample_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=32 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_48_Downsample_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=48 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_48_64_Downsample_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=48 -DOUT_CHANNELS=64 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_80_Downsample_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=80 -DDOWNSAMPLE=1 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_64_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=64 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_64_32_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DOUT_CHANNELS=32 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_80_96_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=80 -DOUT_CHANNELS=96 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_96_96_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DOUT_CHANNELS=96 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_112_112_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DOUT_CHANNELS=112 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16", hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1 -DUSE_FP16=1 " \
                                 "-DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_32_3_img_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=32 -DOUT_CHANNELS=3 -DOUT_IMG=1 -DHDR_TRANSFER1=1 -DTONEMAP=1 -DUSE_FP16=1 " \
                                 "-DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112 -DUSE_FP16=1 " \
                                 "-DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_64_112_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=64 -DOUT_CHANNELS=112 -DUSE_FP16=1 " \
                                 "-DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96 -DUSE_FP16=1 " \
                                 "-DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_112_48_96_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=112 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=48 -DOUT_CHANNELS=96 -DUSE_FP16=1 " \
                                 "-DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64 -DUSE_FP16=1",
                         hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64 -DUSE_FP16=1 " \
                                 "-DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_96_32_64_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=96 -DUPSCALE1=1 -DBUF_INPUT2=1 -DIN_CHANNELS2=32 -DOUT_CHANNELS=64 -DUSE_FP16=1 " \
                                 "-DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_fp32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_fp16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1 -DUSE_FP16=1", hlsl_profile="cs_6_2")
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_coop_8x8x16_CF32.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1 -DUSE_FP16=1 -DCOOP_M=8 -DCOOP_N=8 -DCOOP_K=16 -DC_TYPE=float",
                         hlsl_profile=None)
    compile_shader_async(src_name="convolution.comp.glsl", spv_name="convolution_concat_64_9_64_coop_16x16x16_CF16.comp.spv",
                         defines="-DBUF_INPUT1=1 -DIN_CHANNELS1=64 -DUPSCALE1=1 -DIMG_INPUT2=1 -DIMG_INPUT3=1 -DIMG_INPUT4=1 -DIN_CHANNELS2=9 " \
                                 "-DOUT_CHANNELS=64 -DHDR_TRANSFER2=1 -DPOS_NORMALIZE4=1 -DUSE_FP16=1 -DCOOP_M=16 -DCOOP_N=16 -DCOOP_K=16",
                         hlsl_profile=None)

    for t in g_threads:
        t.join()
    if g_failures:
        print("FAILED shader compilations (%d): %s" % (len(g_failures), ", ".join(g_failures)))
        sys.exit(1)


if __name__ == "__main__":
    main()
