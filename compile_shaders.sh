find internal/output/shaders -name "*.spv" -type f -delete
find internal/output/shaders -name "*.inl" -type f -delete
find internal/output/shaders -name "*.hlsl" -type f -delete

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    BASE_PATH="./third-party/spirv/linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    BASE_PATH="./third-party/spirv/macos"
fi

echo "$BASE_PATH"

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/primary_ray_gen.comp.glsl -o internal/shaders/output/primary_ray_gen.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/primary_ray_gen.comp.spv -o internal/shaders/output/primary_ray_gen.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/primary_ray_gen.comp.inl internal/shaders/output/primary_ray_gen.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=0 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.inl internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=0 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.inl internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=1 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.inl internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=1 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.inl internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=0 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.inl internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=0 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.inl internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=1 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.inl internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=1 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.inl internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_area_lights.comp.glsl -DPRIMARY=0 -o internal/shaders/output/intersect_area_lights.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_area_lights.comp.spv -o internal/shaders/output/intersect_area_lights.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_area_lights.comp.inl internal/shaders/output/intersect_area_lights.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=1 -DBINDLESS=0 -o internal/shaders/output/shade_primary_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/shade_primary_atlas.comp.spv -o internal/shaders/output/shade_primary_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/shade_primary_atlas.comp.inl internal/shaders/output/shade_primary_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=1 -DBINDLESS=1 -o internal/shaders/output/shade_primary_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/shade_primary_bindless.comp.spv -o internal/shaders/output/shade_primary_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/shade_primary_bindless.comp.inl internal/shaders/output/shade_primary_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=0 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.inl internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=0 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.inl internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=1 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.inl internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=1 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.inl internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=0 -DBINDLESS=0 -o internal/shaders/output/shade_secondary_atlas.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/shade_secondary_atlas.comp.spv -o internal/shaders/output/shade_secondary_atlas.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/shade_secondary_atlas.comp.inl internal/shaders/output/shade_secondary_atlas.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=0 -DBINDLESS=1 -o internal/shaders/output/shade_secondary_bindless.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/shade_secondary_bindless.comp.spv -o internal/shaders/output/shade_secondary_bindless.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/shade_secondary_bindless.comp.inl internal/shaders/output/shade_secondary_bindless.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/prepare_indir_args.comp.glsl -o internal/shaders/output/prepare_indir_args.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/prepare_indir_args.comp.spv -o internal/shaders/output/prepare_indir_args.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/prepare_indir_args.comp.inl internal/shaders/output/prepare_indir_args.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/mix_incremental.comp.glsl -o internal/shaders/output/mix_incremental.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/mix_incremental.comp.spv -o internal/shaders/output/mix_incremental.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/mix_incremental.comp.inl internal/shaders/output/mix_incremental.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.3 internal/shaders/postprocess.comp.glsl -o internal/shaders/output/postprocess.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/postprocess.comp.spv -o internal/shaders/output/postprocess.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/postprocess.comp.inl internal/shaders/output/postprocess.comp.spv

"$BASE_PATH"/glslangValidator -V --target-env spirv1.4 internal/shaders/debug_rt.comp.glsl -o internal/shaders/output/debug_rt.comp.spv
"$BASE_PATH"/spirv-opt.sh internal/shaders/output/debug_rt.comp.spv -o internal/shaders/output/debug_rt.comp.spv
"$BASE_PATH"/bin2c -o internal/shaders/output/debug_rt.comp.inl internal/shaders/output/debug_rt.comp.spv
