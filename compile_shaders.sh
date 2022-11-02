
./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/primary_ray_gen.comp.glsl -o internal/shaders/primary_ray_gen.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/primary_ray_gen.comp.spv -o internal/shaders/primary_ray_gen.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/primary_ray_gen.comp.inl internal/shaders/primary_ray_gen.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/trace_rays.comp.glsl -DPRIMARY=1 -DHWRT=0 -o internal/shaders/trace_primary_rays_swrt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/trace_primary_rays_swrt.comp.spv -o internal/shaders/trace_primary_rays_swrt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/trace_primary_rays_swrt.comp.inl internal/shaders/trace_primary_rays_swrt.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/trace_rays.comp.glsl -DPRIMARY=1 -DHWRT=1 -o internal/shaders/trace_primary_rays_hwrt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/trace_primary_rays_hwrt.comp.spv -o internal/shaders/trace_primary_rays_hwrt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/trace_primary_rays_hwrt.comp.inl internal/shaders/trace_primary_rays_hwrt.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/trace_rays.comp.glsl -DPRIMARY=0 -DHWRT=0 -o internal/shaders/trace_secondary_rays_swrt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/trace_secondary_rays_swrt.comp.spv -o internal/shaders/trace_secondary_rays_swrt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/trace_secondary_rays_swrt.comp.inl internal/shaders/trace_secondary_rays_swrt.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/trace_rays.comp.glsl -DPRIMARY=0 -DHWRT=1 -o internal/shaders/trace_secondary_rays_hwrt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/trace_secondary_rays_hwrt.comp.spv -o internal/shaders/trace_secondary_rays_hwrt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/trace_secondary_rays_hwrt.comp.inl internal/shaders/trace_secondary_rays_hwrt.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_area_lights.comp.glsl -DPRIMARY=0 -o internal/shaders/intersect_area_lights.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/intersect_area_lights.comp.spv -o internal/shaders/intersect_area_lights.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/intersect_area_lights.comp.inl internal/shaders/intersect_area_lights.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/shade_hits.comp.glsl -DPRIMARY=1 -o internal/shaders/shade_primary_hits.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/shade_primary_hits.comp.spv -o internal/shaders/shade_primary_hits.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/shade_primary_hits.comp.inl internal/shaders/shade_primary_hits.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/trace_shadow.comp.glsl -DHWRT=0 -o internal/shaders/trace_shadow_swrt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/trace_shadow_swrt.comp.spv -o internal/shaders/trace_shadow_swrt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/trace_shadow_swrt.comp.inl internal/shaders/trace_shadow_swrt.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/trace_shadow.comp.glsl -DHWRT=1 -o internal/shaders/trace_shadow_hwrt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/trace_shadow_hwrt.comp.spv -o internal/shaders/trace_shadow_hwrt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/trace_shadow_hwrt.comp.inl internal/shaders/trace_shadow_hwrt.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/shade_hits.comp.glsl -DPRIMARY=0 -o internal/shaders/shade_secondary_hits.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/shade_secondary_hits.comp.spv -o internal/shaders/shade_secondary_hits.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/shade_secondary_hits.comp.inl internal/shaders/shade_secondary_hits.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/prepare_indir_args.comp.glsl -o internal/shaders/prepare_indir_args.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/prepare_indir_args.comp.spv -o internal/shaders/prepare_indir_args.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/prepare_indir_args.comp.inl internal/shaders/prepare_indir_args.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/mix_incremental.comp.glsl -o internal/shaders/mix_incremental.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/mix_incremental.comp.spv -o internal/shaders/mix_incremental.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/mix_incremental.comp.inl internal/shaders/mix_incremental.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.3 internal/shaders/postprocess.comp.glsl -o internal/shaders/postprocess.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/postprocess.comp.spv -o internal/shaders/postprocess.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/postprocess.comp.inl internal/shaders/postprocess.comp.spv

./third-party/spirv/linux/glslangValidator -V --target-env spirv1.4 internal/shaders/debug_rt.comp.glsl -o internal/shaders/debug_rt.comp.spv
./third-party/spirv/linux/spirv-opt.sh internal/shaders/debug_rt.comp.spv -o internal/shaders/debug_rt.comp.spv
./third-party/spirv/linux/bin2c -o internal/shaders/debug_rt.comp.inl internal/shaders/debug_rt.comp.spv
