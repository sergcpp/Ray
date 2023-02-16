del "internal\shaders\output\*.spv" /S
del "internal\shaders\output\*.inl" /S
del "internal\shaders\output\*.hlsl" /S

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/primary_ray_gen.comp.glsl -o internal/shaders/output/primary_ray_gen.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/primary_ray_gen.comp.spv -o internal/shaders/output/primary_ray_gen.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/primary_ray_gen.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/primary_ray_gen.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/primary_ray_gen.comp.inl internal/shaders/output/primary_ray_gen.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=0 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.inl internal/shaders/output/intersect_scene_primary_swrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=0 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.inl internal/shaders/output/intersect_scene_primary_swrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=1 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.inl internal/shaders/output/intersect_scene_primary_hwrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=1 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.inl internal/shaders/output/intersect_scene_primary_hwrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=0 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.inl internal/shaders/output/intersect_scene_secondary_swrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=0 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.inl internal/shaders/output/intersect_scene_secondary_swrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=1 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.inl internal/shaders/output/intersect_scene_secondary_hwrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=1 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.inl internal/shaders/output/intersect_scene_secondary_hwrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_area_lights.comp.glsl -DPRIMARY=0 -o internal/shaders/output/intersect_area_lights.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_area_lights.comp.spv -o internal/shaders/output/intersect_area_lights.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_area_lights.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_area_lights.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_area_lights.comp.inl internal/shaders/output/intersect_area_lights.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=1 -DBINDLESS=0 -o internal/shaders/output/shade_primary_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/shade_primary_atlas.comp.spv -o internal/shaders/output/shade_primary_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/shade_primary_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/shade_primary_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/shade_primary_atlas.comp.inl internal/shaders/output/shade_primary_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=1 -DBINDLESS=1 -o internal/shaders/output/shade_primary_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/shade_primary_bindless.comp.spv -o internal/shaders/output/shade_primary_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/shade_primary_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/shade_primary_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/shade_primary_bindless.comp.inl internal/shaders/output/shade_primary_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=0 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.inl internal/shaders/output/intersect_scene_shadow_swrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=0 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.inl internal/shaders/output/intersect_scene_shadow_swrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=1 -DBINDLESS=0 -o internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv -o internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.inl internal/shaders/output/intersect_scene_shadow_hwrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=1 -DBINDLESS=1 -o internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv -o internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.inl internal/shaders/output/intersect_scene_shadow_hwrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=0 -DBINDLESS=0 -o internal/shaders/output/shade_secondary_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/shade_secondary_atlas.comp.spv -o internal/shaders/output/shade_secondary_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/shade_secondary_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/shade_secondary_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/shade_secondary_atlas.comp.inl internal/shaders/output/shade_secondary_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=0 -DBINDLESS=1 -o internal/shaders/output/shade_secondary_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/shade_secondary_bindless.comp.spv -o internal/shaders/output/shade_secondary_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/shade_secondary_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/shade_secondary_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/shade_secondary_bindless.comp.inl internal/shaders/output/shade_secondary_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/prepare_indir_args.comp.glsl -o internal/shaders/output/prepare_indir_args.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/prepare_indir_args.comp.spv -o internal/shaders/output/prepare_indir_args.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/prepare_indir_args.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/prepare_indir_args.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/prepare_indir_args.comp.inl internal/shaders/output/prepare_indir_args.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/mix_incremental.comp.glsl -o internal/shaders/output/mix_incremental.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/mix_incremental.comp.spv -o internal/shaders/output/mix_incremental.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/mix_incremental.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/mix_incremental.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/mix_incremental.comp.inl internal/shaders/output/mix_incremental.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/postprocess.comp.glsl -o internal/shaders/output/postprocess.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/postprocess.comp.spv -o internal/shaders/output/postprocess.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/output/postprocess.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/postprocess.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/postprocess.comp.inl internal/shaders/output/postprocess.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 internal/shaders/debug_rt.comp.glsl -o internal/shaders/output/debug_rt.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/output/debug_rt.comp.spv -o internal/shaders/output/debug_rt.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/output/debug_rt.comp.spv --hlsl --shader-model 60 --output internal/shaders/output/debug_rt.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/output/debug_rt.comp.inl internal/shaders/output/debug_rt.comp.spv
