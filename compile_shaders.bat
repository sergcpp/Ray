del "internal\shaders\*.spv" /S
del "internal\shaders\*.inl" /S
del "internal\shaders\*.hlsl" /S

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/primary_ray_gen.comp.glsl -o internal/shaders/primary_ray_gen.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/primary_ray_gen.comp.spv -o internal/shaders/primary_ray_gen.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/primary_ray_gen.comp.spv --hlsl --shader-model 60 --output internal/shaders/primary_ray_gen.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/primary_ray_gen.comp.inl internal/shaders/primary_ray_gen.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=0 -DBINDLESS=0 -o internal/shaders/intersect_scene_primary_swrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_primary_swrt_atlas.comp.spv -o internal/shaders/intersect_scene_primary_swrt_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_primary_swrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_primary_swrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_primary_swrt_atlas.comp.inl internal/shaders/intersect_scene_primary_swrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=0 -DBINDLESS=1 -o internal/shaders/intersect_scene_primary_swrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_primary_swrt_bindless.comp.spv -o internal/shaders/intersect_scene_primary_swrt_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_primary_swrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_primary_swrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_primary_swrt_bindless.comp.inl internal/shaders/intersect_scene_primary_swrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=1 -DBINDLESS=0 -o internal/shaders/intersect_scene_primary_hwrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_primary_hwrt_atlas.comp.spv -o internal/shaders/intersect_scene_primary_hwrt_atlas.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_primary_hwrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_primary_hwrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_primary_hwrt_atlas.comp.inl internal/shaders/intersect_scene_primary_hwrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=1 -DHWRT=1 -DBINDLESS=1 -o internal/shaders/intersect_scene_primary_hwrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_primary_hwrt_bindless.comp.spv -o internal/shaders/intersect_scene_primary_hwrt_bindless.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_primary_hwrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_primary_hwrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_primary_hwrt_bindless.comp.inl internal/shaders/intersect_scene_primary_hwrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=0 -DBINDLESS=0 -o internal/shaders/intersect_scene_secondary_swrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_secondary_swrt_atlas.comp.spv -o internal/shaders/intersect_scene_secondary_swrt_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_secondary_swrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_secondary_swrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_secondary_swrt_atlas.comp.inl internal/shaders/intersect_scene_secondary_swrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=0 -DBINDLESS=1 -o internal/shaders/intersect_scene_secondary_swrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_secondary_swrt_bindless.comp.spv -o internal/shaders/intersect_scene_secondary_swrt_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_secondary_swrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_secondary_swrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_secondary_swrt_bindless.comp.inl internal/shaders/intersect_scene_secondary_swrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=1 -DBINDLESS=0 -o internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.spv -o internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.inl internal/shaders/intersect_scene_secondary_hwrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene.comp.glsl -DPRIMARY=0 -DHWRT=1 -DBINDLESS=1 -o internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.spv -o internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.inl internal/shaders/intersect_scene_secondary_hwrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_area_lights.comp.glsl -DPRIMARY=0 -o internal/shaders/intersect_area_lights.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_area_lights.comp.spv -o internal/shaders/intersect_area_lights.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_area_lights.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_area_lights.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_area_lights.comp.inl internal/shaders/intersect_area_lights.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=1 -DBINDLESS=0 -o internal/shaders/shade_primary_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/shade_primary_atlas.comp.spv -o internal/shaders/shade_primary_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/shade_primary_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/shade_primary_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/shade_primary_atlas.comp.inl internal/shaders/shade_primary_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=1 -DBINDLESS=1 -o internal/shaders/shade_primary_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/shade_primary_bindless.comp.spv -o internal/shaders/shade_primary_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/shade_primary_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/shade_primary_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/shade_primary_bindless.comp.inl internal/shaders/shade_primary_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=0 -DBINDLESS=0 -o internal/shaders/intersect_scene_shadow_swrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_shadow_swrt_atlas.comp.spv -o internal/shaders/intersect_scene_shadow_swrt_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_shadow_swrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_shadow_swrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_shadow_swrt_atlas.comp.inl internal/shaders/intersect_scene_shadow_swrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=0 -DBINDLESS=1 -o internal/shaders/intersect_scene_shadow_swrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_shadow_swrt_bindless.comp.spv -o internal/shaders/intersect_scene_shadow_swrt_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_shadow_swrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_shadow_swrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_shadow_swrt_bindless.comp.inl internal/shaders/intersect_scene_shadow_swrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=1 -DBINDLESS=0 -o internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.spv -o internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.inl internal/shaders/intersect_scene_shadow_hwrt_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 --glsl-version 460 internal/shaders/intersect_scene_shadow.comp.glsl -DHWRT=1 -DBINDLESS=1 -o internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.spv -o internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.inl internal/shaders/intersect_scene_shadow_hwrt_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=0 -DBINDLESS=0 -o internal/shaders/shade_secondary_atlas.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/shade_secondary_atlas.comp.spv -o internal/shaders/shade_secondary_atlas.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/shade_secondary_atlas.comp.spv --hlsl --shader-model 60 --output internal/shaders/shade_secondary_atlas.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/shade_secondary_atlas.comp.inl internal/shaders/shade_secondary_atlas.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/shade.comp.glsl -DPRIMARY=0 -DBINDLESS=1 -o internal/shaders/shade_secondary_bindless.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/shade_secondary_bindless.comp.spv -o internal/shaders/shade_secondary_bindless.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/shade_secondary_bindless.comp.spv --hlsl --shader-model 60 --output internal/shaders/shade_secondary_bindless.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/shade_secondary_bindless.comp.inl internal/shaders/shade_secondary_bindless.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/prepare_indir_args.comp.glsl -o internal/shaders/prepare_indir_args.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/prepare_indir_args.comp.spv -o internal/shaders/prepare_indir_args.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/prepare_indir_args.comp.spv --hlsl --shader-model 60 --output internal/shaders/prepare_indir_args.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/prepare_indir_args.comp.inl internal/shaders/prepare_indir_args.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/mix_incremental.comp.glsl -o internal/shaders/mix_incremental.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/mix_incremental.comp.spv -o internal/shaders/mix_incremental.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/mix_incremental.comp.spv --hlsl --shader-model 60 --output internal/shaders/mix_incremental.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/mix_incremental.comp.inl internal/shaders/mix_incremental.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.3 internal/shaders/postprocess.comp.glsl -o internal/shaders/postprocess.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/postprocess.comp.spv -o internal/shaders/postprocess.comp.spv
third-party\spirv\win32\spirv-cross internal/shaders/postprocess.comp.spv --hlsl --shader-model 60 --output internal/shaders/postprocess.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/postprocess.comp.inl internal/shaders/postprocess.comp.spv

third-party\spirv\win32\glslangValidator -V --target-env spirv1.4 internal/shaders/debug_rt.comp.glsl -o internal/shaders/debug_rt.comp.spv
if %errorlevel% neq 0 exit /b %errorlevel%
call third-party\spirv\win32\spirv-opt.bat internal/shaders/debug_rt.comp.spv -o internal/shaders/debug_rt.comp.spv
REM third-party\spirv\win32\spirv-cross internal/shaders/debug_rt.comp.spv --hlsl --shader-model 60 --output internal/shaders/debug_rt.comp.hlsl
third-party\spirv\win32\bin2c -o internal/shaders/debug_rt.comp.inl internal/shaders/debug_rt.comp.spv
