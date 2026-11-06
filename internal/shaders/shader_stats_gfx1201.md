# Ray shader stats (gfx1201)

Generated 2026-11-07 19:11 by `Ray/scripts/rga_shader_stats.py`, RGA 2.15.0.30, source: `internal/shaders/output/*.spv` (offline SPIR-V mode, release builds only).

| # | Shader | SPV, B | ISA, B | VGPRs | SGPRs | VGPR spills | SGPR spills | Scratch, B/thread | LDS, B | Max waves/SIMD | Occ. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `shade_secondary_atlas_cache_query_sky.comp.spv` | 229,360 | 104,472 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 2 | `shade_secondary_atlas_cache_query.comp.spv` | 228,948 | 104,112 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 3 | `shade_secondary_bindless_cache_query_sky.comp.spv` | 225,876 | 101,868 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 4 | `shade_secondary_bindless_cache_query.comp.spv` | 225,464 | 101,700 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 5 | `shade_secondary_atlas_sky.comp.spv` | 222,636 | 101,516 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 6 | `shade_secondary_atlas.comp.spv` | 222,224 | 101,312 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 7 | `shade_secondary_atlas_cache_update.comp.spv` | 220,584 | 101,044 | 171 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 8 | `shade_secondary_bindless_sky.comp.spv` | 219,152 | 99,248 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 9 | `shade_secondary_bindless.comp.spv` | 218,740 | 99,044 | 175 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 10 | `shade_secondary_bindless_cache_update.comp.spv` | 217,100 | 98,728 | 171 | 104 | 0 | 0 | 432 | 6,144 | 2 | 12% |
| 11 | `shade_primary_atlas_cache_query_sky.comp.spv` | 217,860 | 92,360 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 12 | `shade_primary_atlas_cache_query.comp.spv` | 217,448 | 92,336 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 13 | `shade_primary_bindless_cache_query_sky.comp.spv` | 214,180 | 90,440 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 14 | `shade_primary_bindless_cache_query.comp.spv` | 213,768 | 89,988 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 15 | `shade_primary_atlas_sky.comp.spv` | 211,136 | 89,752 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 16 | `shade_primary_atlas.comp.spv` | 210,724 | 89,544 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 17 | `shade_primary_atlas_cache_update.comp.spv` | 209,180 | 89,128 | 142 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 18 | `shade_primary_bindless_sky.comp.spv` | 207,456 | 87,692 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 19 | `shade_primary_bindless.comp.spv` | 207,044 | 87,484 | 145 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 20 | `shade_primary_bindless_cache_update.comp.spv` | 205,500 | 86,900 | 141 | 106 | 0 | 0 | 0 | 0 | 3 | 18% |
| 21 | `intersect_scene_shadow_swrt_atlas_subgroup.comp.spv` | 56,232 | 20,076 | 88 | 102 | 0 | 0 | 464 | 4,096 | 5 | 31% |
| 22 | `intersect_scene_shadow_swrt_bindless_subgroup.comp.spv` | 54,704 | 20,028 | 89 | 106 | 0 | 0 | 464 | 4,096 | 5 | 31% |
| 23 | `intersect_scene_shadow_swrt_bindless.comp.spv` | 54,608 | 19,176 | 89 | 106 | 0 | 0 | 464 | 4,096 | 5 | 31% |
| 24 | `intersect_scene_shadow_swrt_atlas.comp.spv` | 56,136 | 18,940 | 89 | 106 | 0 | 0 | 464 | 4,096 | 5 | 31% |
| 25 | `shade_sky.comp.spv` | 43,760 | 17,600 | 89 | 106 | 0 | 0 | 0 | 0 | 5 | 31% |
| 26 | `bake_sky.comp.spv` | 38,572 | 16,924 | 83 | 106 | 0 | 0 | 0 | 0 | 5 | 31% |
| 27 | `intersect_scene_swrt_atlas_subgroup.comp.spv` | 42,428 | 14,984 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 28 | `intersect_scene_indirect_swrt_atlas_subgroup.comp.spv` | 42,388 | 14,964 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 29 | `intersect_scene_swrt_bindless_subgroup.comp.spv` | 40,948 | 14,904 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 30 | `intersect_scene_indirect_swrt_bindless_subgroup.comp.spv` | 40,908 | 14,844 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 31 | `intersect_scene_swrt_atlas.comp.spv` | 42,332 | 13,856 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 32 | `intersect_scene_indirect_swrt_atlas.comp.spv` | 42,292 | 13,836 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 33 | `intersect_scene_swrt_bindless.comp.spv` | 40,852 | 13,732 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 34 | `intersect_scene_indirect_swrt_bindless.comp.spv` | 40,812 | 13,716 | 80 | 102 | 0 | 0 | 224 | 3,072 | 6 | 37% |
| 35 | `intersect_area_lights.comp.spv` | 30,892 | 10,836 | 82 | 46 | 0 | 0 | 464 | 4,096 | 5 | 31% |
| 36 | `intersect_scene_shadow_hwrt_atlas.comp.spv` | 36,292 | 10,320 | 85 | 74 | 0 | 0 | 464 | 8,192 | 5 | 31% |
| 37 | `intersect_scene_shadow_hwrt_bindless.comp.spv` | 35,280 | 10,184 | 85 | 78 | 0 | 0 | 464 | 8,192 | 5 | 31% |
| 38 | `convolution_concat_64_9_64_coop_16x16x16_CF16.comp.spv` | 21,120 | 10,052 | 95 | 66 | 0 | 0 | 144 | 2,560 | 5 | 31% |
| 39 | `convolution_Img_9_32_coop_16x16x16_CF16.comp.spv` | 20,516 | 8,948 | 95 | 58 | 0 | 0 | 144 | 2,560 | 5 | 31% |
| 40 | `sort_scatter.comp.spv` | 6,572 | 8,644 | 17 | 27 | 0 | 0 | 0 | 3,072 | 16 | 100% |
| 41 | `spatial_cache_update_compat.comp.spv` | 15,984 | 8,432 | 33 | 62 | 0 | 0 | 0 | 0 | 12 | 75% |
| 42 | `spatial_cache_update.comp.spv` | 15,224 | 6,268 | 33 | 54 | 0 | 0 | 0 | 0 | 12 | 75% |
| 43 | `convolution_concat_96_32_64_coop_16x16x16_CF16.comp.spv` | 16,828 | 6,248 | 100 | 49 | 0 | 0 | 144 | 2,560 | 4 | 25% |
| 44 | `nlm_filter.comp.spv` | 14,348 | 5,996 | 141 | 66 | 0 | 0 | 0 | 6,144 | 3 | 18% |
| 45 | `convolution_32_3_img_fp16.comp.spv` | 12,456 | 5,036 | 37 | 34 | 0 | 0 | 0 | 0 | 12 | 75% |
| 46 | `convolution_32_3_img_fp32.comp.spv` | 12,300 | 4,824 | 37 | 34 | 0 | 0 | 0 | 0 | 12 | 75% |
| 47 | `convolution_concat_64_9_64_fp16.comp.spv` | 16,956 | 4,800 | 44 | 50 | 0 | 0 | 0 | 0 | 10 | 62% |
| 48 | `convolution_concat_64_9_64_fp32.comp.spv` | 16,704 | 4,764 | 45 | 50 | 0 | 0 | 0 | 0 | 10 | 62% |
| 49 | `convolution_32_3_img_coop_16x16x16_CF16.comp.spv` | 14,840 | 4,532 | 84 | 31 | 0 | 0 | 0 | 3,072 | 5 | 31% |
| 50 | `intersect_scene_hwrt_atlas.comp.spv` | 22,060 | 4,504 | 54 | 74 | 0 | 0 | 0 | 4,096 | 9 | 56% |
| 51 | `intersect_scene_indirect_hwrt_atlas.comp.spv` | 22,076 | 4,496 | 54 | 74 | 0 | 0 | 0 | 4,096 | 9 | 56% |
| 52 | `intersect_scene_indirect_hwrt_bindless.comp.spv` | 21,080 | 4,352 | 54 | 74 | 0 | 0 | 0 | 4,096 | 9 | 56% |
| 53 | `intersect_scene_hwrt_bindless.comp.spv` | 21,064 | 4,336 | 54 | 74 | 0 | 0 | 0 | 4,096 | 9 | 56% |
| 54 | `convolution_64_80_Downsample_coop_16x16x16_CF16.comp.spv` | 12,876 | 4,220 | 86 | 42 | 0 | 0 | 0 | 2,560 | 5 | 31% |
| 55 | `convolution_32_48_Downsample_coop_16x16x16_CF16.comp.spv` | 12,860 | 4,200 | 86 | 42 | 0 | 0 | 0 | 2,560 | 5 | 31% |
| 56 | `convolution_32_32_Downsample_coop_16x16x16_CF16.comp.spv` | 12,844 | 4,168 | 86 | 42 | 0 | 0 | 0 | 2,560 | 5 | 31% |
| 57 | `convolution_48_64_Downsample_coop_16x16x16_CF16.comp.spv` | 12,876 | 4,148 | 86 | 42 | 0 | 0 | 0 | 2,560 | 5 | 31% |
| 58 | `convolution_concat_112_48_96_coop_16x16x16_CF16.comp.spv` | 16,860 | 4,056 | 97 | 49 | 0 | 0 | 144 | 2,560 | 4 | 25% |
| 59 | `convolution_concat_96_64_112_coop_16x16x16_CF16.comp.spv` | 16,860 | 4,036 | 96 | 49 | 0 | 0 | 144 | 2,560 | 5 | 31% |
| 60 | `convolution_Img_9_32_fp16.comp.spv` | 13,628 | 3,984 | 44 | 48 | 0 | 0 | 0 | 0 | 10 | 62% |
| 61 | `convolution_Img_9_32_fp32.comp.spv` | 13,376 | 3,956 | 45 | 48 | 0 | 0 | 0 | 0 | 10 | 62% |
| 62 | `debug_rt.comp.spv` | 16,716 | 3,496 | 44 | 28 | 0 | 0 | 0 | 20,480 | 10 | 62% |
| 63 | `primary_ray_gen_adaptive.comp.spv` | 15,220 | 3,212 | 25 | 38 | 0 | 0 | 0 | 0 | 16 | 100% |
| 64 | `primary_ray_gen_simple.comp.spv` | 15,088 | 3,172 | 26 | 46 | 0 | 0 | 0 | 0 | 16 | 100% |
| 65 | `convolution_64_64_coop_16x16x16_CF16.comp.spv` | 12,544 | 2,740 | 67 | 38 | 0 | 0 | 144 | 2,560 | 7 | 43% |
| 66 | `convolution_concat_96_64_112_fp16.comp.spv` | 13,768 | 2,732 | 46 | 36 | 0 | 0 | 0 | 0 | 10 | 62% |
| 67 | `spatial_cache_resolve.comp.spv` | 11,616 | 2,724 | 18 | 84 | 0 | 0 | 0 | 0 | 16 | 100% |
| 68 | `convolution_concat_112_48_96_fp16.comp.spv` | 13,768 | 2,720 | 45 | 34 | 0 | 0 | 0 | 0 | 10 | 62% |
| 69 | `convolution_64_32_coop_16x16x16_CF16.comp.spv` | 12,560 | 2,712 | 67 | 38 | 0 | 0 | 144 | 2,560 | 7 | 43% |
| 70 | `convolution_112_112_coop_16x16x16_CF16.comp.spv` | 12,544 | 2,692 | 67 | 38 | 0 | 0 | 144 | 2,560 | 7 | 43% |
| 71 | `convolution_80_96_coop_16x16x16_CF16.comp.spv` | 12,576 | 2,692 | 67 | 38 | 0 | 0 | 144 | 2,560 | 7 | 43% |
| 72 | `convolution_96_96_coop_16x16x16_CF16.comp.spv` | 12,544 | 2,692 | 67 | 38 | 0 | 0 | 144 | 2,560 | 7 | 43% |
| 73 | `convolution_concat_112_48_96_fp32.comp.spv` | 13,740 | 2,680 | 45 | 34 | 0 | 0 | 0 | 0 | 10 | 62% |
| 74 | `convolution_concat_96_64_112_fp32.comp.spv` | 13,740 | 2,680 | 46 | 36 | 0 | 0 | 0 | 0 | 10 | 62% |
| 75 | `convolution_concat_96_32_64_fp16.comp.spv` | 13,736 | 2,656 | 46 | 36 | 0 | 0 | 0 | 0 | 10 | 62% |
| 76 | `convolution_concat_96_32_64_fp32.comp.spv` | 13,708 | 2,620 | 46 | 36 | 0 | 0 | 0 | 0 | 10 | 62% |
| 77 | `sort_init_count_table.comp.spv` | 3,732 | 2,036 | 13 | 16 | 0 | 0 | 0 | 8,192 | 16 | 100% |
| 78 | `convolution_112_112_fp16.comp.spv` | 10,408 | 2,024 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 79 | `convolution_80_96_fp16.comp.spv` | 10,424 | 2,024 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 80 | `convolution_96_96_fp16.comp.spv` | 10,408 | 2,024 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 81 | `convolution_112_112_fp32.comp.spv` | 10,380 | 1,984 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 82 | `convolution_80_96_fp32.comp.spv` | 10,396 | 1,984 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 83 | `convolution_96_96_fp32.comp.spv` | 10,380 | 1,984 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 84 | `convolution_64_32_fp16.comp.spv` | 10,408 | 1,940 | 46 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 85 | `convolution_64_64_fp16.comp.spv` | 10,408 | 1,940 | 46 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 86 | `convolution_64_32_fp32.comp.spv` | 10,380 | 1,908 | 46 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 87 | `convolution_64_64_fp32.comp.spv` | 10,380 | 1,900 | 46 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 88 | `filter_variance.comp.spv` | 7,808 | 1,864 | 53 | 22 | 0 | 0 | 0 | 3,072 | 9 | 56% |
| 89 | `sort_scan_add.comp.spv` | 4,748 | 1,688 | 19 | 23 | 0 | 0 | 0 | 2,560 | 16 | 100% |
| 90 | `convolution_64_80_Downsample_fp16.comp.spv` | 10,280 | 1,640 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 91 | `convolution_48_64_Downsample_fp16.comp.spv` | 10,280 | 1,628 | 44 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 92 | `convolution_32_48_Downsample_fp16.comp.spv` | 10,264 | 1,624 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 93 | `convolution_32_32_Downsample_fp16.comp.spv` | 10,248 | 1,600 | 45 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 94 | `convolution_64_80_Downsample_fp32.comp.spv` | 10,252 | 1,564 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 95 | `convolution_48_64_Downsample_fp32.comp.spv` | 10,252 | 1,560 | 44 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 96 | `convolution_32_48_Downsample_fp32.comp.spv` | 10,236 | 1,552 | 45 | 30 | 0 | 0 | 0 | 0 | 10 | 62% |
| 97 | `convolution_32_32_Downsample_fp32.comp.spv` | 10,220 | 1,540 | 45 | 29 | 0 | 0 | 0 | 0 | 10 | 62% |
| 98 | `sort_scan.comp.spv` | 4,624 | 1,420 | 19 | 19 | 0 | 0 | 0 | 2,560 | 16 | 100% |
| 99 | `sort_reduce.comp.spv` | 3,280 | 956 | 8 | 15 | 0 | 0 | 0 | 512 | 16 | 100% |
| 100 | `postprocess.comp.spv` | 6,112 | 908 | 16 | 30 | 0 | 0 | 0 | 0 | 16 | 100% |
| 101 | `sort_hash_rays.comp.spv` | 12,168 | 608 | 11 | 22 | 0 | 0 | 0 | 0 | 16 | 100% |
| 102 | `mix_incremental.comp.spv` | 3,764 | 532 | 12 | 32 | 0 | 0 | 0 | 0 | 16 | 100% |
| 103 | `prepare_indir_args.comp.spv` | 2,480 | 356 | 11 | 8 | 0 | 0 | 0 | 0 | 16 | 100% |
| 104 | `sort_reorder_rays.comp.spv` | 2,116 | 260 | 22 | 10 | 0 | 0 | 0 | 0 | 16 | 100% |

Notes:

- Max waves/SIMD is the VGPR-bound estimate: RDNA caps at 16 wave32/SIMD, VGPRs allocate in units of 8, VGPR file is 16K regs/SIMD. Approximate, LDS and dispatch size ignored.
- Offline-mode numbers come from AMD's standalone compiler (same back end as the driver) and may differ slightly from live-driver ISA.
- `4` RT pipeline stage shaders (.rgen/.rchit/.rmiss) were skipped.
- `15` shaders failed to compile:
  - `convolution_112_112_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_32_32_Downsample_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_32_48_Downsample_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_32_3_img_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_48_64_Downsample_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_64_32_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_64_64_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_64_80_Downsample_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_80_96_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_96_96_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_Img_9_32_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_concat_112_48_96_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_concat_64_9_64_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_concat_96_32_64_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
  - `convolution_concat_96_64_112_coop_8x8x16_CF32.comp.spv`: 2.	Running pass "Lower XDL Tensor operations" on function "lgc.shader.CS.main" Exception Code: 0xC0000005
