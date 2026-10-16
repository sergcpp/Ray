.text
.globl ray_conv9_neon
.type ray_conv9_neon, @function
ray_conv9_neon:
.cfi_startproc
    sub sp, sp, 272
    .cfi_def_cfa_offset 272
    stp q8, q9, [sp, #0]
    .cfi_offset d8, -272
    .cfi_offset d9, -264
    stp q10, q11, [sp, #32]
    .cfi_offset d10, -240
    .cfi_offset d11, -232
    stp q12, q13, [sp, #64]
    .cfi_offset d12, -208
    .cfi_offset d13, -200
    stp q14, q15, [sp, #96]
    .cfi_offset d14, -176
    .cfi_offset d15, -168
    stp x19, x20, [sp, #128]
    .cfi_offset x19, -144
    .cfi_offset x20, -136
    stp x21, x22, [sp, #144]
    .cfi_offset x21, -128
    .cfi_offset x22, -120
    stp x23, x24, [sp, #160]
    .cfi_offset x23, -112
    .cfi_offset x24, -104
    stp x25, x26, [sp, #176]
    .cfi_offset x25, -96
    .cfi_offset x26, -88
    stp x27, x28, [sp, #192]
    .cfi_offset x27, -80
    .cfi_offset x28, -72
    mov x19, x0
    str x1, [sp, #208]
    mov w27, w2
    add w25, w2, w4
    add w26, w3, w5
    mov x16, 448
    mul x14, x6, x16
    str x14, [sp, #216]
    mov x16, 192
    mul x14, x7, x16
    str x14, [sp, #224]
    ldr x22, [sp, #272]
    ldr x23, [sp, #280]
    ldr x24, [sp, #288]
    ldr w17, [sp, #296]
    mov x16, 384
    mul x20, x17, x16
    mov w0, w3
    fmov s26, wzr
conv9_neon_y_loop:
    cmp x0, x26
    b.ge conv9_neon_Ly_done
    ldr x12, [sp, #216]
    mov x8, x0
    lsr x8, x8, #1
    mul x8, x8, x12
    add x9, x0, #1
    lsr x9, x9, #1
    mul x9, x9, x12
    add x10, x0, #2
    lsr x10, x10, #1
    mul x10, x10, x12
    cmp x0, #1
    b.lo conv9_neon_LRneg
    sub x11, x0, #1
    lsr x11, x11, #1
    mul x11, x11, x12
    b conv9_neon_LR0
conv9_neon_LRneg:
    neg x11, x12
conv9_neon_LR0:
    str x11, [sp, #232]
    str x8, [sp, #240]
    str x9, [sp, #248]
    str x10, [sp, #256]
    mov x1, x27
conv9_neon_x_loop:
    cmp x1, x25
    b.ge conv9_neon_Lx_done
    mul x8, x0, x20
    add x28, x8, x24
    mov x10, 384
    mul x9, x1, x10
    add x28, x28, x9
    mov x12, xzr
    mov x16, x22
    mov x14, x23
conv9_neon_ii_loop:
    cmp x12, 384
    b.ge conv9_neon_Lii_done
    mov x17, 1344
    mov x13, 2688
    ldr s0, [x14]
    mov v1.16b, v0.16b
    mov v2.16b, v0.16b
    mov v3.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v0.16b
    mov v6.16b, v0.16b
    mov v7.16b, v0.16b
    ldr s8, [x14, #4]
    mov v9.16b, v8.16b
    mov v10.16b, v8.16b
    mov v11.16b, v8.16b
    mov v12.16b, v8.16b
    mov v13.16b, v8.16b
    mov v14.16b, v8.16b
    mov v15.16b, v8.16b
    ldr x11, [sp, #232]
    ldr x8, [sp, #240]
    ldr x9, [sp, #248]
    ldr x10, [sp, #256]
    mov x2, xzr
    add x3, x1, #1
    mov x15, 448
    mov x4, x3
    lsr x4, x4, #1
    sub x4, x4, #1
    mul x4, x4, x15
    add x4, x4, x19
    add x5, x3, #1
    lsr x5, x5, #1
    sub x5, x5, #1
    mul x5, x5, x15
    add x5, x5, x19
    add x6, x3, #2
    lsr x6, x6, #1
    sub x6, x6, #1
    mul x6, x6, x15
    add x6, x6, x19
    add x7, x3, #3
    lsr x7, x7, #1
    sub x7, x7, #1
    mul x7, x7, x15
    add x7, x7, x19
    mov x3, 5760
    add x15, x16, x3
conv9_neon_j1w0:
    ldr q20, [x16, x2]
    ldr q21, [x16, x17]
    ldr q22, [x16, x13]
    ldr q23, [x15, x2]
    ldr q24, [x15, x17]
    ldr q25, [x15, x13]
    ldr q16, [x4, x11]
    ldr q17, [x5, x11]
    ldr q18, [x6, x11]
    ldr q19, [x7, x11]
    fmla v0.4s, v16.4s, v20.4s
    fmla v8.4s, v16.4s, v23.4s
    fmla v1.4s, v17.4s, v20.4s
    fmla v9.4s, v17.4s, v23.4s
    fmla v2.4s, v18.4s, v20.4s
    fmla v10.4s, v18.4s, v23.4s
    fmla v3.4s, v19.4s, v20.4s
    fmla v11.4s, v19.4s, v23.4s
    ldr q16, [x4, x8]
    ldr q17, [x5, x8]
    ldr q18, [x6, x8]
    ldr q19, [x7, x8]
    fmla v0.4s, v16.4s, v21.4s
    fmla v8.4s, v16.4s, v24.4s
    fmla v4.4s, v16.4s, v20.4s
    fmla v12.4s, v16.4s, v23.4s
    fmla v1.4s, v17.4s, v21.4s
    fmla v9.4s, v17.4s, v24.4s
    fmla v5.4s, v17.4s, v20.4s
    fmla v13.4s, v17.4s, v23.4s
    fmla v2.4s, v18.4s, v21.4s
    fmla v10.4s, v18.4s, v24.4s
    fmla v6.4s, v18.4s, v20.4s
    fmla v14.4s, v18.4s, v23.4s
    fmla v3.4s, v19.4s, v21.4s
    fmla v11.4s, v19.4s, v24.4s
    fmla v7.4s, v19.4s, v20.4s
    fmla v15.4s, v19.4s, v23.4s
    ldr q16, [x4, x9]
    ldr q17, [x5, x9]
    ldr q18, [x6, x9]
    ldr q19, [x7, x9]
    fmla v0.4s, v16.4s, v22.4s
    fmla v8.4s, v16.4s, v25.4s
    fmla v4.4s, v16.4s, v21.4s
    fmla v12.4s, v16.4s, v24.4s
    fmla v1.4s, v17.4s, v22.4s
    fmla v9.4s, v17.4s, v25.4s
    fmla v5.4s, v17.4s, v21.4s
    fmla v13.4s, v17.4s, v24.4s
    fmla v2.4s, v18.4s, v22.4s
    fmla v10.4s, v18.4s, v25.4s
    fmla v6.4s, v18.4s, v21.4s
    fmla v14.4s, v18.4s, v24.4s
    fmla v3.4s, v19.4s, v22.4s
    fmla v11.4s, v19.4s, v25.4s
    fmla v7.4s, v19.4s, v21.4s
    fmla v15.4s, v19.4s, v24.4s
    ldr q16, [x4, x10]
    ldr q17, [x5, x10]
    ldr q18, [x6, x10]
    ldr q19, [x7, x10]
    fmla v4.4s, v16.4s, v22.4s
    fmla v12.4s, v16.4s, v25.4s
    fmla v5.4s, v17.4s, v22.4s
    fmla v13.4s, v17.4s, v25.4s
    fmla v6.4s, v18.4s, v22.4s
    fmla v14.4s, v18.4s, v25.4s
    fmla v7.4s, v19.4s, v22.4s
    fmla v15.4s, v19.4s, v25.4s
    add x2, x2, 16
    add x17, x17, 16
    add x13, x13, 16
    add x4, x4, 16
    add x5, x5, 16
    add x6, x6, 16
    add x7, x7, 16
    cmp x2, 448
    b.lt conv9_neon_j1w0
    add x3, x1, #2
    mov x15, 448
    mov x4, x3
    lsr x4, x4, #1
    sub x4, x4, #1
    mul x4, x4, x15
    add x4, x4, x19
    add x5, x3, #1
    lsr x5, x5, #1
    sub x5, x5, #1
    mul x5, x5, x15
    add x5, x5, x19
    add x6, x3, #2
    lsr x6, x6, #1
    sub x6, x6, #1
    mul x6, x6, x15
    add x6, x6, x19
    add x7, x3, #3
    lsr x7, x7, #1
    sub x7, x7, #1
    mul x7, x7, x15
    add x7, x7, x19
    mov x3, 5760
    add x15, x16, x3
conv9_neon_j1w1:
    ldr q20, [x16, x2]
    ldr q21, [x16, x17]
    ldr q22, [x16, x13]
    ldr q23, [x15, x2]
    ldr q24, [x15, x17]
    ldr q25, [x15, x13]
    ldr q16, [x4, x11]
    ldr q17, [x5, x11]
    ldr q18, [x6, x11]
    ldr q19, [x7, x11]
    fmla v0.4s, v16.4s, v20.4s
    fmla v8.4s, v16.4s, v23.4s
    fmla v1.4s, v17.4s, v20.4s
    fmla v9.4s, v17.4s, v23.4s
    fmla v2.4s, v18.4s, v20.4s
    fmla v10.4s, v18.4s, v23.4s
    fmla v3.4s, v19.4s, v20.4s
    fmla v11.4s, v19.4s, v23.4s
    ldr q16, [x4, x8]
    ldr q17, [x5, x8]
    ldr q18, [x6, x8]
    ldr q19, [x7, x8]
    fmla v0.4s, v16.4s, v21.4s
    fmla v8.4s, v16.4s, v24.4s
    fmla v4.4s, v16.4s, v20.4s
    fmla v12.4s, v16.4s, v23.4s
    fmla v1.4s, v17.4s, v21.4s
    fmla v9.4s, v17.4s, v24.4s
    fmla v5.4s, v17.4s, v20.4s
    fmla v13.4s, v17.4s, v23.4s
    fmla v2.4s, v18.4s, v21.4s
    fmla v10.4s, v18.4s, v24.4s
    fmla v6.4s, v18.4s, v20.4s
    fmla v14.4s, v18.4s, v23.4s
    fmla v3.4s, v19.4s, v21.4s
    fmla v11.4s, v19.4s, v24.4s
    fmla v7.4s, v19.4s, v20.4s
    fmla v15.4s, v19.4s, v23.4s
    ldr q16, [x4, x9]
    ldr q17, [x5, x9]
    ldr q18, [x6, x9]
    ldr q19, [x7, x9]
    fmla v0.4s, v16.4s, v22.4s
    fmla v8.4s, v16.4s, v25.4s
    fmla v4.4s, v16.4s, v21.4s
    fmla v12.4s, v16.4s, v24.4s
    fmla v1.4s, v17.4s, v22.4s
    fmla v9.4s, v17.4s, v25.4s
    fmla v5.4s, v17.4s, v21.4s
    fmla v13.4s, v17.4s, v24.4s
    fmla v2.4s, v18.4s, v22.4s
    fmla v10.4s, v18.4s, v25.4s
    fmla v6.4s, v18.4s, v21.4s
    fmla v14.4s, v18.4s, v24.4s
    fmla v3.4s, v19.4s, v22.4s
    fmla v11.4s, v19.4s, v25.4s
    fmla v7.4s, v19.4s, v21.4s
    fmla v15.4s, v19.4s, v24.4s
    ldr q16, [x4, x10]
    ldr q17, [x5, x10]
    ldr q18, [x6, x10]
    ldr q19, [x7, x10]
    fmla v4.4s, v16.4s, v22.4s
    fmla v12.4s, v16.4s, v25.4s
    fmla v5.4s, v17.4s, v22.4s
    fmla v13.4s, v17.4s, v25.4s
    fmla v6.4s, v18.4s, v22.4s
    fmla v14.4s, v18.4s, v25.4s
    fmla v7.4s, v19.4s, v22.4s
    fmla v15.4s, v19.4s, v25.4s
    add x2, x2, 16
    add x17, x17, 16
    add x13, x13, 16
    add x4, x4, 16
    add x5, x5, 16
    add x6, x6, 16
    add x7, x7, 16
    cmp x2, 896
    b.lt conv9_neon_j1w1
    add x3, x1, #3
    mov x15, 448
    mov x4, x3
    lsr x4, x4, #1
    sub x4, x4, #1
    mul x4, x4, x15
    add x4, x4, x19
    add x5, x3, #1
    lsr x5, x5, #1
    sub x5, x5, #1
    mul x5, x5, x15
    add x5, x5, x19
    add x6, x3, #2
    lsr x6, x6, #1
    sub x6, x6, #1
    mul x6, x6, x15
    add x6, x6, x19
    add x7, x3, #3
    lsr x7, x7, #1
    sub x7, x7, #1
    mul x7, x7, x15
    add x7, x7, x19
    mov x3, 5760
    add x15, x16, x3
conv9_neon_j1w2:
    ldr q20, [x16, x2]
    ldr q21, [x16, x17]
    ldr q22, [x16, x13]
    ldr q23, [x15, x2]
    ldr q24, [x15, x17]
    ldr q25, [x15, x13]
    ldr q16, [x4, x11]
    ldr q17, [x5, x11]
    ldr q18, [x6, x11]
    ldr q19, [x7, x11]
    fmla v0.4s, v16.4s, v20.4s
    fmla v8.4s, v16.4s, v23.4s
    fmla v1.4s, v17.4s, v20.4s
    fmla v9.4s, v17.4s, v23.4s
    fmla v2.4s, v18.4s, v20.4s
    fmla v10.4s, v18.4s, v23.4s
    fmla v3.4s, v19.4s, v20.4s
    fmla v11.4s, v19.4s, v23.4s
    ldr q16, [x4, x8]
    ldr q17, [x5, x8]
    ldr q18, [x6, x8]
    ldr q19, [x7, x8]
    fmla v0.4s, v16.4s, v21.4s
    fmla v8.4s, v16.4s, v24.4s
    fmla v4.4s, v16.4s, v20.4s
    fmla v12.4s, v16.4s, v23.4s
    fmla v1.4s, v17.4s, v21.4s
    fmla v9.4s, v17.4s, v24.4s
    fmla v5.4s, v17.4s, v20.4s
    fmla v13.4s, v17.4s, v23.4s
    fmla v2.4s, v18.4s, v21.4s
    fmla v10.4s, v18.4s, v24.4s
    fmla v6.4s, v18.4s, v20.4s
    fmla v14.4s, v18.4s, v23.4s
    fmla v3.4s, v19.4s, v21.4s
    fmla v11.4s, v19.4s, v24.4s
    fmla v7.4s, v19.4s, v20.4s
    fmla v15.4s, v19.4s, v23.4s
    ldr q16, [x4, x9]
    ldr q17, [x5, x9]
    ldr q18, [x6, x9]
    ldr q19, [x7, x9]
    fmla v0.4s, v16.4s, v22.4s
    fmla v8.4s, v16.4s, v25.4s
    fmla v4.4s, v16.4s, v21.4s
    fmla v12.4s, v16.4s, v24.4s
    fmla v1.4s, v17.4s, v22.4s
    fmla v9.4s, v17.4s, v25.4s
    fmla v5.4s, v17.4s, v21.4s
    fmla v13.4s, v17.4s, v24.4s
    fmla v2.4s, v18.4s, v22.4s
    fmla v10.4s, v18.4s, v25.4s
    fmla v6.4s, v18.4s, v21.4s
    fmla v14.4s, v18.4s, v24.4s
    fmla v3.4s, v19.4s, v22.4s
    fmla v11.4s, v19.4s, v25.4s
    fmla v7.4s, v19.4s, v21.4s
    fmla v15.4s, v19.4s, v24.4s
    ldr q16, [x4, x10]
    ldr q17, [x5, x10]
    ldr q18, [x6, x10]
    ldr q19, [x7, x10]
    fmla v4.4s, v16.4s, v22.4s
    fmla v12.4s, v16.4s, v25.4s
    fmla v5.4s, v17.4s, v22.4s
    fmla v13.4s, v17.4s, v25.4s
    fmla v6.4s, v18.4s, v22.4s
    fmla v14.4s, v18.4s, v25.4s
    fmla v7.4s, v19.4s, v22.4s
    fmla v15.4s, v19.4s, v25.4s
    add x2, x2, 16
    add x17, x17, 16
    add x13, x13, 16
    add x4, x4, 16
    add x5, x5, 16
    add x6, x6, 16
    add x7, x7, 16
    cmp x2, 1344
    b.lt conv9_neon_j1w2
    ldr x8, [sp, #208]
    sub x9, x1, #1
    mov x10, 192
    mul x9, x9, x10
    add x8, x8, x9
    ldr x9, [sp, #224]
    mul x10, x0, x9
    add x8, x8, x10
    mov x5, x8
    sub x4, x5, x9
    add x6, x5, x9
    add x7, x6, x9
    add x15, x16, #4032
    mov x3, 5760
    add x3, x15, x3
    mov x17, 576
    mov x13, 1152
    mov x2, xzr
conv9_neon_j2_loop:
    ldr q20, [x15, x2]
    ldr q21, [x15, x17]
    ldr q22, [x15, x13]
    ldr q23, [x3, x2]
    ldr q24, [x3, x17]
    ldr q25, [x3, x13]
    ldr q16, [x4]
    ldr q17, [x5]
    ldr q18, [x6]
    ldr q19, [x7]
    fmla v0.4s, v16.4s, v20.4s
    fmla v0.4s, v17.4s, v21.4s
    fmla v0.4s, v18.4s, v22.4s
    fmla v4.4s, v17.4s, v20.4s
    fmla v4.4s, v18.4s, v21.4s
    fmla v4.4s, v19.4s, v22.4s
    fmla v8.4s, v16.4s, v23.4s
    fmla v8.4s, v17.4s, v24.4s
    fmla v8.4s, v18.4s, v25.4s
    fmla v12.4s, v17.4s, v23.4s
    fmla v12.4s, v18.4s, v24.4s
    fmla v12.4s, v19.4s, v25.4s
    ldr q16, [x4, #192]
    ldr q17, [x5, #192]
    ldr q18, [x6, #192]
    ldr q19, [x7, #192]
    fmla v1.4s, v16.4s, v20.4s
    fmla v1.4s, v17.4s, v21.4s
    fmla v1.4s, v18.4s, v22.4s
    fmla v5.4s, v17.4s, v20.4s
    fmla v5.4s, v18.4s, v21.4s
    fmla v5.4s, v19.4s, v22.4s
    fmla v9.4s, v16.4s, v23.4s
    fmla v9.4s, v17.4s, v24.4s
    fmla v9.4s, v18.4s, v25.4s
    fmla v13.4s, v17.4s, v23.4s
    fmla v13.4s, v18.4s, v24.4s
    fmla v13.4s, v19.4s, v25.4s
    ldr q16, [x4, #384]
    ldr q17, [x5, #384]
    ldr q18, [x6, #384]
    ldr q19, [x7, #384]
    fmla v2.4s, v16.4s, v20.4s
    fmla v2.4s, v17.4s, v21.4s
    fmla v2.4s, v18.4s, v22.4s
    fmla v6.4s, v17.4s, v20.4s
    fmla v6.4s, v18.4s, v21.4s
    fmla v6.4s, v19.4s, v22.4s
    fmla v10.4s, v16.4s, v23.4s
    fmla v10.4s, v17.4s, v24.4s
    fmla v10.4s, v18.4s, v25.4s
    fmla v14.4s, v17.4s, v23.4s
    fmla v14.4s, v18.4s, v24.4s
    fmla v14.4s, v19.4s, v25.4s
    ldr q16, [x4, #576]
    ldr q17, [x5, #576]
    ldr q18, [x6, #576]
    ldr q19, [x7, #576]
    fmla v3.4s, v16.4s, v20.4s
    fmla v3.4s, v17.4s, v21.4s
    fmla v3.4s, v18.4s, v22.4s
    fmla v7.4s, v17.4s, v20.4s
    fmla v7.4s, v18.4s, v21.4s
    fmla v7.4s, v19.4s, v22.4s
    fmla v11.4s, v16.4s, v23.4s
    fmla v11.4s, v17.4s, v24.4s
    fmla v11.4s, v18.4s, v25.4s
    fmla v15.4s, v17.4s, v23.4s
    fmla v15.4s, v18.4s, v24.4s
    fmla v15.4s, v19.4s, v25.4s
    add x2, x2, 16
    add x17, x17, 16
    add x13, x13, 16
    add x4, x4, 16
    add x5, x5, 16
    add x6, x6, 16
    add x7, x7, 16
    cmp x2, 576
    b.lt conv9_neon_j2_loop
    add x8, x0, #1
    cmp x8, x26
    b.ge conv9_neon_Lrow0
    add x28, x28, x20
    add x8, x1, #0
    cmp x8, x25
    b.ge conv9_neon_Lc1_0
    faddp v4.4s, v4.4s, v4.4s
    faddp s4, v4.2s
    faddp v12.4s, v12.4s, v12.4s
    faddp s12, v12.2s
    fmax s4, s4, s26
    fmax s12, s12, s26
    add x8, x28, #0
    str s4, [x8, x12]
    add x8, x8, #4
    str s12, [x8, x12]
conv9_neon_Lc1_0:
    add x8, x1, #1
    cmp x8, x25
    b.ge conv9_neon_Lc1_1
    faddp v5.4s, v5.4s, v5.4s
    faddp s5, v5.2s
    faddp v13.4s, v13.4s, v13.4s
    faddp s13, v13.2s
    fmax s5, s5, s26
    fmax s13, s13, s26
    add x8, x28, #384
    str s5, [x8, x12]
    add x8, x8, #4
    str s13, [x8, x12]
conv9_neon_Lc1_1:
    add x8, x1, #2
    cmp x8, x25
    b.ge conv9_neon_Lc1_2
    faddp v6.4s, v6.4s, v6.4s
    faddp s6, v6.2s
    faddp v14.4s, v14.4s, v14.4s
    faddp s14, v14.2s
    fmax s6, s6, s26
    fmax s14, s14, s26
    add x8, x28, #768
    str s6, [x8, x12]
    add x8, x8, #4
    str s14, [x8, x12]
conv9_neon_Lc1_2:
    add x8, x1, #3
    cmp x8, x25
    b.ge conv9_neon_Lc1_3
    faddp v7.4s, v7.4s, v7.4s
    faddp s7, v7.2s
    faddp v15.4s, v15.4s, v15.4s
    faddp s15, v15.2s
    fmax s7, s7, s26
    fmax s15, s15, s26
    add x8, x28, #1152
    str s7, [x8, x12]
    add x8, x8, #4
    str s15, [x8, x12]
conv9_neon_Lc1_3:
    sub x28, x28, x20
conv9_neon_Lrow0:
    add x8, x1, #0
    cmp x8, x25
    b.ge conv9_neon_Lc0_0
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    faddp v8.4s, v8.4s, v8.4s
    faddp s8, v8.2s
    fmax s0, s0, s26
    fmax s8, s8, s26
    add x8, x28, #0
    str s0, [x8, x12]
    add x8, x8, #4
    str s8, [x8, x12]
conv9_neon_Lc0_0:
    add x8, x1, #1
    cmp x8, x25
    b.ge conv9_neon_Lc0_1
    faddp v1.4s, v1.4s, v1.4s
    faddp s1, v1.2s
    faddp v9.4s, v9.4s, v9.4s
    faddp s9, v9.2s
    fmax s1, s1, s26
    fmax s9, s9, s26
    add x8, x28, #384
    str s1, [x8, x12]
    add x8, x8, #4
    str s9, [x8, x12]
conv9_neon_Lc0_1:
    add x8, x1, #2
    cmp x8, x25
    b.ge conv9_neon_Lc0_2
    faddp v2.4s, v2.4s, v2.4s
    faddp s2, v2.2s
    faddp v10.4s, v10.4s, v10.4s
    faddp s10, v10.2s
    fmax s2, s2, s26
    fmax s10, s10, s26
    add x8, x28, #768
    str s2, [x8, x12]
    add x8, x8, #4
    str s10, [x8, x12]
conv9_neon_Lc0_2:
    add x8, x1, #3
    cmp x8, x25
    b.ge conv9_neon_Lc0_3
    faddp v3.4s, v3.4s, v3.4s
    faddp s3, v3.2s
    faddp v11.4s, v11.4s, v11.4s
    faddp s11, v11.2s
    fmax s3, s3, s26
    fmax s11, s11, s26
    add x8, x28, #1152
    str s3, [x8, x12]
    add x8, x8, #4
    str s11, [x8, x12]
conv9_neon_Lc0_3:
    mov x8, 11520
    add x16, x16, x8
    add x14, x14, #8
    add x12, x12, #8
    b conv9_neon_ii_loop
conv9_neon_Lii_done:
    add x1, x1, 4
    b conv9_neon_x_loop
conv9_neon_Lx_done:
    add x0, x0, #2
    b conv9_neon_y_loop
conv9_neon_Ly_done:
    ldp q8, q9, [sp, #0]
    ldp q10, q11, [sp, #32]
    ldp q12, q13, [sp, #64]
    ldp q14, q15, [sp, #96]
    ldp x19, x20, [sp, #128]
    ldp x21, x22, [sp, #144]
    ldp x23, x24, [sp, #160]
    ldp x25, x26, [sp, #176]
    ldp x27, x28, [sp, #192]
    add sp, sp, 272
    .cfi_def_cfa_offset 0
    ret 
.cfi_endproc
.size ray_conv9_neon, .-ray_conv9_neon
.section .note.GNU-stack,"",@progbits
