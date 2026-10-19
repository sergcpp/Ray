.arch armv8.2-a+fp16
.section __TEXT,__text,regular,pure_instructions
.globl _ray_conv1_neon_f16
_ray_conv1_neon_f16:
    sub sp, sp, 208
    stp q8, q9, [sp, #0]
    stp q10, q11, [sp, #32]
    stp q12, q13, [sp, #64]
    stp q14, q15, [sp, #96]
    stp x19, x20, [sp, #128]
    stp x21, x22, [sp, #144]
    stp x23, x24, [sp, #160]
    stp x25, x26, [sp, #176]
    stp x27, x28, [sp, #192]
    mov x19, x0
    mov w27, w1
    add w25, w1, w3
    add w26, w2, w4
    mov x16, 64
    mul x21, x5, x16
    mov x22, x6
    mov x23, x7
    ldr x24, [sp, #208]
    ldr w17, [sp, #216]
    mov x16, 64
    mul x20, x17, x16
    mov w0, w2
    fmov s26, wzr
conv1_neon_y_loop:
    cmp x0, x26
    b.ge conv1_neon_Ly_done
    mov x1, x27
conv1_neon_x_loop:
    cmp x1, x25
    b.ge conv1_neon_Lx_done
    mov x8, x0
    lsr x8, x8, #1
    mul x8, x8, x20
    add x28, x8, x24
    mov x9, x1
    lsr x9, x9, #1
    mov x10, 64
    mul x9, x9, x10
    add x28, x28, x9
    mov x12, xzr
    mov x16, x22
    add x17, x16, #192
    add x13, x17, #192
    mov x15, x23
conv1_neon_ii_loop:
    cmp x12, 64
    b.ge conv1_neon_Lii_done
    mul x8, x0, x21
    sub x9, x1, #1
    mov x10, 64
    mul x9, x9, x10
    add x8, x8, x9
    add x5, x19, x8
    sub x4, x5, x21
    add x6, x5, x21
    add x7, x6, x21
    add x9, x16, #576
    add x10, x17, #576
    add x11, x13, #576
    ldr h0, [x15]
    mov v1.16b, v0.16b
    mov v2.16b, v0.16b
    mov v3.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v0.16b
    mov v6.16b, v0.16b
    mov v7.16b, v0.16b
    ldr h8, [x15, #2]
    mov v9.16b, v8.16b
    mov v10.16b, v8.16b
    mov v11.16b, v8.16b
    mov v12.16b, v8.16b
    mov v13.16b, v8.16b
    mov v14.16b, v8.16b
    mov v15.16b, v8.16b
    mov x2, xzr
conv1_neon_j_loop:
    ldr q20, [x16, x2]
    ldr q21, [x17, x2]
    ldr q22, [x13, x2]
    ldr q23, [x9, x2]
    ldr q24, [x10, x2]
    ldr q25, [x11, x2]
    ldr q16, [x4]
    ldr q17, [x5]
    ldr q18, [x6]
    ldr q19, [x7]
    fmla v0.8h, v16.8h, v20.8h
    fmla v0.8h, v17.8h, v21.8h
    fmla v0.8h, v18.8h, v22.8h
    fmla v4.8h, v17.8h, v20.8h
    fmla v4.8h, v18.8h, v21.8h
    fmla v4.8h, v19.8h, v22.8h
    fmla v8.8h, v16.8h, v23.8h
    fmla v8.8h, v17.8h, v24.8h
    fmla v8.8h, v18.8h, v25.8h
    fmla v12.8h, v17.8h, v23.8h
    fmla v12.8h, v18.8h, v24.8h
    fmla v12.8h, v19.8h, v25.8h
    ldr q16, [x4, #64]
    ldr q17, [x5, #64]
    ldr q18, [x6, #64]
    ldr q19, [x7, #64]
    fmla v1.8h, v16.8h, v20.8h
    fmla v1.8h, v17.8h, v21.8h
    fmla v1.8h, v18.8h, v22.8h
    fmla v5.8h, v17.8h, v20.8h
    fmla v5.8h, v18.8h, v21.8h
    fmla v5.8h, v19.8h, v22.8h
    fmla v9.8h, v16.8h, v23.8h
    fmla v9.8h, v17.8h, v24.8h
    fmla v9.8h, v18.8h, v25.8h
    fmla v13.8h, v17.8h, v23.8h
    fmla v13.8h, v18.8h, v24.8h
    fmla v13.8h, v19.8h, v25.8h
    ldr q16, [x4, #128]
    ldr q17, [x5, #128]
    ldr q18, [x6, #128]
    ldr q19, [x7, #128]
    fmla v2.8h, v16.8h, v20.8h
    fmla v2.8h, v17.8h, v21.8h
    fmla v2.8h, v18.8h, v22.8h
    fmla v6.8h, v17.8h, v20.8h
    fmla v6.8h, v18.8h, v21.8h
    fmla v6.8h, v19.8h, v22.8h
    fmla v10.8h, v16.8h, v23.8h
    fmla v10.8h, v17.8h, v24.8h
    fmla v10.8h, v18.8h, v25.8h
    fmla v14.8h, v17.8h, v23.8h
    fmla v14.8h, v18.8h, v24.8h
    fmla v14.8h, v19.8h, v25.8h
    ldr q16, [x4, #192]
    ldr q17, [x5, #192]
    ldr q18, [x6, #192]
    ldr q19, [x7, #192]
    fmla v3.8h, v16.8h, v20.8h
    fmla v3.8h, v17.8h, v21.8h
    fmla v3.8h, v18.8h, v22.8h
    fmla v7.8h, v17.8h, v20.8h
    fmla v7.8h, v18.8h, v21.8h
    fmla v7.8h, v19.8h, v22.8h
    fmla v11.8h, v16.8h, v23.8h
    fmla v11.8h, v17.8h, v24.8h
    fmla v11.8h, v18.8h, v25.8h
    fmla v15.8h, v17.8h, v23.8h
    fmla v15.8h, v18.8h, v24.8h
    fmla v15.8h, v19.8h, v25.8h
    add x2, x2, 16
    add x4, x4, 16
    add x5, x5, 16
    add x6, x6, 16
    add x7, x7, 16
    cmp x2, 192
    b.lt conv1_neon_j_loop
    faddp v12.8h, v12.8h, v12.8h
    faddp v12.4h, v12.4h, v12.4h
    faddp h12, v12.2h
    fmax h12, h12, h26
    faddp v13.8h, v13.8h, v13.8h
    faddp v13.4h, v13.4h, v13.4h
    faddp h13, v13.2h
    fmax h13, h13, h26
    faddp v14.8h, v14.8h, v14.8h
    faddp v14.4h, v14.4h, v14.4h
    faddp h14, v14.2h
    fmax h14, h14, h26
    faddp v15.8h, v15.8h, v15.8h
    faddp v15.4h, v15.4h, v15.4h
    faddp h15, v15.2h
    fmax h15, h15, h26
    faddp v8.8h, v8.8h, v8.8h
    faddp v8.4h, v8.4h, v8.4h
    faddp h8, v8.2h
    fmax h8, h8, h26
    faddp v9.8h, v9.8h, v9.8h
    faddp v9.4h, v9.4h, v9.4h
    faddp h9, v9.2h
    fmax h9, h9, h26
    faddp v10.8h, v10.8h, v10.8h
    faddp v10.4h, v10.4h, v10.4h
    faddp h10, v10.2h
    fmax h10, h10, h26
    faddp v11.8h, v11.8h, v11.8h
    faddp v11.4h, v11.4h, v11.4h
    faddp h11, v11.2h
    fmax h11, h11, h26
    faddp v4.8h, v4.8h, v4.8h
    faddp v4.4h, v4.4h, v4.4h
    faddp h4, v4.2h
    fmax h4, h4, h26
    faddp v5.8h, v5.8h, v5.8h
    faddp v5.4h, v5.4h, v5.4h
    faddp h5, v5.2h
    fmax h5, h5, h26
    faddp v6.8h, v6.8h, v6.8h
    faddp v6.4h, v6.4h, v6.4h
    faddp h6, v6.2h
    fmax h6, h6, h26
    faddp v7.8h, v7.8h, v7.8h
    faddp v7.4h, v7.4h, v7.4h
    faddp h7, v7.2h
    fmax h7, h7, h26
    faddp v0.8h, v0.8h, v0.8h
    faddp v0.4h, v0.4h, v0.4h
    faddp h0, v0.2h
    fmax h0, h0, h26
    faddp v1.8h, v1.8h, v1.8h
    faddp v1.4h, v1.4h, v1.4h
    faddp h1, v1.2h
    fmax h1, h1, h26
    faddp v2.8h, v2.8h, v2.8h
    faddp v2.4h, v2.4h, v2.4h
    faddp h2, v2.2h
    fmax h2, h2, h26
    faddp v3.8h, v3.8h, v3.8h
    faddp v3.4h, v3.4h, v3.4h
    faddp h3, v3.2h
    fmax h3, h3, h26
    add x8, x1, #0
    cmp x8, x25
    b.ge conv1_neon_Lp0
    fmax h1, h1, h0
    fmax h5, h5, h4
    fmax h5, h5, h1
    fmax h5, h5, h26
    fmax h9, h9, h8
    fmax h13, h13, h12
    fmax h13, h13, h9
    fmax h13, h13, h26
    add x8, x28, #0
    str h5, [x8, x12]
    add x8, x8, #2
    str h13, [x8, x12]
conv1_neon_Lp0:
    add x8, x1, #2
    cmp x8, x25
    b.ge conv1_neon_Lp2
    fmax h3, h3, h2
    fmax h7, h7, h6
    fmax h7, h7, h3
    fmax h7, h7, h26
    fmax h11, h11, h10
    fmax h15, h15, h14
    fmax h15, h15, h11
    fmax h15, h15, h26
    add x8, x28, #64
    str h7, [x8, x12]
    add x8, x8, #2
    str h15, [x8, x12]
conv1_neon_Lp2:
    mov x8, 1152
    add x16, x16, x8
    add x17, x17, x8
    add x13, x13, x8
    add x15, x15, #4
    add x12, x12, #4
    b conv1_neon_ii_loop
conv1_neon_Lii_done:
    add x1, x1, 4
    b conv1_neon_x_loop
conv1_neon_Lx_done:
    add x0, x0, #2
    b conv1_neon_y_loop
conv1_neon_Ly_done:
    ldp q8, q9, [sp, #0]
    ldp q10, q11, [sp, #32]
    ldp q12, q13, [sp, #64]
    ldp q14, q15, [sp, #96]
    ldp x19, x20, [sp, #128]
    ldp x21, x22, [sp, #144]
    ldp x23, x24, [sp, #160]
    ldp x25, x26, [sp, #176]
    ldp x27, x28, [sp, #192]
    add sp, sp, 208
    ret 
