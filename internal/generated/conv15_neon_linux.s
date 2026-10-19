.arch armv8.2-a+fp16
.text
.globl ray_conv15_neon_f16
.type ray_conv15_neon_f16, @function
ray_conv15_neon_f16:
.cfi_startproc
    sub sp, sp, 304
    .cfi_def_cfa_offset 304
    stp q8, q9, [sp, #0]
    .cfi_offset d8, -304
    .cfi_offset d9, -296
    stp q10, q11, [sp, #32]
    .cfi_offset d10, -272
    .cfi_offset d11, -264
    stp q12, q13, [sp, #64]
    .cfi_offset d12, -240
    .cfi_offset d13, -232
    stp q14, q15, [sp, #96]
    .cfi_offset d14, -208
    .cfi_offset d15, -200
    stp x19, x20, [sp, #128]
    .cfi_offset x19, -176
    .cfi_offset x20, -168
    stp x21, x22, [sp, #144]
    .cfi_offset x21, -160
    .cfi_offset x22, -152
    stp x23, x24, [sp, #160]
    .cfi_offset x23, -144
    .cfi_offset x24, -136
    stp x25, x26, [sp, #176]
    .cfi_offset x25, -128
    .cfi_offset x26, -120
    stp x27, x28, [sp, #192]
    .cfi_offset x27, -112
    .cfi_offset x28, -104
    mov x19, x0
    mov w27, w1
    add w25, w1, w3
    add w26, w2, w4
    mov x16, 64
    mul x21, x5, x16
    mov x22, x6
    mov x23, x7
    ldr x24, [sp, #304]
    ldr w17, [sp, #312]
    mov x16, 16
    mul x20, x17, x16
    mov w0, w2
    stp x29, x30, [sp, #280]
    .cfi_offset x29, -24
    .cfi_offset x30, -16
    fmov s15, wzr
conv15_neon_y_loop:
    cmp x0, x26
    b.ge conv15_neon_Ly_done
    mov x1, x27
conv15_neon_x_loop:
    cmp x1, x25
    b.ge conv15_neon_Lx_done
    mov x8, x0
    mul x8, x8, x20
    add x28, x8, x24
    mov x9, x1
    mov x10, 16
    mul x9, x9, x10
    add x28, x28, x9
    mov x12, xzr
    mov x16, x22
    add x17, x16, #192
    add x13, x17, #192
    mov x15, x23
conv15_neon_ii_loop:
    cmp x12, 12
    b.ge conv15_neon_Lii_done
    mul x8, x0, x21
    sub x9, x1, #1
    mov x10, 64
    mul x9, x9, x10
    add x8, x8, x9
    add x5, x19, x8
    sub x4, x5, x21
    add x6, x5, x21
    add x7, x6, x21
    ldr h0, [x15]
    mov v1.16b, v0.16b
    mov v2.16b, v0.16b
    mov v3.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v0.16b
    mov v6.16b, v0.16b
    mov v7.16b, v0.16b
    mov v4.16b, v0.16b
    mov v5.16b, v1.16b
    mov v6.16b, v2.16b
    mov v7.16b, v3.16b
    mov v8.16b, v4.16b
    mov v9.16b, v5.16b
    mov v10.16b, v6.16b
    mov v11.16b, v7.16b
    mov x2, xzr
conv15_neon_j_loop:
    ldr q12, [x16, x2]
    ldr q13, [x17, x2]
    ldr q14, [x13, x2]
    ldr q8, [x4]
    ldr q9, [x5]
    ldr q10, [x6]
    ldr q11, [x7]
    fmla v0.8h, v8.8h, v12.8h
    fmla v0.8h, v9.8h, v13.8h
    fmla v0.8h, v10.8h, v14.8h
    fmla v4.8h, v9.8h, v12.8h
    fmla v4.8h, v10.8h, v13.8h
    fmla v4.8h, v11.8h, v14.8h
    ldr q8, [x4, #64]
    ldr q9, [x5, #64]
    ldr q10, [x6, #64]
    ldr q11, [x7, #64]
    fmla v1.8h, v8.8h, v12.8h
    fmla v1.8h, v9.8h, v13.8h
    fmla v1.8h, v10.8h, v14.8h
    fmla v5.8h, v9.8h, v12.8h
    fmla v5.8h, v10.8h, v13.8h
    fmla v5.8h, v11.8h, v14.8h
    ldr q8, [x4, #128]
    ldr q9, [x5, #128]
    ldr q10, [x6, #128]
    ldr q11, [x7, #128]
    fmla v2.8h, v8.8h, v12.8h
    fmla v2.8h, v9.8h, v13.8h
    fmla v2.8h, v10.8h, v14.8h
    fmla v6.8h, v9.8h, v12.8h
    fmla v6.8h, v10.8h, v13.8h
    fmla v6.8h, v11.8h, v14.8h
    ldr q8, [x4, #192]
    ldr q9, [x5, #192]
    ldr q10, [x6, #192]
    ldr q11, [x7, #192]
    fmla v3.8h, v8.8h, v12.8h
    fmla v3.8h, v9.8h, v13.8h
    fmla v3.8h, v10.8h, v14.8h
    fmla v7.8h, v9.8h, v12.8h
    fmla v7.8h, v10.8h, v13.8h
    fmla v7.8h, v11.8h, v14.8h
    add x2, x2, 16
    add x4, x4, 16
    add x5, x5, 16
    add x6, x6, 16
    add x7, x7, 16
    cmp x2, 192
    b.lt conv15_neon_j_loop
    add x8, x0, #1
    cmp x8, x26
    b.ge conv15_neon_Lrow0_spill
    faddp v4.8h, v4.8h, v4.8h
    faddp v4.4h, v4.4h, v4.4h
    faddp h4, v4.2h
    fmax h4, h4, h15
    str h4, [sp, #224]
    faddp v5.8h, v5.8h, v5.8h
    faddp v5.4h, v5.4h, v5.4h
    faddp h5, v5.2h
    fmax h5, h5, h15
    str h5, [sp, #228]
    faddp v6.8h, v6.8h, v6.8h
    faddp v6.4h, v6.4h, v6.4h
    faddp h6, v6.2h
    fmax h6, h6, h15
    str h6, [sp, #232]
    faddp v7.8h, v7.8h, v7.8h
    faddp v7.4h, v7.4h, v7.4h
    faddp h7, v7.2h
    fmax h7, h7, h15
    str h7, [sp, #236]
conv15_neon_Lrow0_spill:
    faddp v0.8h, v0.8h, v0.8h
    faddp v0.4h, v0.4h, v0.4h
    faddp h0, v0.2h
    fmax h0, h0, h15
    str h0, [sp, #208]
    faddp v1.8h, v1.8h, v1.8h
    faddp v1.4h, v1.4h, v1.4h
    faddp h1, v1.2h
    fmax h1, h1, h15
    str h1, [sp, #212]
    faddp v2.8h, v2.8h, v2.8h
    faddp v2.4h, v2.4h, v2.4h
    faddp h2, v2.2h
    fmax h2, h2, h15
    str h2, [sp, #216]
    faddp v3.8h, v3.8h, v3.8h
    faddp v3.4h, v3.4h, v3.4h
    faddp h3, v3.2h
    fmax h3, h3, h15
    str h3, [sp, #220]
    str x0, [sp, #240]
    str x1, [sp, #248]
    str x12, [sp, #256]
    str x16, [sp, #264]
    str x15, [sp, #272]
    add x8, x0, #1
    cmp x8, x26
    b.ge conv15_neon_Lrow0_store
    add x28, x28, x20
    ldr x8, [sp, #248]
    add x8, x8, #0
    cmp x8, x25
    b.ge conv15_neon_Lc1_0
    ldr h0, [sp, #224]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc1_0_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc1_0_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc1_0_dn
conv15_neon_Lc1_0_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc1_0_dn
conv15_neon_Lc1_0_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc1_0_dn:
    ldr x12, [sp, #256]
    add x14, x28, #0
    str s0, [x14, x12]
conv15_neon_Lc1_0:
    ldr x8, [sp, #248]
    add x8, x8, #1
    cmp x8, x25
    b.ge conv15_neon_Lc1_1
    ldr h0, [sp, #228]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc1_1_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc1_1_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc1_1_dn
conv15_neon_Lc1_1_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc1_1_dn
conv15_neon_Lc1_1_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc1_1_dn:
    ldr x12, [sp, #256]
    add x14, x28, #16
    str s0, [x14, x12]
conv15_neon_Lc1_1:
    ldr x8, [sp, #248]
    add x8, x8, #2
    cmp x8, x25
    b.ge conv15_neon_Lc1_2
    ldr h0, [sp, #232]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc1_2_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc1_2_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc1_2_dn
conv15_neon_Lc1_2_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc1_2_dn
conv15_neon_Lc1_2_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc1_2_dn:
    ldr x12, [sp, #256]
    add x14, x28, #32
    str s0, [x14, x12]
conv15_neon_Lc1_2:
    ldr x8, [sp, #248]
    add x8, x8, #3
    cmp x8, x25
    b.ge conv15_neon_Lc1_3
    ldr h0, [sp, #236]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc1_3_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc1_3_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc1_3_dn
conv15_neon_Lc1_3_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc1_3_dn
conv15_neon_Lc1_3_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc1_3_dn:
    ldr x12, [sp, #256]
    add x14, x28, #48
    str s0, [x14, x12]
conv15_neon_Lc1_3:
    sub x28, x28, x20
conv15_neon_Lrow0_store:
    ldr x8, [sp, #248]
    add x8, x8, #0
    cmp x8, x25
    b.ge conv15_neon_Lc0_0
    ldr h0, [sp, #208]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc0_0_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc0_0_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc0_0_dn
conv15_neon_Lc0_0_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc0_0_dn
conv15_neon_Lc0_0_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc0_0_dn:
    ldr x12, [sp, #256]
    add x14, x28, #0
    str s0, [x14, x12]
conv15_neon_Lc0_0:
    ldr x8, [sp, #248]
    add x8, x8, #1
    cmp x8, x25
    b.ge conv15_neon_Lc0_1
    ldr h0, [sp, #212]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc0_1_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc0_1_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc0_1_dn
conv15_neon_Lc0_1_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc0_1_dn
conv15_neon_Lc0_1_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc0_1_dn:
    ldr x12, [sp, #256]
    add x14, x28, #16
    str s0, [x14, x12]
conv15_neon_Lc0_1:
    ldr x8, [sp, #248]
    add x8, x8, #2
    cmp x8, x25
    b.ge conv15_neon_Lc0_2
    ldr h0, [sp, #216]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc0_2_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc0_2_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc0_2_dn
conv15_neon_Lc0_2_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc0_2_dn
conv15_neon_Lc0_2_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc0_2_dn:
    ldr x12, [sp, #256]
    add x14, x28, #32
    str s0, [x14, x12]
conv15_neon_Lc0_2:
    ldr x8, [sp, #248]
    add x8, x8, #3
    cmp x8, x25
    b.ge conv15_neon_Lc0_3
    ldr h0, [sp, #220]
    fcvt s0, h0
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #0]
    fmul s0, s0, s1
    ldr s1, [x14, #4]
    fcmp s0, s1
    b.le conv15_neon_Lc0_3_lo
    ldr s1, [x14, #8]
    fcmp s0, s1
    b.le conv15_neon_Lc0_3_mi
    ldr s1, [x14, #12]
    fsub s0, s0, s1
    ldr s1, [x14, #16]
    fdiv s0, s0, s1
    bl expf
    adr x14, conv15_neon_conv15_hdr_c
    ldr s1, [x14, #20]
    fsub s0, s0, s1
    b conv15_neon_Lc0_3_dn
conv15_neon_Lc0_3_mi:
    ldr s1, [x14, #24]
    fsub s0, s0, s1
    ldr s1, [x14, #28]
    fdiv s0, s0, s1
    ldr s1, [x14, #32]
    bl powf
    b conv15_neon_Lc0_3_dn
conv15_neon_Lc0_3_lo:
    ldr s1, [x14, #36]
    fdiv s0, s0, s1
conv15_neon_Lc0_3_dn:
    ldr x12, [sp, #256]
    add x14, x28, #48
    str s0, [x14, x12]
conv15_neon_Lc0_3:
    ldr x0, [sp, #240]
    ldr x1, [sp, #248]
    ldr x12, [sp, #256]
    ldr x16, [sp, #264]
    ldr x15, [sp, #272]
    fmov s15, wzr
    mov x8, 576
    add x16, x16, x8
    add x17, x17, x8
    add x13, x13, x8
    add x15, x15, #2
    add x12, x12, #4
    b conv15_neon_ii_loop
conv15_neon_Lii_done:
    add x1, x1, 4
    b conv15_neon_x_loop
conv15_neon_Lx_done:
    add x0, x0, #2
    b conv15_neon_y_loop
conv15_neon_Ly_done:
    ldp q8, q9, [sp, #0]
    ldp q10, q11, [sp, #32]
    ldp q12, q13, [sp, #64]
    ldp q14, q15, [sp, #96]
    ldp x19, x20, [sp, #128]
    ldp x21, x22, [sp, #144]
    ldp x23, x24, [sp, #160]
    ldp x25, x26, [sp, #176]
    ldp x27, x28, [sp, #192]
    ldp x29, x30, [sp, #280]
    add sp, sp, 304
    .cfi_def_cfa_offset 0
    ret 
.cfi_endproc
.size ray_conv15_neon_f16, .-ray_conv15_neon_f16
.p2align 2
conv15_neon_conv15_hdr_c:
    .long 1078502858
    .long 991051429
    .long 1052635239
    .long 1065330066
    .long 1044727525
    .long 1003299552
    .long 3141583984
    .long 1070771684
    .long 1075076095
    .long 1152424654
.section .note.GNU-stack,"",@progbits
