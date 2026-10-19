.intel_syntax noprefix
.section __TEXT,__text,regular,pure_instructions
.globl _ray_conv9_avx512_f16
_ray_conv9_avx512_f16:
    push rbx
    push rsi
    push rdi
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 272
    movups [rsp], xmm6
    movups [rsp + 16], xmm7
    movups [rsp + 32], xmm8
    movups [rsp + 48], xmm9
    movups [rsp + 64], xmm10
    movups [rsp + 80], xmm11
    movups [rsp + 96], xmm12
    movups [rsp + 112], xmm13
    movups [rsp + 128], xmm14
    movups [rsp + 144], xmm15
    mov [rsp + 208], rsi
    mov eax, r8d
    add eax, edx
    mov [rsp + 184], rax
    mov eax, edx
    mov [rsp + 200], rax
    mov eax, ecx
    add eax, r9d
    mov [rsp + 192], rax
    mov r9d, [rsp + 344]
    mov rax, r9
    imul rax, rax, 224
    mov [rsp + 216], rax
    mov r8d, [rsp + 352]
    mov rax, r8
    imul rax, rax, 96
    mov [rsp + 224], rax
    mov rsi, [rsp + 360]
    mov [rsp + 160], rsi
    mov rbp, [rsp + 368]
    mov [rsp + 168], rbp
    mov rax, [rsp + 376]
    mov [rsp + 176], rax
    mov eax, [rsp + 384]
    imul r12, rax, 192
    mov [rsp + 264], r12
    mov eax, ecx
    vpxor xmm15, xmm15, xmm15
conv9_avx512f16_y_loop:
    cmp rax, [rsp + 192]
    jge conv9_avx512f16_y_done
    mov r15, [rsp + 216]
    mov r9, rax
    shr r9, 1
    imul r9, r15
    mov [rsp + 240], r9
    mov r9, rax
    add r9, 1
    shr r9, 1
    imul r9, r15
    mov [rsp + 248], r9
    mov r9, rax
    add r9, 2
    shr r9, 1
    imul r9, r15
    mov [rsp + 256], r9
    cmp rax, 1
    jb .Lconv9_avx512f16_r0neg
    mov r9, rax
    sub r9, 1
    shr r9, 1
    imul r9, r15
    jmp .Lconv9_avx512f16_r0
.Lconv9_avx512f16_r0neg:
    mov r9, r15
    neg r9
.Lconv9_avx512f16_r0:
    mov [rsp + 232], r9
    mov rcx, [rsp + 200]
conv9_avx512f16_x_loop:
    cmp rcx, [rsp + 184]
    jge conv9_avx512f16_x_done
    mov r12, [rsp + 264]
    mov rbx, rax
    imul rbx, r12
    add rbx, [rsp + 176]
    imul rdx, rcx, 192
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 160]
    mov rbp, [rsp + 168]
conv9_avx512f16_ii_loop:
    cmp r8, 192
    jge conv9_avx512f16_ii_done
    movzx edx, WORD PTR [rbp]
    vmovd xmm0, edx
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vmovaps ymm3, ymm0
    vmovaps ymm4, ymm0
    vmovaps ymm5, ymm0
    vmovaps ymm6, ymm0
    vmovaps ymm7, ymm0
    vmovaps ymm4, ymm0
    vmovaps ymm5, ymm1
    vmovaps ymm6, ymm2
    vmovaps ymm7, ymm3
    vmovaps ymm8, ymm4
    vmovaps ymm9, ymm5
    vmovaps ymm10, ymm6
    vmovaps ymm11, ymm7
    xor edx, edx
    mov r9d, ecx
    add r9d, 1
    mov r10, r9
    shr r10, 1
    sub r10, 1
    imul r10, r10, 224
    add r10, rdi
    mov r11, r9
    add r11, 1
    shr r11, 1
    sub r11, 1
    imul r11, r11, 224
    add r11, rdi
    mov r13, r9
    add r13, 2
    shr r13, 1
    sub r13, 1
    imul r13, r13, 224
    add r13, rdi
    mov r12, r9
    add r12, 3
    shr r12, 1
    sub r12, 1
    imul r12, r12, 224
    add r12, rdi
conv9_avx512f16_j1w0:
    vmovaps ymm12, [rsi + rdx]
    vmovaps ymm13, [rsi + rdx + 672]
    vmovaps ymm14, [rsi + rdx + 1344]
    mov r9, [rsp + 232]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm12
    vfmadd231ph ymm2, ymm10, ymm12
    vfmadd231ph ymm3, ymm11, ymm12
    mov r9, [rsp + 240]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm13
    vfmadd231ph ymm4, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm13
    vfmadd231ph ymm5, ymm9, ymm12
    vfmadd231ph ymm2, ymm10, ymm13
    vfmadd231ph ymm6, ymm10, ymm12
    vfmadd231ph ymm3, ymm11, ymm13
    vfmadd231ph ymm7, ymm11, ymm12
    mov r9, [rsp + 248]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm14
    vfmadd231ph ymm4, ymm8, ymm13
    vfmadd231ph ymm1, ymm9, ymm14
    vfmadd231ph ymm5, ymm9, ymm13
    vfmadd231ph ymm2, ymm10, ymm14
    vfmadd231ph ymm6, ymm10, ymm13
    vfmadd231ph ymm3, ymm11, ymm14
    vfmadd231ph ymm7, ymm11, ymm13
    mov r9, [rsp + 256]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm4, ymm8, ymm14
    vfmadd231ph ymm5, ymm9, ymm14
    vfmadd231ph ymm6, ymm10, ymm14
    vfmadd231ph ymm7, ymm11, ymm14
    add rdx, 32
    add r10, 32
    add r11, 32
    add r13, 32
    add r12, 32
    cmp rdx, 224
    jl conv9_avx512f16_j1w0
    mov r9d, ecx
    add r9d, 2
    mov r10, r9
    shr r10, 1
    sub r10, 1
    imul r10, r10, 224
    add r10, rdi
    mov r11, r9
    add r11, 1
    shr r11, 1
    sub r11, 1
    imul r11, r11, 224
    add r11, rdi
    mov r13, r9
    add r13, 2
    shr r13, 1
    sub r13, 1
    imul r13, r13, 224
    add r13, rdi
    mov r12, r9
    add r12, 3
    shr r12, 1
    sub r12, 1
    imul r12, r12, 224
    add r12, rdi
conv9_avx512f16_j1w1:
    vmovaps ymm12, [rsi + rdx]
    vmovaps ymm13, [rsi + rdx + 672]
    vmovaps ymm14, [rsi + rdx + 1344]
    mov r9, [rsp + 232]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm12
    vfmadd231ph ymm2, ymm10, ymm12
    vfmadd231ph ymm3, ymm11, ymm12
    mov r9, [rsp + 240]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm13
    vfmadd231ph ymm4, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm13
    vfmadd231ph ymm5, ymm9, ymm12
    vfmadd231ph ymm2, ymm10, ymm13
    vfmadd231ph ymm6, ymm10, ymm12
    vfmadd231ph ymm3, ymm11, ymm13
    vfmadd231ph ymm7, ymm11, ymm12
    mov r9, [rsp + 248]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm14
    vfmadd231ph ymm4, ymm8, ymm13
    vfmadd231ph ymm1, ymm9, ymm14
    vfmadd231ph ymm5, ymm9, ymm13
    vfmadd231ph ymm2, ymm10, ymm14
    vfmadd231ph ymm6, ymm10, ymm13
    vfmadd231ph ymm3, ymm11, ymm14
    vfmadd231ph ymm7, ymm11, ymm13
    mov r9, [rsp + 256]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm4, ymm8, ymm14
    vfmadd231ph ymm5, ymm9, ymm14
    vfmadd231ph ymm6, ymm10, ymm14
    vfmadd231ph ymm7, ymm11, ymm14
    add rdx, 32
    add r10, 32
    add r11, 32
    add r13, 32
    add r12, 32
    cmp rdx, 448
    jl conv9_avx512f16_j1w1
    mov r9d, ecx
    add r9d, 3
    mov r10, r9
    shr r10, 1
    sub r10, 1
    imul r10, r10, 224
    add r10, rdi
    mov r11, r9
    add r11, 1
    shr r11, 1
    sub r11, 1
    imul r11, r11, 224
    add r11, rdi
    mov r13, r9
    add r13, 2
    shr r13, 1
    sub r13, 1
    imul r13, r13, 224
    add r13, rdi
    mov r12, r9
    add r12, 3
    shr r12, 1
    sub r12, 1
    imul r12, r12, 224
    add r12, rdi
conv9_avx512f16_j1w2:
    vmovaps ymm12, [rsi + rdx]
    vmovaps ymm13, [rsi + rdx + 672]
    vmovaps ymm14, [rsi + rdx + 1344]
    mov r9, [rsp + 232]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm12
    vfmadd231ph ymm2, ymm10, ymm12
    vfmadd231ph ymm3, ymm11, ymm12
    mov r9, [rsp + 240]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm13
    vfmadd231ph ymm4, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm13
    vfmadd231ph ymm5, ymm9, ymm12
    vfmadd231ph ymm2, ymm10, ymm13
    vfmadd231ph ymm6, ymm10, ymm12
    vfmadd231ph ymm3, ymm11, ymm13
    vfmadd231ph ymm7, ymm11, ymm12
    mov r9, [rsp + 248]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm0, ymm8, ymm14
    vfmadd231ph ymm4, ymm8, ymm13
    vfmadd231ph ymm1, ymm9, ymm14
    vfmadd231ph ymm5, ymm9, ymm13
    vfmadd231ph ymm2, ymm10, ymm14
    vfmadd231ph ymm6, ymm10, ymm13
    vfmadd231ph ymm3, ymm11, ymm14
    vfmadd231ph ymm7, ymm11, ymm13
    mov r9, [rsp + 256]
    vmovups ymm8, [r10 + r9]
    vmovups ymm9, [r11 + r9]
    vmovups ymm10, [r13 + r9]
    vmovups ymm11, [r12 + r9]
    vfmadd231ph ymm4, ymm8, ymm14
    vfmadd231ph ymm5, ymm9, ymm14
    vfmadd231ph ymm6, ymm10, ymm14
    vfmadd231ph ymm7, ymm11, ymm14
    add rdx, 32
    add r10, 32
    add r11, 32
    add r13, 32
    add r12, 32
    cmp rdx, 672
    jl conv9_avx512f16_j1w2
    mov r11, [rsp + 208]
    mov rdx, rax
    imul rdx, [rsp + 224]
    mov r9, rcx
    sub r9, 1
    imul r9, r9, 96
    add rdx, r9
    add r11, rdx
    mov r13, r11
    sub r13, [rsp + 224]
    mov r14, r11
    add r14, [rsp + 224]
    mov r10, r14
    add r10, [rsp + 224]
    xor edx, edx
conv9_avx512f16_j2_loop:
    vmovaps ymm12, [rsi + rdx + 2016]
    vmovaps ymm13, [rsi + rdx + 2304]
    vmovaps ymm14, [rsi + rdx + 2592]
    vmovups ymm8, [r13 + rdx]
    vmovups ymm9, [r11 + rdx]
    vmovups ymm10, [r14 + rdx]
    vmovups ymm11, [r10 + rdx]
    vfmadd231ph ymm0, ymm8, ymm12
    vfmadd231ph ymm0, ymm9, ymm13
    vfmadd231ph ymm0, ymm10, ymm14
    vfmadd231ph ymm4, ymm9, ymm12
    vfmadd231ph ymm4, ymm10, ymm13
    vfmadd231ph ymm4, ymm11, ymm14
    vmovups ymm8, [r13 + rdx + 96]
    vmovups ymm9, [r11 + rdx + 96]
    vmovups ymm10, [r14 + rdx + 96]
    vmovups ymm11, [r10 + rdx + 96]
    vfmadd231ph ymm1, ymm8, ymm12
    vfmadd231ph ymm1, ymm9, ymm13
    vfmadd231ph ymm1, ymm10, ymm14
    vfmadd231ph ymm5, ymm9, ymm12
    vfmadd231ph ymm5, ymm10, ymm13
    vfmadd231ph ymm5, ymm11, ymm14
    vmovups ymm8, [r13 + rdx + 192]
    vmovups ymm9, [r11 + rdx + 192]
    vmovups ymm10, [r14 + rdx + 192]
    vmovups ymm11, [r10 + rdx + 192]
    vfmadd231ph ymm2, ymm8, ymm12
    vfmadd231ph ymm2, ymm9, ymm13
    vfmadd231ph ymm2, ymm10, ymm14
    vfmadd231ph ymm6, ymm9, ymm12
    vfmadd231ph ymm6, ymm10, ymm13
    vfmadd231ph ymm6, ymm11, ymm14
    vmovups ymm8, [r13 + rdx + 288]
    vmovups ymm9, [r11 + rdx + 288]
    vmovups ymm10, [r14 + rdx + 288]
    vmovups ymm11, [r10 + rdx + 288]
    vfmadd231ph ymm3, ymm8, ymm12
    vfmadd231ph ymm3, ymm9, ymm13
    vfmadd231ph ymm3, ymm10, ymm14
    vfmadd231ph ymm7, ymm9, ymm12
    vfmadd231ph ymm7, ymm10, ymm13
    vfmadd231ph ymm7, ymm11, ymm14
    add rdx, 32
    cmp rdx, 288
    jl conv9_avx512f16_j2_loop
    mov r12, [rsp + 264]
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 192]
    jge conv9_avx512f16_row0_only
    add rbx, r12
    lea rdx, [rcx]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c1_0
    vextracti128 xmm8, ymm4, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm4, xmm4
    vaddps ymm4, ymm4, ymm8
    vhaddps ymm4, ymm4, ymm4
    vhaddps ymm4, ymm4, ymm4
    vperm2f128 ymm8, ymm4, ymm4, 1
    vaddps ymm4, ymm8, ymm4
    maxss xmm4, xmm15
    vcvtps2ph xmm8, xmm4, 0
    pextrw WORD PTR [rbx + r8], xmm8, 0
.Lconv9_avx512f16_c1_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c1_1
    vextracti128 xmm8, ymm5, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm5, xmm5
    vaddps ymm5, ymm5, ymm8
    vhaddps ymm5, ymm5, ymm5
    vhaddps ymm5, ymm5, ymm5
    vperm2f128 ymm8, ymm5, ymm5, 1
    vaddps ymm5, ymm8, ymm5
    maxss xmm5, xmm15
    vcvtps2ph xmm8, xmm5, 0
    pextrw WORD PTR [rbx + r8 + 192], xmm8, 0
.Lconv9_avx512f16_c1_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c1_2
    vextracti128 xmm8, ymm6, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm6, xmm6
    vaddps ymm6, ymm6, ymm8
    vhaddps ymm6, ymm6, ymm6
    vhaddps ymm6, ymm6, ymm6
    vperm2f128 ymm8, ymm6, ymm6, 1
    vaddps ymm6, ymm8, ymm6
    maxss xmm6, xmm15
    vcvtps2ph xmm8, xmm6, 0
    pextrw WORD PTR [rbx + r8 + 384], xmm8, 0
.Lconv9_avx512f16_c1_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c1_3
    vextracti128 xmm8, ymm7, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm7, xmm7
    vaddps ymm7, ymm7, ymm8
    vhaddps ymm7, ymm7, ymm7
    vhaddps ymm7, ymm7, ymm7
    vperm2f128 ymm8, ymm7, ymm7, 1
    vaddps ymm7, ymm8, ymm7
    maxss xmm7, xmm15
    vcvtps2ph xmm8, xmm7, 0
    pextrw WORD PTR [rbx + r8 + 576], xmm8, 0
.Lconv9_avx512f16_c1_3:
    sub rbx, r12
conv9_avx512f16_row0_only:
    lea rdx, [rcx]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c0_0
    vextracti128 xmm8, ymm0, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm0, xmm0
    vaddps ymm0, ymm0, ymm8
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vperm2f128 ymm8, ymm0, ymm0, 1
    vaddps ymm0, ymm8, ymm0
    maxss xmm0, xmm15
    vcvtps2ph xmm8, xmm0, 0
    pextrw WORD PTR [rbx + r8], xmm8, 0
.Lconv9_avx512f16_c0_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c0_1
    vextracti128 xmm8, ymm1, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm1, xmm1
    vaddps ymm1, ymm1, ymm8
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vperm2f128 ymm8, ymm1, ymm1, 1
    vaddps ymm1, ymm8, ymm1
    maxss xmm1, xmm15
    vcvtps2ph xmm8, xmm1, 0
    pextrw WORD PTR [rbx + r8 + 192], xmm8, 0
.Lconv9_avx512f16_c0_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c0_2
    vextracti128 xmm8, ymm2, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm2, xmm2
    vaddps ymm2, ymm2, ymm8
    vhaddps ymm2, ymm2, ymm2
    vhaddps ymm2, ymm2, ymm2
    vperm2f128 ymm8, ymm2, ymm2, 1
    vaddps ymm2, ymm8, ymm2
    maxss xmm2, xmm15
    vcvtps2ph xmm8, xmm2, 0
    pextrw WORD PTR [rbx + r8 + 384], xmm8, 0
.Lconv9_avx512f16_c0_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 184]
    jge .Lconv9_avx512f16_c0_3
    vextracti128 xmm8, ymm3, 1
    vcvtph2ps ymm8, xmm8
    vcvtph2ps ymm3, xmm3
    vaddps ymm3, ymm3, ymm8
    vhaddps ymm3, ymm3, ymm3
    vhaddps ymm3, ymm3, ymm3
    vperm2f128 ymm8, ymm3, ymm3, 1
    vaddps ymm3, ymm8, ymm3
    maxss xmm3, xmm15
    vcvtps2ph xmm8, xmm3, 0
    pextrw WORD PTR [rbx + r8 + 576], xmm8, 0
.Lconv9_avx512f16_c0_3:
    add rsi, 2880
    add rbp, 2
    add r8, 2
    jmp conv9_avx512f16_ii_loop
conv9_avx512f16_ii_done:
    add rcx, 4
    jmp conv9_avx512f16_x_loop
conv9_avx512f16_x_done:
    add rax, 2
    jmp conv9_avx512f16_y_loop
conv9_avx512f16_y_done:
    movups xmm6, [rsp]
    movups xmm7, [rsp + 16]
    movups xmm8, [rsp + 32]
    movups xmm9, [rsp + 48]
    movups xmm10, [rsp + 64]
    movups xmm11, [rsp + 80]
    movups xmm12, [rsp + 96]
    movups xmm13, [rsp + 112]
    movups xmm14, [rsp + 128]
    movups xmm15, [rsp + 144]
    add rsp, 272
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rdi
    pop rsi
    pop rbx
    ret 
