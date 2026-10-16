.intel_syntax noprefix
.text
.globl ray_conv7_sse41
.type ray_conv7_sse41, @function
ray_conv7_sse41:
.cfi_startproc
    push rbx
    .cfi_def_cfa_offset 8
    .cfi_offset rbx, -8
    push rsi
    .cfi_def_cfa_offset 16
    .cfi_offset rsi, -16
    push rdi
    .cfi_def_cfa_offset 24
    .cfi_offset rdi, -24
    push rbp
    .cfi_def_cfa_offset 32
    .cfi_offset rbp, -32
    push r12
    .cfi_def_cfa_offset 40
    .cfi_offset r12, -40
    push r13
    .cfi_def_cfa_offset 48
    .cfi_offset r13, -48
    push r14
    .cfi_def_cfa_offset 56
    .cfi_offset r14, -56
    push r15
    .cfi_def_cfa_offset 64
    .cfi_offset r15, -64
    sub rsp, 272
    .cfi_def_cfa_offset 336
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
    imul rax, rax, 384
    mov [rsp + 216], rax
    mov r8d, [rsp + 352]
    mov rax, r8
    imul rax, rax, 256
    mov [rsp + 224], rax
    mov rsi, [rsp + 360]
    mov [rsp + 160], rsi
    mov rbp, [rsp + 368]
    mov [rsp + 168], rbp
    mov rax, [rsp + 376]
    mov [rsp + 176], rax
    mov eax, [rsp + 384]
    imul r12, rax, 448
    mov [rsp + 264], r12
    mov eax, ecx
    xorps xmm15, xmm15
conv7_sse41_y_loop:
    cmp rax, [rsp + 192]
    jge conv7_sse41_y_done
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
    jb .Lconv7_sse41_r0neg
    mov r9, rax
    sub r9, 1
    shr r9, 1
    imul r9, r15
    jmp .Lconv7_sse41_r0
.Lconv7_sse41_r0neg:
    mov r9, r15
    neg r9
.Lconv7_sse41_r0:
    mov [rsp + 232], r9
    mov rcx, [rsp + 200]
conv7_sse41_x_loop:
    cmp rcx, [rsp + 184]
    jge conv7_sse41_x_done
    mov r12, [rsp + 264]
    mov rbx, rax
    imul rbx, r12
    add rbx, [rsp + 176]
    imul rdx, rcx, 448
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 160]
    mov rbp, [rsp + 168]
conv7_sse41_ii_loop:
    cmp r8, 448
    jge conv7_sse41_ii_done
    movss xmm0, DWORD PTR [rbp]
    movaps xmm1, xmm0
    movaps xmm2, xmm0
    movaps xmm3, xmm0
    movaps xmm4, xmm0
    movaps xmm5, xmm0
    movaps xmm6, xmm0
    movaps xmm7, xmm0
    movaps xmm4, xmm0
    movaps xmm5, xmm1
    movaps xmm6, xmm2
    movaps xmm7, xmm3
    movaps xmm8, xmm4
    movaps xmm9, xmm5
    movaps xmm10, xmm6
    movaps xmm11, xmm7
    xor edx, edx
    mov r9d, ecx
    add r9d, 1
    mov r10, r9
    shr r10, 1
    sub r10, 1
    imul r10, r10, 384
    add r10, rdi
    mov r11, r9
    add r11, 1
    shr r11, 1
    sub r11, 1
    imul r11, r11, 384
    add r11, rdi
    mov r13, r9
    add r13, 2
    shr r13, 1
    sub r13, 1
    imul r13, r13, 384
    add r13, rdi
    mov r12, r9
    add r12, 3
    shr r12, 1
    sub r12, 1
    imul r12, r12, 384
    add r12, rdi
conv7_sse41_j1w0:
    movaps xmm12, [rsi + rdx]
    movaps xmm13, [rsi + rdx + 1152]
    movaps xmm14, [rsi + rdx + 2304]
    mov r9, [rsp + 232]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    mulps xmm8, xmm12
    addps xmm0, xmm8
    mulps xmm9, xmm12
    addps xmm1, xmm9
    mulps xmm10, xmm12
    addps xmm2, xmm10
    mulps xmm11, xmm12
    addps xmm3, xmm11
    mov r9, [rsp + 240]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    movaps xmm15, xmm8
    mulps xmm15, xmm13
    addps xmm0, xmm15
    mulps xmm8, xmm12
    addps xmm4, xmm8
    movaps xmm15, xmm9
    mulps xmm15, xmm13
    addps xmm1, xmm15
    mulps xmm9, xmm12
    addps xmm5, xmm9
    movaps xmm15, xmm10
    mulps xmm15, xmm13
    addps xmm2, xmm15
    mulps xmm10, xmm12
    addps xmm6, xmm10
    movaps xmm15, xmm11
    mulps xmm15, xmm13
    addps xmm3, xmm15
    mulps xmm11, xmm12
    addps xmm7, xmm11
    mov r9, [rsp + 248]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    movaps xmm15, xmm8
    mulps xmm15, xmm14
    addps xmm0, xmm15
    mulps xmm8, xmm13
    addps xmm4, xmm8
    movaps xmm15, xmm9
    mulps xmm15, xmm14
    addps xmm1, xmm15
    mulps xmm9, xmm13
    addps xmm5, xmm9
    movaps xmm15, xmm10
    mulps xmm15, xmm14
    addps xmm2, xmm15
    mulps xmm10, xmm13
    addps xmm6, xmm10
    movaps xmm15, xmm11
    mulps xmm15, xmm14
    addps xmm3, xmm15
    mulps xmm11, xmm13
    addps xmm7, xmm11
    mov r9, [rsp + 256]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    mulps xmm8, xmm14
    addps xmm4, xmm8
    mulps xmm9, xmm14
    addps xmm5, xmm9
    mulps xmm10, xmm14
    addps xmm6, xmm10
    mulps xmm11, xmm14
    addps xmm7, xmm11
    add rdx, 16
    add r10, 16
    add r11, 16
    add r13, 16
    add r12, 16
    cmp rdx, 384
    jl conv7_sse41_j1w0
    mov r9d, ecx
    add r9d, 2
    mov r10, r9
    shr r10, 1
    sub r10, 1
    imul r10, r10, 384
    add r10, rdi
    mov r11, r9
    add r11, 1
    shr r11, 1
    sub r11, 1
    imul r11, r11, 384
    add r11, rdi
    mov r13, r9
    add r13, 2
    shr r13, 1
    sub r13, 1
    imul r13, r13, 384
    add r13, rdi
    mov r12, r9
    add r12, 3
    shr r12, 1
    sub r12, 1
    imul r12, r12, 384
    add r12, rdi
conv7_sse41_j1w1:
    movaps xmm12, [rsi + rdx]
    movaps xmm13, [rsi + rdx + 1152]
    movaps xmm14, [rsi + rdx + 2304]
    mov r9, [rsp + 232]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    mulps xmm8, xmm12
    addps xmm0, xmm8
    mulps xmm9, xmm12
    addps xmm1, xmm9
    mulps xmm10, xmm12
    addps xmm2, xmm10
    mulps xmm11, xmm12
    addps xmm3, xmm11
    mov r9, [rsp + 240]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    movaps xmm15, xmm8
    mulps xmm15, xmm13
    addps xmm0, xmm15
    mulps xmm8, xmm12
    addps xmm4, xmm8
    movaps xmm15, xmm9
    mulps xmm15, xmm13
    addps xmm1, xmm15
    mulps xmm9, xmm12
    addps xmm5, xmm9
    movaps xmm15, xmm10
    mulps xmm15, xmm13
    addps xmm2, xmm15
    mulps xmm10, xmm12
    addps xmm6, xmm10
    movaps xmm15, xmm11
    mulps xmm15, xmm13
    addps xmm3, xmm15
    mulps xmm11, xmm12
    addps xmm7, xmm11
    mov r9, [rsp + 248]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    movaps xmm15, xmm8
    mulps xmm15, xmm14
    addps xmm0, xmm15
    mulps xmm8, xmm13
    addps xmm4, xmm8
    movaps xmm15, xmm9
    mulps xmm15, xmm14
    addps xmm1, xmm15
    mulps xmm9, xmm13
    addps xmm5, xmm9
    movaps xmm15, xmm10
    mulps xmm15, xmm14
    addps xmm2, xmm15
    mulps xmm10, xmm13
    addps xmm6, xmm10
    movaps xmm15, xmm11
    mulps xmm15, xmm14
    addps xmm3, xmm15
    mulps xmm11, xmm13
    addps xmm7, xmm11
    mov r9, [rsp + 256]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    mulps xmm8, xmm14
    addps xmm4, xmm8
    mulps xmm9, xmm14
    addps xmm5, xmm9
    mulps xmm10, xmm14
    addps xmm6, xmm10
    mulps xmm11, xmm14
    addps xmm7, xmm11
    add rdx, 16
    add r10, 16
    add r11, 16
    add r13, 16
    add r12, 16
    cmp rdx, 768
    jl conv7_sse41_j1w1
    mov r9d, ecx
    add r9d, 3
    mov r10, r9
    shr r10, 1
    sub r10, 1
    imul r10, r10, 384
    add r10, rdi
    mov r11, r9
    add r11, 1
    shr r11, 1
    sub r11, 1
    imul r11, r11, 384
    add r11, rdi
    mov r13, r9
    add r13, 2
    shr r13, 1
    sub r13, 1
    imul r13, r13, 384
    add r13, rdi
    mov r12, r9
    add r12, 3
    shr r12, 1
    sub r12, 1
    imul r12, r12, 384
    add r12, rdi
conv7_sse41_j1w2:
    movaps xmm12, [rsi + rdx]
    movaps xmm13, [rsi + rdx + 1152]
    movaps xmm14, [rsi + rdx + 2304]
    mov r9, [rsp + 232]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    mulps xmm8, xmm12
    addps xmm0, xmm8
    mulps xmm9, xmm12
    addps xmm1, xmm9
    mulps xmm10, xmm12
    addps xmm2, xmm10
    mulps xmm11, xmm12
    addps xmm3, xmm11
    mov r9, [rsp + 240]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    movaps xmm15, xmm8
    mulps xmm15, xmm13
    addps xmm0, xmm15
    mulps xmm8, xmm12
    addps xmm4, xmm8
    movaps xmm15, xmm9
    mulps xmm15, xmm13
    addps xmm1, xmm15
    mulps xmm9, xmm12
    addps xmm5, xmm9
    movaps xmm15, xmm10
    mulps xmm15, xmm13
    addps xmm2, xmm15
    mulps xmm10, xmm12
    addps xmm6, xmm10
    movaps xmm15, xmm11
    mulps xmm15, xmm13
    addps xmm3, xmm15
    mulps xmm11, xmm12
    addps xmm7, xmm11
    mov r9, [rsp + 248]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    movaps xmm15, xmm8
    mulps xmm15, xmm14
    addps xmm0, xmm15
    mulps xmm8, xmm13
    addps xmm4, xmm8
    movaps xmm15, xmm9
    mulps xmm15, xmm14
    addps xmm1, xmm15
    mulps xmm9, xmm13
    addps xmm5, xmm9
    movaps xmm15, xmm10
    mulps xmm15, xmm14
    addps xmm2, xmm15
    mulps xmm10, xmm13
    addps xmm6, xmm10
    movaps xmm15, xmm11
    mulps xmm15, xmm14
    addps xmm3, xmm15
    mulps xmm11, xmm13
    addps xmm7, xmm11
    mov r9, [rsp + 256]
    movups xmm8, [r10 + r9]
    movups xmm9, [r11 + r9]
    movups xmm10, [r13 + r9]
    movups xmm11, [r12 + r9]
    mulps xmm8, xmm14
    addps xmm4, xmm8
    mulps xmm9, xmm14
    addps xmm5, xmm9
    mulps xmm10, xmm14
    addps xmm6, xmm10
    mulps xmm11, xmm14
    addps xmm7, xmm11
    add rdx, 16
    add r10, 16
    add r11, 16
    add r13, 16
    add r12, 16
    cmp rdx, 1152
    jl conv7_sse41_j1w2
    mov r11, [rsp + 208]
    mov rdx, rax
    imul rdx, [rsp + 224]
    mov r9, rcx
    sub r9, 1
    imul r9, r9, 256
    add rdx, r9
    add r11, rdx
    mov r13, r11
    sub r13, [rsp + 224]
    mov r14, r11
    add r14, [rsp + 224]
    mov r10, r14
    add r10, [rsp + 224]
    xor edx, edx
conv7_sse41_j2_loop:
    movaps xmm12, [rsi + rdx + 3456]
    movaps xmm13, [rsi + rdx + 4224]
    movaps xmm14, [rsi + rdx + 4992]
    movups xmm8, [r13 + rdx]
    movups xmm9, [r11 + rdx]
    movups xmm10, [r14 + rdx]
    movups xmm11, [r10 + rdx]
    mulps xmm8, xmm12
    addps xmm0, xmm8
    movaps xmm8, xmm9
    mulps xmm8, xmm13
    addps xmm0, xmm8
    movaps xmm8, xmm10
    mulps xmm8, xmm14
    addps xmm0, xmm8
    mulps xmm9, xmm12
    addps xmm4, xmm9
    mulps xmm10, xmm13
    addps xmm4, xmm10
    mulps xmm11, xmm14
    addps xmm4, xmm11
    movups xmm8, [r13 + rdx + 256]
    movups xmm9, [r11 + rdx + 256]
    movups xmm10, [r14 + rdx + 256]
    movups xmm11, [r10 + rdx + 256]
    mulps xmm8, xmm12
    addps xmm1, xmm8
    movaps xmm8, xmm9
    mulps xmm8, xmm13
    addps xmm1, xmm8
    movaps xmm8, xmm10
    mulps xmm8, xmm14
    addps xmm1, xmm8
    mulps xmm9, xmm12
    addps xmm5, xmm9
    mulps xmm10, xmm13
    addps xmm5, xmm10
    mulps xmm11, xmm14
    addps xmm5, xmm11
    movups xmm8, [r13 + rdx + 512]
    movups xmm9, [r11 + rdx + 512]
    movups xmm10, [r14 + rdx + 512]
    movups xmm11, [r10 + rdx + 512]
    mulps xmm8, xmm12
    addps xmm2, xmm8
    movaps xmm8, xmm9
    mulps xmm8, xmm13
    addps xmm2, xmm8
    movaps xmm8, xmm10
    mulps xmm8, xmm14
    addps xmm2, xmm8
    mulps xmm9, xmm12
    addps xmm6, xmm9
    mulps xmm10, xmm13
    addps xmm6, xmm10
    mulps xmm11, xmm14
    addps xmm6, xmm11
    movups xmm8, [r13 + rdx + 768]
    movups xmm9, [r11 + rdx + 768]
    movups xmm10, [r14 + rdx + 768]
    movups xmm11, [r10 + rdx + 768]
    mulps xmm8, xmm12
    addps xmm3, xmm8
    movaps xmm8, xmm9
    mulps xmm8, xmm13
    addps xmm3, xmm8
    movaps xmm8, xmm10
    mulps xmm8, xmm14
    addps xmm3, xmm8
    mulps xmm9, xmm12
    addps xmm7, xmm9
    mulps xmm10, xmm13
    addps xmm7, xmm10
    mulps xmm11, xmm14
    addps xmm7, xmm11
    add rdx, 16
    cmp rdx, 768
    jl conv7_sse41_j2_loop
    xorps xmm15, xmm15
    mov r12, [rsp + 264]
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 192]
    jge conv7_sse41_row0_only
    add rbx, r12
    lea rdx, [rcx]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c1_0
    haddps xmm4, xmm4
    haddps xmm4, xmm4
    maxss xmm4, xmm15
    movss DWORD PTR [rbx + r8], xmm4
.Lconv7_sse41_c1_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c1_1
    haddps xmm5, xmm5
    haddps xmm5, xmm5
    maxss xmm5, xmm15
    movss DWORD PTR [rbx + r8 + 448], xmm5
.Lconv7_sse41_c1_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c1_2
    haddps xmm6, xmm6
    haddps xmm6, xmm6
    maxss xmm6, xmm15
    movss DWORD PTR [rbx + r8 + 896], xmm6
.Lconv7_sse41_c1_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c1_3
    haddps xmm7, xmm7
    haddps xmm7, xmm7
    maxss xmm7, xmm15
    movss DWORD PTR [rbx + r8 + 1344], xmm7
.Lconv7_sse41_c1_3:
    sub rbx, r12
conv7_sse41_row0_only:
    lea rdx, [rcx]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c0_0
    haddps xmm0, xmm0
    haddps xmm0, xmm0
    maxss xmm0, xmm15
    movss DWORD PTR [rbx + r8], xmm0
.Lconv7_sse41_c0_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c0_1
    haddps xmm1, xmm1
    haddps xmm1, xmm1
    maxss xmm1, xmm15
    movss DWORD PTR [rbx + r8 + 448], xmm1
.Lconv7_sse41_c0_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c0_2
    haddps xmm2, xmm2
    haddps xmm2, xmm2
    maxss xmm2, xmm15
    movss DWORD PTR [rbx + r8 + 896], xmm2
.Lconv7_sse41_c0_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 184]
    jge .Lconv7_sse41_c0_3
    haddps xmm3, xmm3
    haddps xmm3, xmm3
    maxss xmm3, xmm15
    movss DWORD PTR [rbx + r8 + 1344], xmm3
.Lconv7_sse41_c0_3:
    add rsi, 5760
    add rbp, 4
    add r8, 4
    jmp conv7_sse41_ii_loop
conv7_sse41_ii_done:
    add rcx, 4
    jmp conv7_sse41_x_loop
conv7_sse41_x_done:
    add rax, 2
    jmp conv7_sse41_y_loop
conv7_sse41_y_done:
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
    .cfi_def_cfa_offset 64
    pop r15
    .cfi_def_cfa_offset 56
    pop r14
    .cfi_def_cfa_offset 48
    pop r13
    .cfi_def_cfa_offset 40
    pop r12
    .cfi_def_cfa_offset 32
    pop rbp
    .cfi_def_cfa_offset 24
    pop rdi
    .cfi_def_cfa_offset 16
    pop rsi
    .cfi_def_cfa_offset 8
    pop rbx
    .cfi_def_cfa_offset 0
    ret 
.cfi_endproc
.size ray_conv7_sse41, .-ray_conv7_sse41
.section .note.GNU-stack,"",@progbits
