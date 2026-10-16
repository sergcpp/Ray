.intel_syntax noprefix
.section __TEXT,__text,regular,pure_instructions
.globl _ray_conv5_sse41
_ray_conv5_sse41:
    push rbx
    push rsi
    push rdi
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 216
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
    mov eax, ecx
    add eax, esi
    mov [rsp + 184], rax
    mov eax, esi
    mov [rsp + 200], rax
    mov eax, edx
    add eax, r8d
    mov [rsp + 192], rax
    mov rsi, [rsp + 288]
    mov [rsp + 160], rsi
    mov rbp, [rsp + 296]
    mov [rsp + 168], rbp
    mov rax, [rsp + 304]
    mov [rsp + 176], rax
    mov ecx, [rsp + 312]
    imul r12, rcx, 384
    imul r15, r9, 320
    mov eax, edx
    xorps xmm15, xmm15
conv5_sse41_y_loop:
    cmp rax, [rsp + 192]
    jge conv5_sse41_y_done
    mov rcx, [rsp + 200]
conv5_sse41_x_loop:
    cmp rcx, [rsp + 184]
    jge conv5_sse41_x_done
    mov rdx, rax
    imul rdx, r9
    lea rdx, [rdx + rcx - 1]
    imul rdx, rdx, 320
    lea r11, [rdi + rdx]
    mov r13, r11
    sub r13, r15
    mov r14, r11
    add r14, r15
    mov r10, r14
    add r10, r15
    mov rbx, rax
    imul rbx, r12
    add rbx, [rsp + 176]
    imul rdx, rcx, 384
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 160]
    mov rbp, [rsp + 168]
conv5_sse41_ii_loop:
    cmp r8, 384
    jge conv5_sse41_ii_done
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
conv5_sse41_j_loop:
    movaps xmm12, [rsi + rdx]
    movaps xmm13, [rsi + rdx + 960]
    movaps xmm14, [rsi + rdx + 1920]
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
    movups xmm8, [r13 + rdx + 320]
    movups xmm9, [r11 + rdx + 320]
    movups xmm10, [r14 + rdx + 320]
    movups xmm11, [r10 + rdx + 320]
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
    movups xmm8, [r13 + rdx + 640]
    movups xmm9, [r11 + rdx + 640]
    movups xmm10, [r14 + rdx + 640]
    movups xmm11, [r10 + rdx + 640]
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
    movups xmm8, [r13 + rdx + 960]
    movups xmm9, [r11 + rdx + 960]
    movups xmm10, [r14 + rdx + 960]
    movups xmm11, [r10 + rdx + 960]
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
    cmp rdx, 960
    jl conv5_sse41_j_loop
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 192]
    jge conv5_sse41_row0_only
    add rbx, r12
    lea rdx, [rcx]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c1_0
    haddps xmm4, xmm4
    haddps xmm4, xmm4
    maxss xmm4, xmm15
    movss DWORD PTR [rbx + r8], xmm4
.Lconv5_sse41_c1_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c1_1
    haddps xmm5, xmm5
    haddps xmm5, xmm5
    maxss xmm5, xmm15
    movss DWORD PTR [rbx + r8 + 384], xmm5
.Lconv5_sse41_c1_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c1_2
    haddps xmm6, xmm6
    haddps xmm6, xmm6
    maxss xmm6, xmm15
    movss DWORD PTR [rbx + r8 + 768], xmm6
.Lconv5_sse41_c1_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c1_3
    haddps xmm7, xmm7
    haddps xmm7, xmm7
    maxss xmm7, xmm15
    movss DWORD PTR [rbx + r8 + 1152], xmm7
.Lconv5_sse41_c1_3:
    sub rbx, r12
conv5_sse41_row0_only:
    lea rdx, [rcx]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c0_0
    haddps xmm0, xmm0
    haddps xmm0, xmm0
    maxss xmm0, xmm15
    movss DWORD PTR [rbx + r8], xmm0
.Lconv5_sse41_c0_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c0_1
    haddps xmm1, xmm1
    haddps xmm1, xmm1
    maxss xmm1, xmm15
    movss DWORD PTR [rbx + r8 + 384], xmm1
.Lconv5_sse41_c0_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c0_2
    haddps xmm2, xmm2
    haddps xmm2, xmm2
    maxss xmm2, xmm15
    movss DWORD PTR [rbx + r8 + 768], xmm2
.Lconv5_sse41_c0_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 184]
    jge .Lconv5_sse41_c0_3
    haddps xmm3, xmm3
    haddps xmm3, xmm3
    maxss xmm3, xmm15
    movss DWORD PTR [rbx + r8 + 1152], xmm3
.Lconv5_sse41_c0_3:
    add rsi, 2880
    add rbp, 4
    add r8, 4
    jmp conv5_sse41_ii_loop
conv5_sse41_ii_done:
    add rcx, 4
    jmp conv5_sse41_x_loop
conv5_sse41_x_done:
    add rax, 2
    jmp conv5_sse41_y_loop
conv5_sse41_y_done:
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
    add rsp, 216
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rdi
    pop rsi
    pop rbx
    ret 
