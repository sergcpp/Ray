.intel_syntax noprefix
.section __TEXT,__text,regular,pure_instructions
.globl _ray_conv7_avx512
_ray_conv7_avx512:
    push rbx
    push rsi
    push rdi
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 752
    vmovups [rsp], zmm6
    vmovups [rsp + 64], zmm7
    vmovups [rsp + 128], zmm8
    vmovups [rsp + 192], zmm9
    vmovups [rsp + 256], zmm10
    vmovups [rsp + 320], zmm11
    vmovups [rsp + 384], zmm12
    vmovups [rsp + 448], zmm13
    vmovups [rsp + 512], zmm14
    vmovups [rsp + 576], zmm15
    mov [rsp + 688], rsi
    mov eax, r8d
    add eax, edx
    mov [rsp + 664], rax
    mov eax, edx
    mov [rsp + 680], rax
    mov eax, ecx
    add eax, r9d
    mov [rsp + 672], rax
    mov r9d, [rsp + 824]
    mov rax, r9
    imul rax, rax, 384
    mov [rsp + 696], rax
    mov r8d, [rsp + 832]
    mov rax, r8
    imul rax, rax, 256
    mov [rsp + 704], rax
    mov rsi, [rsp + 840]
    mov [rsp + 640], rsi
    mov rbp, [rsp + 848]
    mov [rsp + 648], rbp
    mov rax, [rsp + 856]
    mov [rsp + 656], rax
    mov eax, [rsp + 864]
    imul r12, rax, 448
    mov [rsp + 744], r12
    mov eax, ecx
    vpxor xmm15, xmm15, xmm15
conv7_avx512_y_loop:
    cmp rax, [rsp + 672]
    jge conv7_avx512_y_done
    mov r15, [rsp + 696]
    mov r9, rax
    shr r9, 1
    imul r9, r15
    mov [rsp + 720], r9
    mov r9, rax
    add r9, 1
    shr r9, 1
    imul r9, r15
    mov [rsp + 728], r9
    mov r9, rax
    add r9, 2
    shr r9, 1
    imul r9, r15
    mov [rsp + 736], r9
    cmp rax, 1
    jb .Lconv7_avx512_r0neg
    mov r9, rax
    sub r9, 1
    shr r9, 1
    imul r9, r15
    jmp .Lconv7_avx512_r0
.Lconv7_avx512_r0neg:
    mov r9, r15
    neg r9
.Lconv7_avx512_r0:
    mov [rsp + 712], r9
    mov rcx, [rsp + 680]
conv7_avx512_x_loop:
    cmp rcx, [rsp + 664]
    jge conv7_avx512_x_done
    mov r12, [rsp + 744]
    mov rbx, rax
    imul rbx, r12
    add rbx, [rsp + 656]
    imul rdx, rcx, 448
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 640]
    mov rbp, [rsp + 648]
conv7_avx512_ii_loop:
    cmp r8, 448
    jge conv7_avx512_ii_done
    vmovss xmm0, DWORD PTR [rbp]
    vmovaps zmm1, zmm0
    vmovaps zmm2, zmm0
    vmovaps zmm3, zmm0
    vmovaps zmm4, zmm0
    vmovaps zmm5, zmm0
    vmovaps zmm6, zmm0
    vmovaps zmm7, zmm0
    vmovaps zmm4, zmm0
    vmovaps zmm5, zmm1
    vmovaps zmm6, zmm2
    vmovaps zmm7, zmm3
    vmovaps zmm8, zmm4
    vmovaps zmm9, zmm5
    vmovaps zmm10, zmm6
    vmovaps zmm11, zmm7
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
conv7_avx512_j1w0:
    vmovaps zmm12, [rsi + rdx]
    vmovaps zmm13, [rsi + rdx + 1152]
    vmovaps zmm14, [rsi + rdx + 2304]
    mov r9, [rsp + 712]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm12
    vfmadd231ps zmm2, zmm10, zmm12
    vfmadd231ps zmm3, zmm11, zmm12
    mov r9, [rsp + 720]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm13
    vfmadd231ps zmm4, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm13
    vfmadd231ps zmm5, zmm9, zmm12
    vfmadd231ps zmm2, zmm10, zmm13
    vfmadd231ps zmm6, zmm10, zmm12
    vfmadd231ps zmm3, zmm11, zmm13
    vfmadd231ps zmm7, zmm11, zmm12
    mov r9, [rsp + 728]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm14
    vfmadd231ps zmm4, zmm8, zmm13
    vfmadd231ps zmm1, zmm9, zmm14
    vfmadd231ps zmm5, zmm9, zmm13
    vfmadd231ps zmm2, zmm10, zmm14
    vfmadd231ps zmm6, zmm10, zmm13
    vfmadd231ps zmm3, zmm11, zmm14
    vfmadd231ps zmm7, zmm11, zmm13
    mov r9, [rsp + 736]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm4, zmm8, zmm14
    vfmadd231ps zmm5, zmm9, zmm14
    vfmadd231ps zmm6, zmm10, zmm14
    vfmadd231ps zmm7, zmm11, zmm14
    add rdx, 64
    add r10, 64
    add r11, 64
    add r13, 64
    add r12, 64
    cmp rdx, 384
    jl conv7_avx512_j1w0
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
conv7_avx512_j1w1:
    vmovaps zmm12, [rsi + rdx]
    vmovaps zmm13, [rsi + rdx + 1152]
    vmovaps zmm14, [rsi + rdx + 2304]
    mov r9, [rsp + 712]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm12
    vfmadd231ps zmm2, zmm10, zmm12
    vfmadd231ps zmm3, zmm11, zmm12
    mov r9, [rsp + 720]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm13
    vfmadd231ps zmm4, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm13
    vfmadd231ps zmm5, zmm9, zmm12
    vfmadd231ps zmm2, zmm10, zmm13
    vfmadd231ps zmm6, zmm10, zmm12
    vfmadd231ps zmm3, zmm11, zmm13
    vfmadd231ps zmm7, zmm11, zmm12
    mov r9, [rsp + 728]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm14
    vfmadd231ps zmm4, zmm8, zmm13
    vfmadd231ps zmm1, zmm9, zmm14
    vfmadd231ps zmm5, zmm9, zmm13
    vfmadd231ps zmm2, zmm10, zmm14
    vfmadd231ps zmm6, zmm10, zmm13
    vfmadd231ps zmm3, zmm11, zmm14
    vfmadd231ps zmm7, zmm11, zmm13
    mov r9, [rsp + 736]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm4, zmm8, zmm14
    vfmadd231ps zmm5, zmm9, zmm14
    vfmadd231ps zmm6, zmm10, zmm14
    vfmadd231ps zmm7, zmm11, zmm14
    add rdx, 64
    add r10, 64
    add r11, 64
    add r13, 64
    add r12, 64
    cmp rdx, 768
    jl conv7_avx512_j1w1
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
conv7_avx512_j1w2:
    vmovaps zmm12, [rsi + rdx]
    vmovaps zmm13, [rsi + rdx + 1152]
    vmovaps zmm14, [rsi + rdx + 2304]
    mov r9, [rsp + 712]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm12
    vfmadd231ps zmm2, zmm10, zmm12
    vfmadd231ps zmm3, zmm11, zmm12
    mov r9, [rsp + 720]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm13
    vfmadd231ps zmm4, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm13
    vfmadd231ps zmm5, zmm9, zmm12
    vfmadd231ps zmm2, zmm10, zmm13
    vfmadd231ps zmm6, zmm10, zmm12
    vfmadd231ps zmm3, zmm11, zmm13
    vfmadd231ps zmm7, zmm11, zmm12
    mov r9, [rsp + 728]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm0, zmm8, zmm14
    vfmadd231ps zmm4, zmm8, zmm13
    vfmadd231ps zmm1, zmm9, zmm14
    vfmadd231ps zmm5, zmm9, zmm13
    vfmadd231ps zmm2, zmm10, zmm14
    vfmadd231ps zmm6, zmm10, zmm13
    vfmadd231ps zmm3, zmm11, zmm14
    vfmadd231ps zmm7, zmm11, zmm13
    mov r9, [rsp + 736]
    vmovups zmm8, [r10 + r9]
    vmovups zmm9, [r11 + r9]
    vmovups zmm10, [r13 + r9]
    vmovups zmm11, [r12 + r9]
    vfmadd231ps zmm4, zmm8, zmm14
    vfmadd231ps zmm5, zmm9, zmm14
    vfmadd231ps zmm6, zmm10, zmm14
    vfmadd231ps zmm7, zmm11, zmm14
    add rdx, 64
    add r10, 64
    add r11, 64
    add r13, 64
    add r12, 64
    cmp rdx, 1152
    jl conv7_avx512_j1w2
    mov r11, [rsp + 688]
    mov rdx, rax
    imul rdx, [rsp + 704]
    mov r9, rcx
    sub r9, 1
    imul r9, r9, 256
    add rdx, r9
    add r11, rdx
    mov r13, r11
    sub r13, [rsp + 704]
    mov r14, r11
    add r14, [rsp + 704]
    mov r10, r14
    add r10, [rsp + 704]
    xor edx, edx
conv7_avx512_j2_loop:
    vmovaps zmm12, [rsi + rdx + 3456]
    vmovaps zmm13, [rsi + rdx + 4224]
    vmovaps zmm14, [rsi + rdx + 4992]
    vmovups zmm8, [r13 + rdx]
    vmovups zmm9, [r11 + rdx]
    vmovups zmm10, [r14 + rdx]
    vmovups zmm11, [r10 + rdx]
    vfmadd231ps zmm0, zmm8, zmm12
    vfmadd231ps zmm0, zmm9, zmm13
    vfmadd231ps zmm0, zmm10, zmm14
    vfmadd231ps zmm4, zmm9, zmm12
    vfmadd231ps zmm4, zmm10, zmm13
    vfmadd231ps zmm4, zmm11, zmm14
    vmovups zmm8, [r13 + rdx + 256]
    vmovups zmm9, [r11 + rdx + 256]
    vmovups zmm10, [r14 + rdx + 256]
    vmovups zmm11, [r10 + rdx + 256]
    vfmadd231ps zmm1, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm13
    vfmadd231ps zmm1, zmm10, zmm14
    vfmadd231ps zmm5, zmm9, zmm12
    vfmadd231ps zmm5, zmm10, zmm13
    vfmadd231ps zmm5, zmm11, zmm14
    vmovups zmm8, [r13 + rdx + 512]
    vmovups zmm9, [r11 + rdx + 512]
    vmovups zmm10, [r14 + rdx + 512]
    vmovups zmm11, [r10 + rdx + 512]
    vfmadd231ps zmm2, zmm8, zmm12
    vfmadd231ps zmm2, zmm9, zmm13
    vfmadd231ps zmm2, zmm10, zmm14
    vfmadd231ps zmm6, zmm9, zmm12
    vfmadd231ps zmm6, zmm10, zmm13
    vfmadd231ps zmm6, zmm11, zmm14
    vmovups zmm8, [r13 + rdx + 768]
    vmovups zmm9, [r11 + rdx + 768]
    vmovups zmm10, [r14 + rdx + 768]
    vmovups zmm11, [r10 + rdx + 768]
    vfmadd231ps zmm3, zmm8, zmm12
    vfmadd231ps zmm3, zmm9, zmm13
    vfmadd231ps zmm3, zmm10, zmm14
    vfmadd231ps zmm7, zmm9, zmm12
    vfmadd231ps zmm7, zmm10, zmm13
    vfmadd231ps zmm7, zmm11, zmm14
    add rdx, 64
    cmp rdx, 768
    jl conv7_avx512_j2_loop
    mov r12, [rsp + 744]
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 672]
    jge conv7_avx512_row0_only
    add rbx, r12
    lea rdx, [rcx]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c1_0
    vextractf32x8 ymm8, zmm4, 1
    vaddps ymm4, ymm4, ymm8
    vextractf32x4 xmm8, ymm4, 1
    vaddps xmm4, xmm4, xmm8
    vpsrldq xmm8, xmm4, 8
    vaddps xmm4, xmm4, xmm8
    vpsrldq xmm8, xmm4, 4
    vaddss xmm4, xmm4, xmm8
    maxss xmm4, xmm15
    vmovss DWORD PTR [rbx + r8], xmm4
.Lconv7_avx512_c1_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c1_1
    vextractf32x8 ymm8, zmm5, 1
    vaddps ymm5, ymm5, ymm8
    vextractf32x4 xmm8, ymm5, 1
    vaddps xmm5, xmm5, xmm8
    vpsrldq xmm8, xmm5, 8
    vaddps xmm5, xmm5, xmm8
    vpsrldq xmm8, xmm5, 4
    vaddss xmm5, xmm5, xmm8
    maxss xmm5, xmm15
    vmovss DWORD PTR [rbx + r8 + 448], xmm5
.Lconv7_avx512_c1_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c1_2
    vextractf32x8 ymm8, zmm6, 1
    vaddps ymm6, ymm6, ymm8
    vextractf32x4 xmm8, ymm6, 1
    vaddps xmm6, xmm6, xmm8
    vpsrldq xmm8, xmm6, 8
    vaddps xmm6, xmm6, xmm8
    vpsrldq xmm8, xmm6, 4
    vaddss xmm6, xmm6, xmm8
    maxss xmm6, xmm15
    vmovss DWORD PTR [rbx + r8 + 896], xmm6
.Lconv7_avx512_c1_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c1_3
    vextractf32x8 ymm8, zmm7, 1
    vaddps ymm7, ymm7, ymm8
    vextractf32x4 xmm8, ymm7, 1
    vaddps xmm7, xmm7, xmm8
    vpsrldq xmm8, xmm7, 8
    vaddps xmm7, xmm7, xmm8
    vpsrldq xmm8, xmm7, 4
    vaddss xmm7, xmm7, xmm8
    maxss xmm7, xmm15
    vmovss DWORD PTR [rbx + r8 + 1344], xmm7
.Lconv7_avx512_c1_3:
    sub rbx, r12
conv7_avx512_row0_only:
    lea rdx, [rcx]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c0_0
    vextractf32x8 ymm8, zmm0, 1
    vaddps ymm0, ymm0, ymm8
    vextractf32x4 xmm8, ymm0, 1
    vaddps xmm0, xmm0, xmm8
    vpsrldq xmm8, xmm0, 8
    vaddps xmm0, xmm0, xmm8
    vpsrldq xmm8, xmm0, 4
    vaddss xmm0, xmm0, xmm8
    maxss xmm0, xmm15
    vmovss DWORD PTR [rbx + r8], xmm0
.Lconv7_avx512_c0_0:
    lea rdx, [rcx + 1]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c0_1
    vextractf32x8 ymm8, zmm1, 1
    vaddps ymm1, ymm1, ymm8
    vextractf32x4 xmm8, ymm1, 1
    vaddps xmm1, xmm1, xmm8
    vpsrldq xmm8, xmm1, 8
    vaddps xmm1, xmm1, xmm8
    vpsrldq xmm8, xmm1, 4
    vaddss xmm1, xmm1, xmm8
    maxss xmm1, xmm15
    vmovss DWORD PTR [rbx + r8 + 448], xmm1
.Lconv7_avx512_c0_1:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c0_2
    vextractf32x8 ymm8, zmm2, 1
    vaddps ymm2, ymm2, ymm8
    vextractf32x4 xmm8, ymm2, 1
    vaddps xmm2, xmm2, xmm8
    vpsrldq xmm8, xmm2, 8
    vaddps xmm2, xmm2, xmm8
    vpsrldq xmm8, xmm2, 4
    vaddss xmm2, xmm2, xmm8
    maxss xmm2, xmm15
    vmovss DWORD PTR [rbx + r8 + 896], xmm2
.Lconv7_avx512_c0_2:
    lea rdx, [rcx + 3]
    cmp rdx, [rsp + 664]
    jge .Lconv7_avx512_c0_3
    vextractf32x8 ymm8, zmm3, 1
    vaddps ymm3, ymm3, ymm8
    vextractf32x4 xmm8, ymm3, 1
    vaddps xmm3, xmm3, xmm8
    vpsrldq xmm8, xmm3, 8
    vaddps xmm3, xmm3, xmm8
    vpsrldq xmm8, xmm3, 4
    vaddss xmm3, xmm3, xmm8
    maxss xmm3, xmm15
    vmovss DWORD PTR [rbx + r8 + 1344], xmm3
.Lconv7_avx512_c0_3:
    add rsi, 5760
    add rbp, 4
    add r8, 4
    jmp conv7_avx512_ii_loop
conv7_avx512_ii_done:
    add rcx, 4
    jmp conv7_avx512_x_loop
conv7_avx512_x_done:
    add rax, 2
    jmp conv7_avx512_y_loop
conv7_avx512_y_done:
    vmovups zmm6, [rsp]
    vmovups zmm7, [rsp + 64]
    vmovups zmm8, [rsp + 128]
    vmovups zmm9, [rsp + 192]
    vmovups zmm10, [rsp + 256]
    vmovups zmm11, [rsp + 320]
    vmovups zmm12, [rsp + 384]
    vmovups zmm13, [rsp + 448]
    vmovups zmm14, [rsp + 512]
    vmovups zmm15, [rsp + 576]
    add rsp, 752
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rdi
    pop rsi
    pop rbx
    ret 
