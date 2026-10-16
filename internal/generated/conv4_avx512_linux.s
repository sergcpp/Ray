.intel_syntax noprefix
.text
.globl ray_conv4_avx512
.type ray_conv4_avx512, @function
ray_conv4_avx512:
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
    sub rsp, 696
    .cfi_def_cfa_offset 760
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
    mov eax, ecx
    add eax, esi
    mov [rsp + 664], rax
    mov eax, esi
    mov [rsp + 680], rax
    mov eax, edx
    add eax, r8d
    mov [rsp + 672], rax
    mov rsi, [rsp + 768]
    mov [rsp + 640], rsi
    mov rbp, [rsp + 776]
    mov [rsp + 648], rbp
    mov rax, [rsp + 784]
    mov [rsp + 656], rax
    mov ecx, [rsp + 792]
    imul r12, rcx, 320
    imul r15, r9, 256
    mov eax, edx
    vpxord zmm19, zmm19, zmm19
conv4_avx512_y_loop:
    cmp rax, [rsp + 672]
    jge conv4_avx512_y_done
    mov rcx, [rsp + 680]
conv4_avx512_x_loop:
    cmp rcx, [rsp + 664]
    jge conv4_avx512_x_done
    mov rdx, rax
    imul rdx, r9
    lea rdx, [rdx + rcx - 1]
    imul rdx, rdx, 256
    lea r11, [rdi + rdx]
    mov r13, r11
    sub r13, r15
    mov r14, r11
    add r14, r15
    mov r10, r14
    add r10, r15
    mov rbx, rax
    shr rbx, 1
    imul rbx, r12
    add rbx, [rsp + 656]
    mov rdx, rcx
    shr rdx, 1
    imul rdx, rdx, 320
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 640]
    mov rbp, [rsp + 648]
conv4_avx512_ii_loop:
    cmp r8, 320
    jge conv4_avx512_ii_done
    vmovss xmm0, DWORD PTR [rbp]
    vmovaps zmm1, zmm0
    vmovaps zmm2, zmm0
    vmovaps zmm3, zmm0
    vmovaps zmm4, zmm0
    vmovaps zmm5, zmm0
    vmovaps zmm6, zmm0
    vmovaps zmm7, zmm0
    vmovaps zmm8, zmm0
    vmovaps zmm9, zmm0
    vmovaps zmm10, zmm0
    vmovaps zmm11, zmm0
    vmovaps zmm6, zmm0
    vmovaps zmm7, zmm1
    vmovaps zmm8, zmm2
    vmovaps zmm9, zmm3
    vmovaps zmm10, zmm4
    vmovaps zmm11, zmm5
    vmovaps zmm12, zmm6
    vmovaps zmm13, zmm7
    vmovaps zmm14, zmm8
    vmovaps zmm15, zmm9
    xor edx, edx
conv4_avx512_j_loop:
    vmovaps zmm16, [rsi + rdx]
    vmovaps zmm17, [rsi + rdx + 768]
    vmovaps zmm18, [rsi + rdx + 1536]
    vmovups zmm12, [r13 + rdx]
    vmovups zmm13, [r11 + rdx]
    vmovups zmm14, [r14 + rdx]
    vmovups zmm15, [r10 + rdx]
    vfmadd231ps zmm0, zmm12, zmm16
    vfmadd231ps zmm0, zmm13, zmm17
    vfmadd231ps zmm0, zmm14, zmm18
    vfmadd231ps zmm6, zmm13, zmm16
    vfmadd231ps zmm6, zmm14, zmm17
    vfmadd231ps zmm6, zmm15, zmm18
    vmovups zmm12, [r13 + rdx + 256]
    vmovups zmm13, [r11 + rdx + 256]
    vmovups zmm14, [r14 + rdx + 256]
    vmovups zmm15, [r10 + rdx + 256]
    vfmadd231ps zmm1, zmm12, zmm16
    vfmadd231ps zmm1, zmm13, zmm17
    vfmadd231ps zmm1, zmm14, zmm18
    vfmadd231ps zmm7, zmm13, zmm16
    vfmadd231ps zmm7, zmm14, zmm17
    vfmadd231ps zmm7, zmm15, zmm18
    vmovups zmm12, [r13 + rdx + 512]
    vmovups zmm13, [r11 + rdx + 512]
    vmovups zmm14, [r14 + rdx + 512]
    vmovups zmm15, [r10 + rdx + 512]
    vfmadd231ps zmm2, zmm12, zmm16
    vfmadd231ps zmm2, zmm13, zmm17
    vfmadd231ps zmm2, zmm14, zmm18
    vfmadd231ps zmm8, zmm13, zmm16
    vfmadd231ps zmm8, zmm14, zmm17
    vfmadd231ps zmm8, zmm15, zmm18
    vmovups zmm12, [r13 + rdx + 768]
    vmovups zmm13, [r11 + rdx + 768]
    vmovups zmm14, [r14 + rdx + 768]
    vmovups zmm15, [r10 + rdx + 768]
    vfmadd231ps zmm3, zmm12, zmm16
    vfmadd231ps zmm3, zmm13, zmm17
    vfmadd231ps zmm3, zmm14, zmm18
    vfmadd231ps zmm9, zmm13, zmm16
    vfmadd231ps zmm9, zmm14, zmm17
    vfmadd231ps zmm9, zmm15, zmm18
    vmovups zmm12, [r13 + rdx + 1024]
    vmovups zmm13, [r11 + rdx + 1024]
    vmovups zmm14, [r14 + rdx + 1024]
    vmovups zmm15, [r10 + rdx + 1024]
    vfmadd231ps zmm4, zmm12, zmm16
    vfmadd231ps zmm4, zmm13, zmm17
    vfmadd231ps zmm4, zmm14, zmm18
    vfmadd231ps zmm10, zmm13, zmm16
    vfmadd231ps zmm10, zmm14, zmm17
    vfmadd231ps zmm10, zmm15, zmm18
    vmovups zmm12, [r13 + rdx + 1280]
    vmovups zmm13, [r11 + rdx + 1280]
    vmovups zmm14, [r14 + rdx + 1280]
    vmovups zmm15, [r10 + rdx + 1280]
    vfmadd231ps zmm5, zmm12, zmm16
    vfmadd231ps zmm5, zmm13, zmm17
    vfmadd231ps zmm5, zmm14, zmm18
    vfmadd231ps zmm11, zmm13, zmm16
    vfmadd231ps zmm11, zmm14, zmm17
    vfmadd231ps zmm11, zmm15, zmm18
    add rdx, 64
    cmp rdx, 768
    jl conv4_avx512_j_loop
    vextractf32x8 ymm12, zmm6, 1
    vaddps ymm6, ymm6, ymm12
    vextractf32x4 xmm12, ymm6, 1
    vaddps xmm6, xmm6, xmm12
    vpsrldq xmm12, xmm6, 8
    vaddps xmm6, xmm6, xmm12
    vpsrldq xmm12, xmm6, 4
    vaddss xmm6, xmm6, xmm12
    vmaxss xmm6, xmm6, xmm19
    vextractf32x8 ymm12, zmm7, 1
    vaddps ymm7, ymm7, ymm12
    vextractf32x4 xmm12, ymm7, 1
    vaddps xmm7, xmm7, xmm12
    vpsrldq xmm12, xmm7, 8
    vaddps xmm7, xmm7, xmm12
    vpsrldq xmm12, xmm7, 4
    vaddss xmm7, xmm7, xmm12
    vmaxss xmm7, xmm7, xmm19
    vextractf32x8 ymm12, zmm8, 1
    vaddps ymm8, ymm8, ymm12
    vextractf32x4 xmm12, ymm8, 1
    vaddps xmm8, xmm8, xmm12
    vpsrldq xmm12, xmm8, 8
    vaddps xmm8, xmm8, xmm12
    vpsrldq xmm12, xmm8, 4
    vaddss xmm8, xmm8, xmm12
    vmaxss xmm8, xmm8, xmm19
    vextractf32x8 ymm12, zmm9, 1
    vaddps ymm9, ymm9, ymm12
    vextractf32x4 xmm12, ymm9, 1
    vaddps xmm9, xmm9, xmm12
    vpsrldq xmm12, xmm9, 8
    vaddps xmm9, xmm9, xmm12
    vpsrldq xmm12, xmm9, 4
    vaddss xmm9, xmm9, xmm12
    vmaxss xmm9, xmm9, xmm19
    vextractf32x8 ymm12, zmm10, 1
    vaddps ymm10, ymm10, ymm12
    vextractf32x4 xmm12, ymm10, 1
    vaddps xmm10, xmm10, xmm12
    vpsrldq xmm12, xmm10, 8
    vaddps xmm10, xmm10, xmm12
    vpsrldq xmm12, xmm10, 4
    vaddss xmm10, xmm10, xmm12
    vmaxss xmm10, xmm10, xmm19
    vextractf32x8 ymm12, zmm11, 1
    vaddps ymm11, ymm11, ymm12
    vextractf32x4 xmm12, ymm11, 1
    vaddps xmm11, xmm11, xmm12
    vpsrldq xmm12, xmm11, 8
    vaddps xmm11, xmm11, xmm12
    vpsrldq xmm12, xmm11, 4
    vaddss xmm11, xmm11, xmm12
    vmaxss xmm11, xmm11, xmm19
    vextractf32x8 ymm12, zmm0, 1
    vaddps ymm0, ymm0, ymm12
    vextractf32x4 xmm12, ymm0, 1
    vaddps xmm0, xmm0, xmm12
    vpsrldq xmm12, xmm0, 8
    vaddps xmm0, xmm0, xmm12
    vpsrldq xmm12, xmm0, 4
    vaddss xmm0, xmm0, xmm12
    vmaxss xmm0, xmm0, xmm19
    vextractf32x8 ymm12, zmm1, 1
    vaddps ymm1, ymm1, ymm12
    vextractf32x4 xmm12, ymm1, 1
    vaddps xmm1, xmm1, xmm12
    vpsrldq xmm12, xmm1, 8
    vaddps xmm1, xmm1, xmm12
    vpsrldq xmm12, xmm1, 4
    vaddss xmm1, xmm1, xmm12
    vmaxss xmm1, xmm1, xmm19
    vextractf32x8 ymm12, zmm2, 1
    vaddps ymm2, ymm2, ymm12
    vextractf32x4 xmm12, ymm2, 1
    vaddps xmm2, xmm2, xmm12
    vpsrldq xmm12, xmm2, 8
    vaddps xmm2, xmm2, xmm12
    vpsrldq xmm12, xmm2, 4
    vaddss xmm2, xmm2, xmm12
    vmaxss xmm2, xmm2, xmm19
    vextractf32x8 ymm12, zmm3, 1
    vaddps ymm3, ymm3, ymm12
    vextractf32x4 xmm12, ymm3, 1
    vaddps xmm3, xmm3, xmm12
    vpsrldq xmm12, xmm3, 8
    vaddps xmm3, xmm3, xmm12
    vpsrldq xmm12, xmm3, 4
    vaddss xmm3, xmm3, xmm12
    vmaxss xmm3, xmm3, xmm19
    vextractf32x8 ymm12, zmm4, 1
    vaddps ymm4, ymm4, ymm12
    vextractf32x4 xmm12, ymm4, 1
    vaddps xmm4, xmm4, xmm12
    vpsrldq xmm12, xmm4, 8
    vaddps xmm4, xmm4, xmm12
    vpsrldq xmm12, xmm4, 4
    vaddss xmm4, xmm4, xmm12
    vmaxss xmm4, xmm4, xmm19
    vextractf32x8 ymm12, zmm5, 1
    vaddps ymm5, ymm5, ymm12
    vextractf32x4 xmm12, ymm5, 1
    vaddps xmm5, xmm5, xmm12
    vpsrldq xmm12, xmm5, 8
    vaddps xmm5, xmm5, xmm12
    vpsrldq xmm12, xmm5, 4
    vaddss xmm5, xmm5, xmm12
    vmaxss xmm5, xmm5, xmm19
    lea rdx, [rcx]
    cmp rdx, [rsp + 664]
    jge .Lconv4_avx512_p0
    vmaxss xmm1, xmm1, xmm0
    vmaxss xmm7, xmm7, xmm6
    vmaxss xmm7, xmm7, xmm1
    vmaxss xmm7, xmm7, xmm19
    vmovss DWORD PTR [rbx + r8], xmm7
.Lconv4_avx512_p0:
    lea rdx, [rcx + 2]
    cmp rdx, [rsp + 664]
    jge .Lconv4_avx512_p2
    vmaxss xmm3, xmm3, xmm2
    vmaxss xmm9, xmm9, xmm8
    vmaxss xmm9, xmm9, xmm3
    vmaxss xmm9, xmm9, xmm19
    vmovss DWORD PTR [rbx + r8 + 320], xmm9
.Lconv4_avx512_p2:
    lea rdx, [rcx + 4]
    cmp rdx, [rsp + 664]
    jge .Lconv4_avx512_p4
    vmaxss xmm5, xmm5, xmm4
    vmaxss xmm11, xmm11, xmm10
    vmaxss xmm11, xmm11, xmm5
    vmaxss xmm11, xmm11, xmm19
    vmovss DWORD PTR [rbx + r8 + 640], xmm11
.Lconv4_avx512_p4:
    add rsi, 2304
    add rbp, 4
    add r8, 4
    jmp conv4_avx512_ii_loop
conv4_avx512_ii_done:
    add rcx, 6
    jmp conv4_avx512_x_loop
conv4_avx512_x_done:
    add rax, 2
    jmp conv4_avx512_y_loop
conv4_avx512_y_done:
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
    add rsp, 696
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
.size ray_conv4_avx512, .-ray_conv4_avx512
.section .note.GNU-stack,"",@progbits
