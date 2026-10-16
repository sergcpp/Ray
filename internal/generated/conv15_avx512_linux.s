.intel_syntax noprefix
.text
.globl ray_conv15_avx512
.type ray_conv15_avx512, @function
ray_conv15_avx512:
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
    sub rsp, 792
    .cfi_def_cfa_offset 856
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
    mov rsi, [rsp + 864]
    mov [rsp + 640], rsi
    mov rbp, [rsp + 872]
    mov [rsp + 648], rbp
    mov rax, [rsp + 880]
    mov [rsp + 656], rax
    mov ecx, [rsp + 888]
    imul r12, rcx, 16
    imul r15, r9, 128
    mov eax, edx
    vpxor xmm15, xmm15, xmm15
conv15_avx512_y_loop:
    cmp rax, [rsp + 672]
    jge conv15_avx512_y_done
    mov rcx, [rsp + 680]
conv15_avx512_x_loop:
    cmp rcx, [rsp + 664]
    jge conv15_avx512_x_done
    mov rdx, rax
    imul rdx, r9
    lea rdx, [rdx + rcx - 1]
    imul rdx, rdx, 128
    lea r11, [rdi + rdx]
    mov r13, r11
    sub r13, r15
    mov r14, r11
    add r14, r15
    mov r10, r14
    add r10, r15
    mov rbx, rax
    imul rbx, r12
    add rbx, [rsp + 656]
    imul rdx, rcx, 16
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 640]
    mov rbp, [rsp + 648]
conv15_avx512_ii_loop:
    cmp r8, 12
    jge conv15_avx512_ii_done
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
conv15_avx512_j_loop:
    vmovaps zmm12, [rsi + rdx]
    vmovaps zmm13, [rsi + rdx + 384]
    vmovaps zmm14, [rsi + rdx + 768]
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
    vmovups zmm8, [r13 + rdx + 128]
    vmovups zmm9, [r11 + rdx + 128]
    vmovups zmm10, [r14 + rdx + 128]
    vmovups zmm11, [r10 + rdx + 128]
    vfmadd231ps zmm1, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm13
    vfmadd231ps zmm1, zmm10, zmm14
    vfmadd231ps zmm5, zmm9, zmm12
    vfmadd231ps zmm5, zmm10, zmm13
    vfmadd231ps zmm5, zmm11, zmm14
    vmovups zmm8, [r13 + rdx + 256]
    vmovups zmm9, [r11 + rdx + 256]
    vmovups zmm10, [r14 + rdx + 256]
    vmovups zmm11, [r10 + rdx + 256]
    vfmadd231ps zmm2, zmm8, zmm12
    vfmadd231ps zmm2, zmm9, zmm13
    vfmadd231ps zmm2, zmm10, zmm14
    vfmadd231ps zmm6, zmm9, zmm12
    vfmadd231ps zmm6, zmm10, zmm13
    vfmadd231ps zmm6, zmm11, zmm14
    vmovups zmm8, [r13 + rdx + 384]
    vmovups zmm9, [r11 + rdx + 384]
    vmovups zmm10, [r14 + rdx + 384]
    vmovups zmm11, [r10 + rdx + 384]
    vfmadd231ps zmm3, zmm8, zmm12
    vfmadd231ps zmm3, zmm9, zmm13
    vfmadd231ps zmm3, zmm10, zmm14
    vfmadd231ps zmm7, zmm9, zmm12
    vfmadd231ps zmm7, zmm10, zmm13
    vfmadd231ps zmm7, zmm11, zmm14
    add rdx, 64
    cmp rdx, 384
    jl conv15_avx512_j_loop
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 672]
    jge conv15_avx512_row0_spill
    vextractf32x8 ymm8, zmm4, 1
    vaddps ymm4, ymm4, ymm8
    vextractf32x4 xmm8, ymm4, 1
    vaddps xmm4, xmm4, xmm8
    vpsrldq xmm8, xmm4, 8
    vaddps xmm4, xmm4, xmm8
    vpsrldq xmm8, xmm4, 4
    vaddss xmm4, xmm4, xmm8
    maxss xmm4, xmm15
    vmovss DWORD PTR [rsp + 712], xmm4
    vextractf32x8 ymm8, zmm5, 1
    vaddps ymm5, ymm5, ymm8
    vextractf32x4 xmm8, ymm5, 1
    vaddps xmm5, xmm5, xmm8
    vpsrldq xmm8, xmm5, 8
    vaddps xmm5, xmm5, xmm8
    vpsrldq xmm8, xmm5, 4
    vaddss xmm5, xmm5, xmm8
    maxss xmm5, xmm15
    vmovss DWORD PTR [rsp + 716], xmm5
    vextractf32x8 ymm8, zmm6, 1
    vaddps ymm6, ymm6, ymm8
    vextractf32x4 xmm8, ymm6, 1
    vaddps xmm6, xmm6, xmm8
    vpsrldq xmm8, xmm6, 8
    vaddps xmm6, xmm6, xmm8
    vpsrldq xmm8, xmm6, 4
    vaddss xmm6, xmm6, xmm8
    maxss xmm6, xmm15
    vmovss DWORD PTR [rsp + 720], xmm6
    vextractf32x8 ymm8, zmm7, 1
    vaddps ymm7, ymm7, ymm8
    vextractf32x4 xmm8, ymm7, 1
    vaddps xmm7, xmm7, xmm8
    vpsrldq xmm8, xmm7, 8
    vaddps xmm7, xmm7, xmm8
    vpsrldq xmm8, xmm7, 4
    vaddss xmm7, xmm7, xmm8
    maxss xmm7, xmm15
    vmovss DWORD PTR [rsp + 724], xmm7
conv15_avx512_row0_spill:
    vextractf32x8 ymm8, zmm0, 1
    vaddps ymm0, ymm0, ymm8
    vextractf32x4 xmm8, ymm0, 1
    vaddps xmm0, xmm0, xmm8
    vpsrldq xmm8, xmm0, 8
    vaddps xmm0, xmm0, xmm8
    vpsrldq xmm8, xmm0, 4
    vaddss xmm0, xmm0, xmm8
    maxss xmm0, xmm15
    vmovss DWORD PTR [rsp + 696], xmm0
    vextractf32x8 ymm8, zmm1, 1
    vaddps ymm1, ymm1, ymm8
    vextractf32x4 xmm8, ymm1, 1
    vaddps xmm1, xmm1, xmm8
    vpsrldq xmm8, xmm1, 8
    vaddps xmm1, xmm1, xmm8
    vpsrldq xmm8, xmm1, 4
    vaddss xmm1, xmm1, xmm8
    maxss xmm1, xmm15
    vmovss DWORD PTR [rsp + 700], xmm1
    vextractf32x8 ymm8, zmm2, 1
    vaddps ymm2, ymm2, ymm8
    vextractf32x4 xmm8, ymm2, 1
    vaddps xmm2, xmm2, xmm8
    vpsrldq xmm8, xmm2, 8
    vaddps xmm2, xmm2, xmm8
    vpsrldq xmm8, xmm2, 4
    vaddss xmm2, xmm2, xmm8
    maxss xmm2, xmm15
    vmovss DWORD PTR [rsp + 704], xmm2
    vextractf32x8 ymm8, zmm3, 1
    vaddps ymm3, ymm3, ymm8
    vextractf32x4 xmm8, ymm3, 1
    vaddps xmm3, xmm3, xmm8
    vpsrldq xmm8, xmm3, 8
    vaddps xmm3, xmm3, xmm8
    vpsrldq xmm8, xmm3, 4
    vaddss xmm3, xmm3, xmm8
    maxss xmm3, xmm15
    vmovss DWORD PTR [rsp + 708], xmm3
    mov [rsp + 688], rax
    mov [rsp + 728], rcx
    mov [rsp + 736], r9
    mov [rsp + 744], r8
    mov [rsp + 752], rdi
    mov [rsp + 760], rsi
    mov [rsp + 768], r11
    lea r14, [rip + conv15_avx512_conv15_hdr_c]
    mov r13, r8
    mov rdx, [rsp + 688]
    add rdx, 1
    cmp rdx, [rsp + 672]
    jge conv15_avx512_row0_store
    add rbx, r12
    mov rdx, [rsp + 728]
    add rdx, 0
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c1_0
    vmovss xmm0, DWORD PTR [rsp + 712]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c1_0_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c1_0_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c1_0_dn
.Lconv15_avx512_c1_0_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c1_0_dn
.Lconv15_avx512_c1_0_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c1_0_dn:
    vmovss DWORD PTR [rbx + r13], xmm0
.Lconv15_avx512_c1_0:
    mov rdx, [rsp + 728]
    add rdx, 1
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c1_1
    vmovss xmm0, DWORD PTR [rsp + 716]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c1_1_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c1_1_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c1_1_dn
.Lconv15_avx512_c1_1_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c1_1_dn
.Lconv15_avx512_c1_1_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c1_1_dn:
    vmovss DWORD PTR [rbx + r13 + 16], xmm0
.Lconv15_avx512_c1_1:
    mov rdx, [rsp + 728]
    add rdx, 2
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c1_2
    vmovss xmm0, DWORD PTR [rsp + 720]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c1_2_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c1_2_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c1_2_dn
.Lconv15_avx512_c1_2_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c1_2_dn
.Lconv15_avx512_c1_2_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c1_2_dn:
    vmovss DWORD PTR [rbx + r13 + 32], xmm0
.Lconv15_avx512_c1_2:
    mov rdx, [rsp + 728]
    add rdx, 3
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c1_3
    vmovss xmm0, DWORD PTR [rsp + 724]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c1_3_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c1_3_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c1_3_dn
.Lconv15_avx512_c1_3_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c1_3_dn
.Lconv15_avx512_c1_3_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c1_3_dn:
    vmovss DWORD PTR [rbx + r13 + 48], xmm0
.Lconv15_avx512_c1_3:
    sub rbx, r12
conv15_avx512_row0_store:
    mov rdx, [rsp + 728]
    add rdx, 0
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c0_0
    vmovss xmm0, DWORD PTR [rsp + 696]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c0_0_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c0_0_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c0_0_dn
.Lconv15_avx512_c0_0_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c0_0_dn
.Lconv15_avx512_c0_0_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c0_0_dn:
    vmovss DWORD PTR [rbx + r13], xmm0
.Lconv15_avx512_c0_0:
    mov rdx, [rsp + 728]
    add rdx, 1
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c0_1
    vmovss xmm0, DWORD PTR [rsp + 700]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c0_1_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c0_1_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c0_1_dn
.Lconv15_avx512_c0_1_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c0_1_dn
.Lconv15_avx512_c0_1_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c0_1_dn:
    vmovss DWORD PTR [rbx + r13 + 16], xmm0
.Lconv15_avx512_c0_1:
    mov rdx, [rsp + 728]
    add rdx, 2
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c0_2
    vmovss xmm0, DWORD PTR [rsp + 704]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c0_2_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c0_2_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c0_2_dn
.Lconv15_avx512_c0_2_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c0_2_dn
.Lconv15_avx512_c0_2_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c0_2_dn:
    vmovss DWORD PTR [rbx + r13 + 32], xmm0
.Lconv15_avx512_c0_2:
    mov rdx, [rsp + 728]
    add rdx, 3
    cmp rdx, [rsp + 664]
    jge .Lconv15_avx512_c0_3
    vmovss xmm0, DWORD PTR [rsp + 708]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx512_c0_3_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx512_c0_3_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx512_c0_3_dn
.Lconv15_avx512_c0_3_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx512_c0_3_dn
.Lconv15_avx512_c0_3_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx512_c0_3_dn:
    vmovss DWORD PTR [rbx + r13 + 48], xmm0
.Lconv15_avx512_c0_3:
    mov rax, [rsp + 688]
    mov rcx, [rsp + 728]
    mov r9, [rsp + 736]
    mov r8, [rsp + 744]
    mov rdi, [rsp + 752]
    mov rsi, [rsp + 760]
    mov r11, [rsp + 768]
    vpxor xmm15, xmm15, xmm15
    mov r13, r11
    sub r13, r15
    mov r14, r11
    add r14, r15
    mov r10, r14
    add r10, r15
    add rsi, 1152
    add rbp, 4
    add r8, 4
    jmp conv15_avx512_ii_loop
conv15_avx512_ii_done:
    add rcx, 4
    jmp conv15_avx512_x_loop
conv15_avx512_x_done:
    add rax, 2
    jmp conv15_avx512_y_loop
conv15_avx512_y_done:
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
    add rsp, 792
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
.size ray_conv15_avx512, .-ray_conv15_avx512
.p2align 2
conv15_avx512_conv15_hdr_c:
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
