.intel_syntax noprefix
.text
.globl ray_conv15_avx2
.type ray_conv15_avx2, @function
ray_conv15_avx2:
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
    sub rsp, 312
    .cfi_def_cfa_offset 376
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
    mov rsi, [rsp + 384]
    mov [rsp + 160], rsi
    mov rbp, [rsp + 392]
    mov [rsp + 168], rbp
    mov rax, [rsp + 400]
    mov [rsp + 176], rax
    mov ecx, [rsp + 408]
    imul r12, rcx, 16
    imul r15, r9, 128
    mov eax, edx
    vpxor xmm15, xmm15, xmm15
conv15_avx2_y_loop:
    cmp rax, [rsp + 192]
    jge conv15_avx2_y_done
    mov rcx, [rsp + 200]
conv15_avx2_x_loop:
    cmp rcx, [rsp + 184]
    jge conv15_avx2_x_done
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
    add rbx, [rsp + 176]
    imul rdx, rcx, 16
    add rbx, rdx
    xor r8d, r8d
    mov rsi, [rsp + 160]
    mov rbp, [rsp + 168]
conv15_avx2_ii_loop:
    cmp r8, 12
    jge conv15_avx2_ii_done
    vmovss xmm0, DWORD PTR [rbp]
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
conv15_avx2_j_loop:
    vmovaps ymm12, [rsi + rdx]
    vmovaps ymm13, [rsi + rdx + 384]
    vmovaps ymm14, [rsi + rdx + 768]
    vmovups ymm8, [r13 + rdx]
    vmovups ymm9, [r11 + rdx]
    vmovups ymm10, [r14 + rdx]
    vmovups ymm11, [r10 + rdx]
    vfmadd231ps ymm0, ymm8, ymm12
    vfmadd231ps ymm0, ymm9, ymm13
    vfmadd231ps ymm0, ymm10, ymm14
    vfmadd231ps ymm4, ymm9, ymm12
    vfmadd231ps ymm4, ymm10, ymm13
    vfmadd231ps ymm4, ymm11, ymm14
    vmovups ymm8, [r13 + rdx + 128]
    vmovups ymm9, [r11 + rdx + 128]
    vmovups ymm10, [r14 + rdx + 128]
    vmovups ymm11, [r10 + rdx + 128]
    vfmadd231ps ymm1, ymm8, ymm12
    vfmadd231ps ymm1, ymm9, ymm13
    vfmadd231ps ymm1, ymm10, ymm14
    vfmadd231ps ymm5, ymm9, ymm12
    vfmadd231ps ymm5, ymm10, ymm13
    vfmadd231ps ymm5, ymm11, ymm14
    vmovups ymm8, [r13 + rdx + 256]
    vmovups ymm9, [r11 + rdx + 256]
    vmovups ymm10, [r14 + rdx + 256]
    vmovups ymm11, [r10 + rdx + 256]
    vfmadd231ps ymm2, ymm8, ymm12
    vfmadd231ps ymm2, ymm9, ymm13
    vfmadd231ps ymm2, ymm10, ymm14
    vfmadd231ps ymm6, ymm9, ymm12
    vfmadd231ps ymm6, ymm10, ymm13
    vfmadd231ps ymm6, ymm11, ymm14
    vmovups ymm8, [r13 + rdx + 384]
    vmovups ymm9, [r11 + rdx + 384]
    vmovups ymm10, [r14 + rdx + 384]
    vmovups ymm11, [r10 + rdx + 384]
    vfmadd231ps ymm3, ymm8, ymm12
    vfmadd231ps ymm3, ymm9, ymm13
    vfmadd231ps ymm3, ymm10, ymm14
    vfmadd231ps ymm7, ymm9, ymm12
    vfmadd231ps ymm7, ymm10, ymm13
    vfmadd231ps ymm7, ymm11, ymm14
    add rdx, 32
    cmp rdx, 384
    jl conv15_avx2_j_loop
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 192]
    jge conv15_avx2_row0_spill
    vhaddps ymm4, ymm4, ymm4
    vhaddps ymm4, ymm4, ymm4
    vperm2f128 ymm8, ymm4, ymm4, 1
    vaddps ymm4, ymm8, ymm4
    maxss xmm4, xmm15
    vmovss DWORD PTR [rsp + 232], xmm4
    vhaddps ymm5, ymm5, ymm5
    vhaddps ymm5, ymm5, ymm5
    vperm2f128 ymm8, ymm5, ymm5, 1
    vaddps ymm5, ymm8, ymm5
    maxss xmm5, xmm15
    vmovss DWORD PTR [rsp + 236], xmm5
    vhaddps ymm6, ymm6, ymm6
    vhaddps ymm6, ymm6, ymm6
    vperm2f128 ymm8, ymm6, ymm6, 1
    vaddps ymm6, ymm8, ymm6
    maxss xmm6, xmm15
    vmovss DWORD PTR [rsp + 240], xmm6
    vhaddps ymm7, ymm7, ymm7
    vhaddps ymm7, ymm7, ymm7
    vperm2f128 ymm8, ymm7, ymm7, 1
    vaddps ymm7, ymm8, ymm7
    maxss xmm7, xmm15
    vmovss DWORD PTR [rsp + 244], xmm7
conv15_avx2_row0_spill:
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vperm2f128 ymm8, ymm0, ymm0, 1
    vaddps ymm0, ymm8, ymm0
    maxss xmm0, xmm15
    vmovss DWORD PTR [rsp + 216], xmm0
    vhaddps ymm1, ymm1, ymm1
    vhaddps ymm1, ymm1, ymm1
    vperm2f128 ymm8, ymm1, ymm1, 1
    vaddps ymm1, ymm8, ymm1
    maxss xmm1, xmm15
    vmovss DWORD PTR [rsp + 220], xmm1
    vhaddps ymm2, ymm2, ymm2
    vhaddps ymm2, ymm2, ymm2
    vperm2f128 ymm8, ymm2, ymm2, 1
    vaddps ymm2, ymm8, ymm2
    maxss xmm2, xmm15
    vmovss DWORD PTR [rsp + 224], xmm2
    vhaddps ymm3, ymm3, ymm3
    vhaddps ymm3, ymm3, ymm3
    vperm2f128 ymm8, ymm3, ymm3, 1
    vaddps ymm3, ymm8, ymm3
    maxss xmm3, xmm15
    vmovss DWORD PTR [rsp + 228], xmm3
    mov [rsp + 208], rax
    mov [rsp + 248], rcx
    mov [rsp + 256], r9
    mov [rsp + 264], r8
    mov [rsp + 272], rdi
    mov [rsp + 280], rsi
    mov [rsp + 288], r11
    lea r14, [rip + conv15_avx2_conv15_hdr_c]
    mov r13, r8
    mov rdx, [rsp + 208]
    add rdx, 1
    cmp rdx, [rsp + 192]
    jge conv15_avx2_row0_store
    add rbx, r12
    mov rdx, [rsp + 248]
    add rdx, 0
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c1_0
    vmovss xmm0, DWORD PTR [rsp + 232]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c1_0_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c1_0_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c1_0_dn
.Lconv15_avx2_c1_0_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c1_0_dn
.Lconv15_avx2_c1_0_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c1_0_dn:
    vmovss DWORD PTR [rbx + r13], xmm0
.Lconv15_avx2_c1_0:
    mov rdx, [rsp + 248]
    add rdx, 1
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c1_1
    vmovss xmm0, DWORD PTR [rsp + 236]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c1_1_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c1_1_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c1_1_dn
.Lconv15_avx2_c1_1_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c1_1_dn
.Lconv15_avx2_c1_1_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c1_1_dn:
    vmovss DWORD PTR [rbx + r13 + 16], xmm0
.Lconv15_avx2_c1_1:
    mov rdx, [rsp + 248]
    add rdx, 2
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c1_2
    vmovss xmm0, DWORD PTR [rsp + 240]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c1_2_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c1_2_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c1_2_dn
.Lconv15_avx2_c1_2_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c1_2_dn
.Lconv15_avx2_c1_2_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c1_2_dn:
    vmovss DWORD PTR [rbx + r13 + 32], xmm0
.Lconv15_avx2_c1_2:
    mov rdx, [rsp + 248]
    add rdx, 3
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c1_3
    vmovss xmm0, DWORD PTR [rsp + 244]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c1_3_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c1_3_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c1_3_dn
.Lconv15_avx2_c1_3_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c1_3_dn
.Lconv15_avx2_c1_3_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c1_3_dn:
    vmovss DWORD PTR [rbx + r13 + 48], xmm0
.Lconv15_avx2_c1_3:
    sub rbx, r12
conv15_avx2_row0_store:
    mov rdx, [rsp + 248]
    add rdx, 0
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c0_0
    vmovss xmm0, DWORD PTR [rsp + 216]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c0_0_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c0_0_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c0_0_dn
.Lconv15_avx2_c0_0_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c0_0_dn
.Lconv15_avx2_c0_0_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c0_0_dn:
    vmovss DWORD PTR [rbx + r13], xmm0
.Lconv15_avx2_c0_0:
    mov rdx, [rsp + 248]
    add rdx, 1
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c0_1
    vmovss xmm0, DWORD PTR [rsp + 220]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c0_1_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c0_1_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c0_1_dn
.Lconv15_avx2_c0_1_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c0_1_dn
.Lconv15_avx2_c0_1_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c0_1_dn:
    vmovss DWORD PTR [rbx + r13 + 16], xmm0
.Lconv15_avx2_c0_1:
    mov rdx, [rsp + 248]
    add rdx, 2
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c0_2
    vmovss xmm0, DWORD PTR [rsp + 224]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c0_2_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c0_2_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c0_2_dn
.Lconv15_avx2_c0_2_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c0_2_dn
.Lconv15_avx2_c0_2_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c0_2_dn:
    vmovss DWORD PTR [rbx + r13 + 32], xmm0
.Lconv15_avx2_c0_2:
    mov rdx, [rsp + 248]
    add rdx, 3
    cmp rdx, [rsp + 184]
    jge .Lconv15_avx2_c0_3
    vmovss xmm0, DWORD PTR [rsp + 228]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_avx2_c0_3_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_avx2_c0_3_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_avx2_c0_3_dn
.Lconv15_avx2_c0_3_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    vmovss xmm1, DWORD PTR [r14 + 32]
    call powf
    jmp .Lconv15_avx2_c0_3_dn
.Lconv15_avx2_c0_3_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_avx2_c0_3_dn:
    vmovss DWORD PTR [rbx + r13 + 48], xmm0
.Lconv15_avx2_c0_3:
    mov rax, [rsp + 208]
    mov rcx, [rsp + 248]
    mov r9, [rsp + 256]
    mov r8, [rsp + 264]
    mov rdi, [rsp + 272]
    mov rsi, [rsp + 280]
    mov r11, [rsp + 288]
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
    jmp conv15_avx2_ii_loop
conv15_avx2_ii_done:
    add rcx, 4
    jmp conv15_avx2_x_loop
conv15_avx2_x_done:
    add rax, 2
    jmp conv15_avx2_y_loop
conv15_avx2_y_done:
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
    add rsp, 312
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
.size ray_conv15_avx2, .-ray_conv15_avx2
.p2align 2
conv15_avx2_conv15_hdr_c:
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
