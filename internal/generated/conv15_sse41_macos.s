.intel_syntax noprefix
.section __TEXT,__text,regular,pure_instructions
.globl _ray_conv15_sse41
_ray_conv15_sse41:
    push rbx
    push rsi
    push rdi
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 312
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
    xorps xmm15, xmm15
conv15_sse41_y_loop:
    cmp rax, [rsp + 192]
    jge conv15_sse41_y_done
    mov rcx, [rsp + 200]
conv15_sse41_x_loop:
    cmp rcx, [rsp + 184]
    jge conv15_sse41_x_done
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
conv15_sse41_ii_loop:
    cmp r8, 12
    jge conv15_sse41_ii_done
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
conv15_sse41_j_loop:
    movaps xmm12, [rsi + rdx]
    movaps xmm13, [rsi + rdx + 384]
    movaps xmm14, [rsi + rdx + 768]
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
    movups xmm8, [r13 + rdx + 128]
    movups xmm9, [r11 + rdx + 128]
    movups xmm10, [r14 + rdx + 128]
    movups xmm11, [r10 + rdx + 128]
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
    movups xmm8, [r13 + rdx + 256]
    movups xmm9, [r11 + rdx + 256]
    movups xmm10, [r14 + rdx + 256]
    movups xmm11, [r10 + rdx + 256]
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
    movups xmm8, [r13 + rdx + 384]
    movups xmm9, [r11 + rdx + 384]
    movups xmm10, [r14 + rdx + 384]
    movups xmm11, [r10 + rdx + 384]
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
    cmp rdx, 384
    jl conv15_sse41_j_loop
    lea rdx, [rax + 1]
    cmp rdx, [rsp + 192]
    jge conv15_sse41_row0_spill
    haddps xmm4, xmm4
    haddps xmm4, xmm4
    maxss xmm4, xmm15
    movss DWORD PTR [rsp + 232], xmm4
    haddps xmm5, xmm5
    haddps xmm5, xmm5
    maxss xmm5, xmm15
    movss DWORD PTR [rsp + 236], xmm5
    haddps xmm6, xmm6
    haddps xmm6, xmm6
    maxss xmm6, xmm15
    movss DWORD PTR [rsp + 240], xmm6
    haddps xmm7, xmm7
    haddps xmm7, xmm7
    maxss xmm7, xmm15
    movss DWORD PTR [rsp + 244], xmm7
conv15_sse41_row0_spill:
    haddps xmm0, xmm0
    haddps xmm0, xmm0
    maxss xmm0, xmm15
    movss DWORD PTR [rsp + 216], xmm0
    haddps xmm1, xmm1
    haddps xmm1, xmm1
    maxss xmm1, xmm15
    movss DWORD PTR [rsp + 220], xmm1
    haddps xmm2, xmm2
    haddps xmm2, xmm2
    maxss xmm2, xmm15
    movss DWORD PTR [rsp + 224], xmm2
    haddps xmm3, xmm3
    haddps xmm3, xmm3
    maxss xmm3, xmm15
    movss DWORD PTR [rsp + 228], xmm3
    mov [rsp + 208], rax
    mov [rsp + 248], rcx
    mov [rsp + 256], r9
    mov [rsp + 264], r8
    mov [rsp + 272], rdi
    mov [rsp + 280], rsi
    mov [rsp + 288], r11
    lea r14, [rip + conv15_sse41_conv15_hdr_c]
    mov r13, r8
    mov rdx, [rsp + 208]
    add rdx, 1
    cmp rdx, [rsp + 192]
    jge conv15_sse41_row0_store
    add rbx, r12
    mov rdx, [rsp + 248]
    add rdx, 0
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c1_0
    movss xmm0, DWORD PTR [rsp + 232]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c1_0_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c1_0_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c1_0_dn
.Lconv15_sse41_c1_0_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c1_0_dn
.Lconv15_sse41_c1_0_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c1_0_dn:
    movss DWORD PTR [rbx + r13], xmm0
.Lconv15_sse41_c1_0:
    mov rdx, [rsp + 248]
    add rdx, 1
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c1_1
    movss xmm0, DWORD PTR [rsp + 236]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c1_1_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c1_1_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c1_1_dn
.Lconv15_sse41_c1_1_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c1_1_dn
.Lconv15_sse41_c1_1_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c1_1_dn:
    movss DWORD PTR [rbx + r13 + 16], xmm0
.Lconv15_sse41_c1_1:
    mov rdx, [rsp + 248]
    add rdx, 2
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c1_2
    movss xmm0, DWORD PTR [rsp + 240]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c1_2_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c1_2_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c1_2_dn
.Lconv15_sse41_c1_2_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c1_2_dn
.Lconv15_sse41_c1_2_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c1_2_dn:
    movss DWORD PTR [rbx + r13 + 32], xmm0
.Lconv15_sse41_c1_2:
    mov rdx, [rsp + 248]
    add rdx, 3
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c1_3
    movss xmm0, DWORD PTR [rsp + 244]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c1_3_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c1_3_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c1_3_dn
.Lconv15_sse41_c1_3_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c1_3_dn
.Lconv15_sse41_c1_3_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c1_3_dn:
    movss DWORD PTR [rbx + r13 + 48], xmm0
.Lconv15_sse41_c1_3:
    sub rbx, r12
conv15_sse41_row0_store:
    mov rdx, [rsp + 248]
    add rdx, 0
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c0_0
    movss xmm0, DWORD PTR [rsp + 216]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c0_0_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c0_0_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c0_0_dn
.Lconv15_sse41_c0_0_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c0_0_dn
.Lconv15_sse41_c0_0_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c0_0_dn:
    movss DWORD PTR [rbx + r13], xmm0
.Lconv15_sse41_c0_0:
    mov rdx, [rsp + 248]
    add rdx, 1
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c0_1
    movss xmm0, DWORD PTR [rsp + 220]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c0_1_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c0_1_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c0_1_dn
.Lconv15_sse41_c0_1_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c0_1_dn
.Lconv15_sse41_c0_1_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c0_1_dn:
    movss DWORD PTR [rbx + r13 + 16], xmm0
.Lconv15_sse41_c0_1:
    mov rdx, [rsp + 248]
    add rdx, 2
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c0_2
    movss xmm0, DWORD PTR [rsp + 224]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c0_2_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c0_2_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c0_2_dn
.Lconv15_sse41_c0_2_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c0_2_dn
.Lconv15_sse41_c0_2_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c0_2_dn:
    movss DWORD PTR [rbx + r13 + 32], xmm0
.Lconv15_sse41_c0_2:
    mov rdx, [rsp + 248]
    add rdx, 3
    cmp rdx, [rsp + 184]
    jge .Lconv15_sse41_c0_3
    movss xmm0, DWORD PTR [rsp + 228]
    mulss xmm0, DWORD PTR [r14]
    comiss xmm0, DWORD PTR [r14 + 4]
    jbe .Lconv15_sse41_c0_3_lo
    comiss xmm0, DWORD PTR [r14 + 8]
    jbe .Lconv15_sse41_c0_3_mi
    subss xmm0, DWORD PTR [r14 + 12]
    divss xmm0, DWORD PTR [r14 + 16]
    call _expf
    subss xmm0, DWORD PTR [r14 + 20]
    jmp .Lconv15_sse41_c0_3_dn
.Lconv15_sse41_c0_3_mi:
    subss xmm0, DWORD PTR [r14 + 24]
    divss xmm0, DWORD PTR [r14 + 28]
    movss xmm1, DWORD PTR [r14 + 32]
    call _powf
    jmp .Lconv15_sse41_c0_3_dn
.Lconv15_sse41_c0_3_lo:
    divss xmm0, DWORD PTR [r14 + 36]
.Lconv15_sse41_c0_3_dn:
    movss DWORD PTR [rbx + r13 + 48], xmm0
.Lconv15_sse41_c0_3:
    mov rax, [rsp + 208]
    mov rcx, [rsp + 248]
    mov r9, [rsp + 256]
    mov r8, [rsp + 264]
    mov rdi, [rsp + 272]
    mov rsi, [rsp + 280]
    mov r11, [rsp + 288]
    xorps xmm15, xmm15
    mov r13, r11
    sub r13, r15
    mov r14, r11
    add r14, r15
    mov r10, r14
    add r10, r15
    add rsi, 1152
    add rbp, 4
    add r8, 4
    jmp conv15_sse41_ii_loop
conv15_sse41_ii_done:
    add rcx, 4
    jmp conv15_sse41_x_loop
conv15_sse41_x_done:
    add rax, 2
    jmp conv15_sse41_y_loop
conv15_sse41_y_done:
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
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rdi
    pop rsi
    pop rbx
    ret 
.p2align 2
conv15_sse41_conv15_hdr_c:
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
