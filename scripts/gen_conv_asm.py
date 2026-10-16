#!/usr/bin/env python3
# Generates AVX2/AVX1/SSE41/AVX512 (x86) and NEON (AArch64) kernels for UNet conv passes
# as plain asm: MASM for MSVC, GAS Intel syntax for GCC/Clang (Linux) and Clang (macOS).
#
# Each kernel mirrors Ray::NS::Convolution3x3 (Convolution.h) instruction-for-
# instruction so golden tests stay bit-identical:
#   - bias in lane 0 only (movss), per-lane j-ascending chains (vfmadd231ps on AVX2/AVX512;
#     mulps+addps on SSE4.1, vmulps+vaddps on AVX1 - the C++ fmadd() lowers to mul-then-add
#     under /arch:SSE2 and /arch:AVX)
#   - hsum exactly as the simd header: vhaddps/vhaddps/vperm2f128/vaddps (fvec<8>),
#     haddps/haddps (fvec<4>)
#   - ReLU via maxss(x, 0): matches std::max(0.0f, x) incl. NaN and -0.0
#   - aligned B loads (movaps, weight rows are vector-aligned), unaligned A (movups)
#   - HDR post-op (pass 15): transfer::output<HDRTransfer> as a comiss/jbe chain calling the
#     same CRT powf/expf the C++ calls; loop state lives in stack slots across the calls
# SSE4.1 kernels use legacy (non-VEX) encodings - the SSE41 arch must run on CPUs
# without AVX. AVX1 kernels use VEX (ymm) encodings but no FMA - /arch:AVX has no FMA3.
#
# One kernel description per pass (channel counts are parameters), per-ISA/per-target
# emitters. The body is width-parameterized (V lanes/vector).
# kind='concat2' kernels mirror the 2-input ConvolutionConcat3x3 (PreOp1=Upsample):
# j1 walks input1 with Upsample addressing (tap = j/In1 unrolled, source row offsets
# row_r*stride1*COL1 in stack slots), j2 walks input2 plain; same accumulator chain order.
#
# Verified: MASM/ml64 assembles + test_Ray --no-gpu bit-identical on MSVC and ClangCL x64 (AVX2 and
# SSE41); clang integrated assembler accepts the Linux variants. The macOS variants are emitted
# WITHOUT CFI directives (Mach-O's CFI object writer rejects label-difference advance_loc
# expressions; the kernels are never unwound through - like the ml64/armasm64 objects) and are
# assembled by the CMake macOS build; Linux .s files keep CFI and carry .note.GNU-stack.
# NEON kernels (AArch64): armasm64 assembles all 15 (MSVC ARM64: Ray.lib/test_Ray/TestApp link
# with them); clang cross-assembles the _linux.s; bit-identity vs the C++ NEON kernel runs on
# ARM hardware (CI) - the FP lowering is mirrored instruction-for-instruction from an
# MSVC arm64 cl /FA listing of Convolution.h (faddp pair hsum, fmla, fcmp+fmax, fcmp+b.le hdr chain).
#
# Regenerate:  python src/Ray/scripts/gen_conv_asm.py
# Outputs:     src/Ray/internal/generated/conv<N>_{avx2,avx1,sse41,avx512}{.asm,_linux.s,_macos.s}
#              src/Ray/internal/generated/conv<N>_neon{.asm,_linux.s,_macos.s}   (AArch64,
#              armasm64 syntax on Windows; GAS AArch64 for Linux/macOS - see ARM64 emitters)
#              plus master TUs (conv_all_win_x64.asm, conv_all_{linux,macos}_{x86_64,aarch64}.s)
#              that pull the per-kernel sources into ONE object per target/arch - internal labels
#              are namespaced per kernel (ns_labels) so the kernels can be assembled in one pass;
#              Windows kernels carry no END directive (the x64 master supplies it; CMake
#              concatenates the 15 arm64 kernels via scripts/asm_concat.cmake and appends one).

import os
import re

TROWS_REQ = 6   # tile width for plain/downsample passes. AVX512: 6 wins on i9-10980XE (bench: conv8
                # 956 vs 1025@T4 vs 993@T8 ns/call, bit-identical); AVX2/SSE41 clamp to 4 (16 regs).
TROWS = 4       # active per-file tile width, recomputed by _set_plan() below
OB2 = False     # arm64 OUT_BLK=2 mode: two output channels share one channel walk (see _set_plan)
B3 = B4 = B5 = 0  # OB2 only: the second output channel's tap regs


class Isa:
    def __init__(self, name, V, fma, arch='x64', vex=None):
        self.name = name                       # 'avx2' | 'avx1' | 'sse41' | 'avx512' | 'neon'
        self.V = V                             # lanes/vector
        self.fma = fma                         # True: vfmadd231ps/fmla; False: mulps+addps
        self.vex = fma if vex is None else vex # True: VEX encodings (vmovups/vmulps ymm...);
                                               # False: legacy SSE (xmm only - SSE41)
        self.arch = arch                       # 'x64' | 'arm64'
        self.vec = 'zmm' if V == 16 else ('ymm' if V == 8 else 'xmm')  # wide vector prefix

    def v(self, n):
        return f'{self.vec}{n}'

    def y(self, n):
        return f'ymm{n}'

    def x(self, n):
        return f'xmm{n}'


AVX2 = Isa('avx2', 8, True)
AVX1 = Isa('avx1', 8, False, vex=True)  # VEX ymm encodings, no FMA (MSVC lowers fmadd() to mul-then-add)
SSE41 = Isa('sse41', 4, False)
AVX512 = Isa('avx512', 16, True)
NEON = Isa('neon', 4, True, 'arm64')


class Pass:
    def __init__(self, num, in_ch, out_ch, res_desc, downsample=False, pitch=None, post_op='none',
                 kind='plain', in2=None):
        self.num = num                       # denoise pass number (RendererCPU.h switch)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.res_desc = res_desc
        self.downsample = downsample         # ePostOp::Downsample: 2x2 max-pool, store at half coords
        self.post_op = post_op               # 'none' | 'hdr' (transfer::output<HDRTransfer>, calls powf/expf)
        self.kind = kind                     # 'plain' | 'concat2' (2-input ConvolutionConcat3x3, PreOp1=Upsample)
        self.in2 = in2                       # concat2: second input's channel count
        rt = 8 * ((3 * in_ch + 7) // 8)      # RoundedTriple (same rounding as the C++ kernel)
        self.J_END = 3 * in_ch * 4           # one tap block: 3 pixels * in_ch channels * 4 B
        self.RT = rt
        self.W_ROW = 3 * rt * 4              # bytes per output-channel weight row (3 tap blocks)
        self.TAP = rt * 4                    # bytes between tap blocks within a weight row
        self.PIX = (pitch or out_ch) * 4     # bytes per output pixel column (OutPxPitch * 4)
        self.COL = in_ch * 4                 # bytes per input pixel column within a row
        self.II_END = out_ch * 4             # r8 counts 4*ii (ii = output channel)
        if kind == 'concat2':
            # two j loops: input1 (Upsample addressing, weight block RT1) + input2 (plain, RT2)
            self.RT1 = rt
            self.RT2 = 8 * ((3 * in2 + 7) // 8)
            self.COL1, self.COL2 = in_ch * 4, in2 * 4
            self.J1_END, self.J2_END = 3 * in_ch * 4, 3 * in2 * 4
            self.TAP1, self.TAP2 = rt * 4, self.RT2 * 4
            self.W_OFF = 3 * rt * 4          # input2 weight block offset within a row
            self.W_ROW = (rt + self.RT2) * 3 * 4

    def frame(self, abi, isa):
        # hdr passes spill loop state + scalars and call powf/expf:
        #   win: prologue leaves rsp == 8 (mod 16), each call does sub/add rsp,40 (shadow below frame)
        #   sysv: prologue leaves rsp == 0 (mod 16), bare calls
        # concat2 needs slots for data2/row strides/row offsets/saved out stride.
        # AVX512: the zmm6-15 save area is 4x the xmm6-15 one (base 160 -> 640).
        base = 640 if isa.V == 16 else 160
        if self.kind == 'concat2':
            return base + 112
        if self.post_op != 'hdr':
            return base + 56
        # hdr + 3 extra slots (rdi/rsi/r11 - volatile on BOTH x86-64 ABIs, live across the
        # powf/expf calls): win needs base+144 (rsp == 8 after prologue), sysv base+152
        # (rsp == 0 -> bare calls aligned). Padding keeps those invariants (mod 16).
        return base + (144 if abi == 'win' else 152)

    def j_step(self, isa):
        # j/j2 loops are cmp-based: a partial final window (in_ch not a multiple of V, e.g. the
        # 9-channel staged inputs of passes 0/13) mirrors the C++ staged walk - the garbage
        # lanes multiply against the zero-padded weight tail. j1 is unrolled per tap block, so
        # input1 must be V-aligned (per-tap channel run an exact multiple of the window).
        if self.kind == 'concat2':
            assert self.J1_END % (isa.V * 4) == 0 and (self.in_ch * 4) % (isa.V * 4) == 0, "!"
        return isa.V * 4

    @property
    def tag(self):
        return f'conv{self.num}'

    def symbol(self, isa):
        return f'ray_conv{self.num}_{isa.name}'


PASSES = [
    # pass 0 walks the staged 3-input buffer (3+3+3ch, pre-ops applied by the C++ staging);
    # pass 13 walks data1 (Upsample) + the staged 4-input aux buffer (3+3+3ch). in=9 gives a
    # partial final j window - see j_step.
    Pass(0, 9, 32, '9 staged (3+3+3)->32 @ full res, enc conv 0'),
    Pass(1, 32, 32, '32->32 @ full res -> 1/2, downsample', downsample=True),
    Pass(2, 32, 48, '32->48 @ 1/2 -> 1/4, downsample', downsample=True),
    Pass(3, 48, 64, '48->64 @ 1/4 -> 1/8, downsample', downsample=True),
    Pass(4, 64, 80, '64->80 @ 1/8 -> 1/16, downsample', downsample=True),
    Pass(5, 80, 96, '80->96 @ 1/16 res'),
    Pass(6, 96, 96, '96->96 @ 1/16 res, also serves pass 10 @ 1/4 res'),
    Pass(7, 96, 112, '96(+64)->112 @ 1/8 res, concat Upsample', kind='concat2', in2=64),
    Pass(8, 112, 112, '112->112 @ 1/8 res'),
    Pass(9, 112, 96, '112(+48)->96 @ 1/4 res, concat Upsample', kind='concat2', in2=48),
    Pass(11, 96, 64, '96(+32)->64 @ 1/2 res, concat Upsample', kind='concat2', in2=32),
    Pass(12, 64, 64, '64->64 @ 1/2 res'),
    Pass(13, 64, 64, '64(+9 staged aux)->64 @ full res, concat Upsample', kind='concat2', in2=9),
    Pass(14, 64, 32, '64->32 @ full res'),
    Pass(15, 32, 3, '32->3 @ full res, HDR post-op (final pass)', pitch=4, post_op='hdr'),
]

ISAS = [AVX2, AVX1, SSE41, AVX512]
ISAS_ARM = [NEON]


class Target:
    def __init__(self, name, syntax, abi):
        self.name = name
        self.syntax = syntax                 # 'masm' | 'gas'
        self.abi = abi                       # 'win' | 'sysv'

    def sym(self, p, isa):
        return ('_' if 'macos' in self.name else '') + p.symbol(isa)

    def r(self, reg):
        return reg  # GAS .intel_syntax uses bare register names (AT&T's % is not used here)

    def m(self, base=None, index=None, disp=0):
        s = ''
        if base:
            s += self.r(base)
        if index:
            s += ' + ' + self.r(index)
        if disp:
            s += (' - ' + str(-disp)) if disp < 0 else (' + ' + str(disp))
        return '[' + (s if s else '0') + ']'

    def dword(self, mem):
        # ml64 needs the size keyword for movss memory operands; GAS Intel accepts it too
        return ('dword ptr ' if self.syntax == 'masm' else 'DWORD PTR ') + mem

    def csym(self, name):
        # libm callee (powf/expf); macOS prefixes C symbols with '_'
        return ('_' if 'macos' in self.name else '') + name

    def lea_sym(self, reg, label):
        return self.i('lea', self.r(reg), f'[rip + {label}]' if self.syntax == 'gas' else f'OFFSET {label}')

    def const_block(self, label, values):
        # float constants as raw bit patterns (decimal: ml64 rejects 0x hex)
        if self.syntax == 'masm':
            return ['align 4', f'{label}:'] + [f'    dd {v}' for v in values]
        return ['.p2align 2', f'{label}:'] + [f'    .long {v}' for v in values]

    def i(self, op, *args):
        return '    ' + op + ' ' + ', '.join(args)

    def lbl(self, name):
        # MASM identifiers cannot start with '.', GAS local labels conventionally do
        return ('.L' + name) if self.syntax == 'gas' else ('L' + name)


CALLEE_SAVED = ('rbx', 'rsi', 'rdi', 'rbp', 'r12', 'r13', 'r14', 'r15')  # Win x64 and SysV agree
# Win64 also callee-saves xmm6-xmm15 (SysV does not); spill them on all targets for one layout.
# AVX512 kernels use zmm6-15 as accumulators, so the save area goes to 64 B/vector (640 B).
XMM_SAVE = tuple(range(6, 16))

# stack slots (rsp-relative, after prologue): weights_g, biases_g, output_g, x_end, y_end, rect_x;
# hdr/cc2 add their slots at S_BASE+48. All recomputed per file by _set_plan().
S_BASE = S_W = S_B = S_O = S_XE = S_YE = S_RX = 0
S_Y = S_SC = S_X = S_ST = S_II = 0
S_D2 = S_ST1 = S_ST2 = S_R0 = S_R1 = S_R2 = S_R3 = S_OR12 = 0
S_DT = S_WT = S_R11 = 0

# register plan (active TROWS): C0[k] = v(k), C1[k] = v(TROWS+k); A 2T..2T+3, B 2T+4..2T+6,
# ReLU zero 2T+7. AVX2/SSE41 have 16 vector regs -> TROWS <= 4 (spill beyond); AVX512 has 32 zmm
# -> TROWS up to 12 stays register-resident (uses ymm/xmm views of regs 16+, needs AVX512VL).
ACC = range(8)
A0, A1, A2, A3 = 8, 9, 10, 11
B0, B1, B2 = 12, 13, 14
ZR = 15


def _set_plan(p, isa):
    """Recompute the per-file register plan + stack layout globals. cc2 hardcodes 4 GPR column
    pointer regs and hdr sizes its scalar spill slots for 4 columns: those kinds stay at 4."""
    global TROWS, ACC, A0, A1, A2, A3, B0, B1, B2, B3, B4, B5, ZR, OB2
    global S_BASE, S_W, S_B, S_O, S_XE, S_YE, S_RX, S_Y, S_SC, S_X, S_ST, S_II
    global S_D2, S_ST1, S_ST2, S_R0, S_R1, S_R2, S_R3, S_OR12, S_DT, S_WT, S_R11
    TROWS = 4 if (p.kind == 'concat2' or p.post_op == 'hdr') else TROWS_REQ
    OB2 = False
    if isa.arch == 'arm64':
        import os
        _wide = p.kind != 'concat2' and p.post_op != 'hdr'
        # OB2: walk two output channels per ii. The A window is loaded ONCE per step and
        # serves both channels' accumulator chains: loads/vec-FMA 35/48 -> 26/60, walk body
        # 94 instrs/60 FMA (was 90/48). Measured on Apple M4 (TBUD=0.8, interleaved): every
        # plain pass beats the C++ kernel now - conv14 1.47->0.78, conv12 1.28->0.68,
        # conv4 1.28->0.70, conv1 1.12->0.92, conv2 1.18->0.96 (this is also the structure
        # Apple clang's own codegen relies on: its A regs feed 8 accumulator slots).
        # acc layout: ch0 v(k)/v(T+k), ch1 v(2T+k)/v(3T+k) (row0/row1); per-output tap
        # order unchanged -> bit-identical. DS pooling wants an even tile width -> T=4
        # there, 5 for plain walks (accs 4T + A4 + B6 + ZR <= 32 v regs).
        # cc2 joins OB2 at T=4 (j1 hardcodes 4 column-pointer GPRs): its tap displacements
        # x17/x13 are walking byte OFFSETS, so each input needs just ONE extra weight base
        # reg (j1: x15, j2: x3) instead of three.
        OB2 = p.post_op != 'hdr' and p.out_ch % 2 == 0
        if OB2:
            TROWS = 5 if (_wide and not p.downsample) else 4
        else:
            TROWS = 8 if _wide else 4  # 32 v regs: all 16 accs register-resident
    elif isa.V != 16:
        TROWS = min(TROWS, 4)  # 16 vector regs: accs 2T + A4 + B3 + ZR1 spills beyond T=4
    if OB2:
        ACC = range(4 * TROWS)
        A0, A1, A2, A3 = 4 * TROWS, 4 * TROWS + 1, 4 * TROWS + 2, 4 * TROWS + 3
        B0, B1, B2 = 4 * TROWS + 4, 4 * TROWS + 5, 4 * TROWS + 6
        B3, B4, B5 = 4 * TROWS + 7, 4 * TROWS + 8, 4 * TROWS + 9
        ZR = 4 * TROWS + 10
    else:
        ACC = range(2 * TROWS)
        A0, A1, A2, A3 = 2 * TROWS, 2 * TROWS + 1, 2 * TROWS + 2, 2 * TROWS + 3
        B0, B1, B2 = 2 * TROWS + 4, 2 * TROWS + 5, 2 * TROWS + 6
        ZR = 2 * TROWS + 7
    S_BASE = len(XMM_SAVE) * (64 if isa.V == 16 else 16)
    S_W, S_B, S_O, S_XE, S_YE, S_RX = (S_BASE + 8 * k for k in range(6))
    S_Y, S_SC, S_X, S_ST, S_II = (S_BASE + 48 + d for d in (0, 8, 40, 48, 56))
    S_D2, S_ST1, S_ST2, S_R0, S_R1, S_R2, S_R3, S_OR12 = (S_BASE + 48 + 8 * k for k in range(8))
    # hdr only: volatile ptr state live across powf/expf (rdi=data, rsi=weights walk,
    # r11=row base). Volatile in BOTH x86-64 ABIs - the prologue pushes only protect
    # them across the kernel boundary, NOT across the internal libm calls.
    S_DT, S_WT, S_R11 = (S_BASE + 112 + 8 * k for k in range(3))


import struct


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def _f32bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


# transfer::output<HDRTransfer> constants (Convolution.h), as float32 bit patterns, in use order:
# norm_scale, x0, x1, g, e, f, d, b, q=1.0f/c (compile-time folded), a
HDR_CONSTS = [_f32bits(v) for v in (3.13511896, 2.23151711e-03, 3.70974749e-01, 0.998620152, 0.192653254,
                                    6.26026094e-03, -2.94139609e-03, 1.64593172,
                                    _f32(1.0 / _f32(4.31384981e-01)), 1.41283765e+03)]


def prologue(t, p, isa):
    if p.kind == 'concat2':
        return prologue_cc2(t, p, isa)
    r, m, i = t.r, t.m, t.i
    frame = p.frame(t.abi, isa)
    L = [i('push', r(x)) for x in CALLEE_SAVED]
    L.append(i('sub', r('rsp'), str(frame)))
    for n, xr in enumerate(XMM_SAVE):  # spill callee-saved xmm6-15 (Win64 ABI); full zmm on AVX512
        if isa.V == 16:
            L.append(i('vmovups', m('rsp', disp=64 * n), f'zmm{xr}'))
        else:
            L.append(i('movups', m('rsp', disp=16 * n), f'xmm{xr}'))
    if t.abi == 'win':
        # rcx=data edx=rect_x r8d=rect_y r9d=rect_w; stack args at 0x28 (entry) + 8*8 + frame
        a = 0x28 + 64 + frame
        L += [
            i('mov', r('rdi'), r('rcx')),                    # data
            i('mov', r('eax'), r('r9d')),
            i('add', r('eax'), r('edx')),
            i('mov', m('rsp', disp=S_XE), r('rax')),         # x_end
            i('mov', r('eax'), r('edx')),
            i('mov', m('rsp', disp=S_RX), r('rax')),         # rect_x
            i('mov', r('r10d'), m('rsp', disp=a)),           # rect_h
            i('mov', r('eax'), r('r8d')),
            i('add', r('eax'), r('r10d')),
            i('mov', m('rsp', disp=S_YE), r('rax')),         # y_end
            i('mov', r('r9d'), m('rsp', disp=a + 0x08)),     # stride
            i('mov', r('rsi'), m('rsp', disp=a + 0x10)),     # weights
            i('mov', m('rsp', disp=S_W), r('rsi')),
            i('mov', r('rbp'), m('rsp', disp=a + 0x18)),     # biases
            i('mov', m('rsp', disp=S_B), r('rbp')),
            i('mov', r('rax'), m('rsp', disp=a + 0x20)),     # output
            i('mov', m('rsp', disp=S_O), r('rax')),
            i('mov', r('ecx'), m('rsp', disp=a + 0x28)),     # output_stride (32-bit arg: slot upper bits undefined)
            i('imul', r('r12'), r('rcx'), str(p.PIX)),       # out row stride B
            i('imul', r('r15'), r('r9'), str(p.COL)),        # in row stride B
            i('mov', r('eax'), r('r8d')),                    # y = rect_y
        ]
    else:
        # rdi=data rsi=rect_x rdx=rect_y rcx=rect_w r8=rect_h r9=stride;
        # stack args at 0x08 (entry) + 8*8 + frame
        a = 0x08 + 64 + frame
        L += [
            i('mov', r('eax'), r('ecx')),
            i('add', r('eax'), r('esi')),
            i('mov', m('rsp', disp=S_XE), r('rax')),         # x_end
            i('mov', r('eax'), r('esi')),
            i('mov', m('rsp', disp=S_RX), r('rax')),         # rect_x
            i('mov', r('eax'), r('edx')),
            i('add', r('eax'), r('r8d')),
            i('mov', m('rsp', disp=S_YE), r('rax')),         # y_end
            i('mov', r('rsi'), m('rsp', disp=a)),            # weights
            i('mov', m('rsp', disp=S_W), r('rsi')),
            i('mov', r('rbp'), m('rsp', disp=a + 0x08)),     # biases
            i('mov', m('rsp', disp=S_B), r('rbp')),
            i('mov', r('rax'), m('rsp', disp=a + 0x10)),     # output
            i('mov', m('rsp', disp=S_O), r('rax')),
            i('mov', r('ecx'), m('rsp', disp=a + 0x18)),     # output_stride (32-bit arg: slot upper bits undefined)
            i('imul', r('r12'), r('rcx'), str(p.PIX)),
            i('imul', r('r15'), r('r9'), str(p.COL)),
            i('mov', r('eax'), r('edx')),                    # y = rect_y
        ]
    if isa.fma and isa.V == 16 and ZR >= 16:
        L.append(i('vpxord', t.r(isa.v(ZR)), t.r(isa.v(ZR)), t.r(isa.v(ZR))))  # ReLU zero (EVEX-DQ)
    else:
        L.append(i('xorps' if not isa.vex else 'vpxor', t.r(isa.x(ZR)), t.r(isa.x(ZR)),
                   *(t.r(isa.x(ZR)),) if isa.vex else ()))  # ReLU zero
    return L


def prologue_cc2(t, p, isa):
    """2-input concat kernel prologue: (data1, data2, rect_x, rect_y, rect_w, rect_h,
    stride1, stride2, weights, biases, output, output_stride)."""
    r, m, i = t.r, t.m, t.i
    frame = p.frame(t.abi, isa)
    L = [i('push', r(x)) for x in CALLEE_SAVED]
    L.append(i('sub', r('rsp'), str(frame)))
    for n, xr in enumerate(XMM_SAVE):
        if isa.V == 16:
            L.append(i('vmovups', m('rsp', disp=64 * n), f'zmm{xr}'))
        else:
            L.append(i('movups', m('rsp', disp=16 * n), f'xmm{xr}'))
    if t.abi == 'win':
        # rcx=data1 rdx=data2 r8d=rect_x r9d=rect_y; stack args at 0x28 (entry) + 8*8 + frame
        a = 0x28 + 64 + frame
        L += [
            i('mov', r('rdi'), r('rcx')),                    # data1
            i('mov', m('rsp', disp=S_D2), r('rdx')),         # data2
            i('mov', r('r10d'), m('rsp', disp=a)),           # rect_w
            i('mov', r('eax'), r('r8d')),
            i('add', r('eax'), r('r10d')),
            i('mov', m('rsp', disp=S_XE), r('rax')),         # x_end
            i('mov', r('eax'), r('r8d')),
            i('mov', m('rsp', disp=S_RX), r('rax')),         # rect_x
            i('mov', r('r11d'), r('r9d')),                   # save rect_y (r9 reused below)
            i('mov', r('r10d'), m('rsp', disp=a + 0x08)),    # rect_h
            i('mov', r('eax'), r('r9d')),
            i('add', r('eax'), r('r10d')),
            i('mov', m('rsp', disp=S_YE), r('rax')),         # y_end
            i('mov', r('r9d'), m('rsp', disp=a + 0x10)),     # stride1
            i('mov', r('rax'), r('r9')),
            i('imul', r('rax'), r('rax'), str(p.COL1)),
            i('mov', m('rsp', disp=S_ST1), r('rax')),        # in1 row stride B
            i('mov', r('r9d'), m('rsp', disp=a + 0x18)),     # stride2
            i('mov', r('rax'), r('r9')),
            i('imul', r('rax'), r('rax'), str(p.COL2)),
            i('mov', m('rsp', disp=S_ST2), r('rax')),        # in2 row stride B
            i('mov', r('rsi'), m('rsp', disp=a + 0x20)),     # weights
            i('mov', m('rsp', disp=S_W), r('rsi')),
            i('mov', r('rbp'), m('rsp', disp=a + 0x28)),     # biases
            i('mov', m('rsp', disp=S_B), r('rbp')),
            i('mov', r('rax'), m('rsp', disp=a + 0x30)),     # output
            i('mov', m('rsp', disp=S_O), r('rax')),
            i('mov', r('ecx'), m('rsp', disp=a + 0x38)),     # output_stride (32-bit arg)
            i('imul', r('r12'), r('rcx'), str(p.PIX)),
            i('mov', m('rsp', disp=S_OR12), r('r12')),       # out row stride B (r12 reused by j1)
            i('mov', r('eax'), r('r11d')),                   # y = rect_y
        ]
    else:
        # rdi=data1 rsi=data2 edx=rect_x ecx=rect_y r8d=rect_w r9d=rect_h;
        # stack args at 0x08 (entry) + 8*8 + frame
        a = 0x08 + 64 + frame
        L += [
            i('mov', m('rsp', disp=S_D2), r('rsi')),         # data2
            i('mov', r('eax'), r('r8d')),
            i('add', r('eax'), r('edx')),
            i('mov', m('rsp', disp=S_XE), r('rax')),         # x_end
            i('mov', r('eax'), r('edx')),
            i('mov', m('rsp', disp=S_RX), r('rax')),         # rect_x
            i('mov', r('eax'), r('ecx')),
            i('add', r('eax'), r('r9d')),
            i('mov', m('rsp', disp=S_YE), r('rax')),         # y_end
            i('mov', r('r9d'), m('rsp', disp=a)),            # stride1 (32-bit load: kills slot garbage)
            i('mov', r('rax'), r('r9')),
            i('imul', r('rax'), r('rax'), str(p.COL1)),
            i('mov', m('rsp', disp=S_ST1), r('rax')),
            i('mov', r('r8d'), m('rsp', disp=a + 0x08)),     # stride2
            i('mov', r('rax'), r('r8')),
            i('imul', r('rax'), r('rax'), str(p.COL2)),
            i('mov', m('rsp', disp=S_ST2), r('rax')),
            i('mov', r('rsi'), m('rsp', disp=a + 0x10)),     # weights
            i('mov', m('rsp', disp=S_W), r('rsi')),
            i('mov', r('rbp'), m('rsp', disp=a + 0x18)),     # biases
            i('mov', m('rsp', disp=S_B), r('rbp')),
            i('mov', r('rax'), m('rsp', disp=a + 0x20)),     # output
            i('mov', m('rsp', disp=S_O), r('rax')),
            i('mov', r('eax'), m('rsp', disp=a + 0x28)),     # output_stride (32-bit arg; NOT ecx - ecx holds rect_y)
            i('imul', r('r12'), r('rax'), str(p.PIX)),
            i('mov', m('rsp', disp=S_OR12), r('r12')),
            i('mov', r('eax'), r('ecx')),                    # y = rect_y (ecx survives: ostr went into eax)
        ]
    if isa.fma and isa.V == 16 and ZR >= 16:
        L.append(i('vpxord', t.r(isa.v(ZR)), t.r(isa.v(ZR)), t.r(isa.v(ZR))))  # ReLU zero (EVEX-DQ)
    else:
        L.append(i('xorps' if not isa.vex else 'vpxor', t.r(isa.x(ZR)), t.r(isa.x(ZR)),
                   *(t.r(isa.x(ZR)),) if isa.vex else ()))  # ReLU zero
    return L


def kblock(t, p, isa, k, col=None):
    """One tile column k: 4 A loads + the 6 accumulation chains, bit-identical to the
    C++ fmadd order. SSE4.1 lowers fmadd(a,b,c) to mulps+addps, reusing A regs as
    scratch once their last use has passed."""
    i, m = t.i, t.m
    v = isa.v
    if col is None:
        col = p.COL
    L = [
        i('movups' if not isa.vex else 'vmovups', v(A0), m('r13', 'rdx', disp=col * k)),  # A0 (row y-1)
        i('movups' if not isa.vex else 'vmovups', v(A1), m('r11', 'rdx', disp=col * k)),  # A1 (row y)
        i('movups' if not isa.vex else 'vmovups', v(A2), m('r14', 'rdx', disp=col * k)),  # A2 (row y+1)
        i('movups' if not isa.vex else 'vmovups', v(A3), m('r10', 'rdx', disp=col * k)),  # A3 (row y+2)
    ]
    if isa.fma:
        L += [
            i('vfmadd231ps', v(k), v(A0), v(B0)),
            i('vfmadd231ps', v(k), v(A1), v(B1)),
            i('vfmadd231ps', v(k), v(A2), v(B2)),
            i('vfmadd231ps', v(TROWS + k), v(A1), v(B0)),
            i('vfmadd231ps', v(TROWS + k), v(A2), v(B1)),
            i('vfmadd231ps', v(TROWS + k), v(A3), v(B2)),
        ]
    else:
        # mulps dst,src: dst = dst*src; the product lands in a scratch that is an A reg
        # past its last use (A0 after chain 1; A1/A2 need a movaps copy first since they
        # feed both rows). Values are identical to the C++ mul-then-add lowering.
        # VEX-prefixed for VEX ISAs (avx1 ymm regs need vmulps/vaddps; sse41 stays legacy SSE);
        # masm requires the 3-operand VEX arithmetic form, so dst is also the first source.
        vp = 'v' if isa.vex else ''

        def ma(op, dst, src):  # mul/add accumulate: 2-op legacy, 3-op VEX
            return i(vp + op, dst, dst, src) if isa.vex else i(op, dst, src)
        L += [
            ma('mulps', v(A0), v(B0)), ma('addps', v(k), v(A0)),           # C0 += A0*B0 (A0 dead)
            i(vp + 'movaps', v(A0), v(A1)), ma('mulps', v(A0), v(B1)), ma('addps', v(k), v(A0)),
            i(vp + 'movaps', v(A0), v(A2)), ma('mulps', v(A0), v(B2)), ma('addps', v(k), v(A0)),
            ma('mulps', v(A1), v(B0)), ma('addps', v(TROWS + k), v(A1)),   # C1 += A1*B0 (A1 dead)
            ma('mulps', v(A2), v(B1)), ma('addps', v(TROWS + k), v(A2)),   # C1 += A2*B1 (A2 dead)
            ma('mulps', v(A3), v(B2)), ma('addps', v(TROWS + k), v(A3)),   # C1 += A3*B2 (A3 dead)
        ]
    return L


def smax(t, isa, a, b):
    # Scalar ReLU max: keep legacy 2-operand maxss (bit-identical to the C++ std::max lowering)
    # unless the ReLU zero reg is above 15 - SSE encodings can't reach it. The 3-operand vmaxss
    # form has identical NaN/less-than semantics.
    if isa.V == 16 and ZR >= 16:
        return t.i('vmaxss', t.r(isa.x(a)), t.r(isa.x(a)), t.r(isa.x(b)))
    return t.i('maxss', t.r(isa.x(a)), t.r(isa.x(b)))


def hsum(t, isa, n):
    """Mirror fvec<V>::hsum exactly (result consumed from lane 0)."""
    i = t.i
    if isa.V == 16:
        # MSVC _mm512_reduce_add_ps lowering (simd_avx512.h hsum): pairwise fold
        # 16->8->4 lanes, then psrldq shifts + addss; result lands in x(n) lane 0.
        # Tree ((L0+L8)+(L4+L12)) + ((L2+L10)+(L6+L14)) etc., scratch = A0 (A regs dead).
        # vextractf32x4 (not vextractf128): identical result, but ml64 rejects the EVEX-VL
        # vextractf128 form with a high (16+) xmm destination.
        return [
            i('vextractf32x8', isa.y(A0), isa.v(n), '1'),
            i('vaddps', isa.y(n), isa.y(n), isa.y(A0)),
            i('vextractf32x4', isa.x(A0), isa.y(n), '1'),
            i('vaddps', isa.x(n), isa.x(n), isa.x(A0)),
            i('vpsrldq', isa.x(A0), isa.x(n), '8'),
            i('vaddps', isa.x(n), isa.x(n), isa.x(A0)),
            i('vpsrldq', isa.x(A0), isa.x(n), '4'),
            i('vaddss', isa.x(n), isa.x(n), isa.x(A0)),
        ]
    if isa.V == 8:
        return [
            i('vhaddps', isa.v(n), isa.v(n), isa.v(n)),
            i('vhaddps', isa.v(n), isa.v(n), isa.v(n)),
            i('vperm2f128', isa.v(A0), isa.v(n), isa.v(n), '1'),
            i('vaddps', isa.v(n), isa.v(A0), isa.v(n)),
        ]
    return [i('haddps', isa.v(n), isa.v(n)), i('haddps', isa.v(n), isa.v(n))]


def body(t, p, isa):
    if p.kind == 'concat2':
        return body_cc2(t, p, isa)
    r, m, i = t.r, t.m, t.i
    L = []
    add = L.append
    js = p.j_step(isa)

    add('y_loop:')
    add(i('cmp', r('rax'), m('rsp', disp=S_YE)))
    add(i('jge', 'y_done'))
    add(i('mov', r('rcx'), m('rsp', disp=S_RX)))             # x = rect_x

    add('x_loop:')
    add(i('cmp', r('rcx'), m('rsp', disp=S_XE)))
    add(i('jge', 'x_done'))
    # tap row bases: r13=(y-1) r11=y r14=(y+1) r10=(y+2), all + COL*(x-1)
    add(i('mov', r('rdx'), r('rax')))
    add(i('imul', r('rdx'), r('r9')))                        # y * stride
    add(i('lea', r('rdx'), m('rdx', 'rcx', disp=-1)))        # + x - 1
    add(i('imul', r('rdx'), r('rdx'), str(p.COL)))
    add(i('lea', r('r11'), m('rdi', 'rdx')))                 # A1 base (row y)
    add(i('mov', r('r13'), r('r11')))
    add(i('sub', r('r13'), r('r15')))                        # A0 base (row y-1)
    add(i('mov', r('r14'), r('r11')))
    add(i('add', r('r14'), r('r15')))                        # A2 base (row y+1)
    add(i('mov', r('r10'), r('r14')))
    add(i('add', r('r10'), r('r15')))                        # A3 base (row y+2)
    # rbx = output_g + out_row_stride*(y>>ds) + PIX*(x>>ds)  (ds = 1 for downsample passes)
    add(i('mov', r('rbx'), r('rax')))
    if p.downsample:
        add(i('shr', r('rbx'), '1'))
    add(i('imul', r('rbx'), r('r12')))
    add(i('add', r('rbx'), m('rsp', disp=S_O)))
    if p.downsample:
        add(i('mov', r('rdx'), r('rcx')))
        add(i('shr', r('rdx'), '1'))
        add(i('imul', r('rdx'), r('rdx'), str(p.PIX)))
    else:
        add(i('imul', r('rdx'), r('rcx'), str(p.PIX)))
    add(i('add', r('rbx'), r('rdx')))

    add(i('xor', r('r8d'), r('r8d')))                        # r8 = 4*ii
    add(i('mov', r('rsi'), m('rsp', disp=S_W)))              # weights ptr (advances W_ROW/ii)
    add(i('mov', r('rbp'), m('rsp', disp=S_B)))              # biases ptr (advances 4/ii)

    add('ii_loop:')
    add(i('cmp', r('r8'), str(p.II_END)))
    add(i('jge', 'ii_done'))
    # acc = set<0>(biases[ii]): lane 0 = bias, other lanes = 0
    add(i('movss' if not isa.vex else 'vmovss', t.r(isa.x(0)), t.dword(m('rbp'))))
    for k in list(ACC)[1:]:
        add(i('movaps' if not isa.vex else 'vmovaps', isa.v(k), isa.v(0)))
    # C1[k] = C0[k] for k < TROWS; the extra k (up to A regs, targets v(T..T+3+TROWS-1)) mirror
    # the original ACC-wide broadcast - they write A regs (reloaded each j step), never B/ZR.
    for k in range(min(2 * TROWS, TROWS + 4)):
        add(i('movaps' if not isa.vex else 'vmovaps', isa.v(TROWS + k), isa.v(k)))

    add(i('xor', r('edx'), r('edx')))                        # rdx = 4*j byte offset
    add('j_loop:')
    add(i('movaps' if not isa.vex else 'vmovaps', isa.v(B0), m('rsi', 'rdx')))            # B0
    add(i('movaps' if not isa.vex else 'vmovaps', isa.v(B1), m('rsi', 'rdx', disp=p.TAP)))      # B1
    add(i('movaps' if not isa.vex else 'vmovaps', isa.v(B2), m('rsi', 'rdx', disp=2 * p.TAP)))  # B2
    for k in range(TROWS):
        L += kblock(t, p, isa, k)
    add(i('add', r('rdx'), str(js)))
    add(i('cmp', r('rdx'), str(p.J_END)))
    add(i('jl', 'j_loop'))

    if p.downsample:
        # downsample epilogue: hsum + ReLU all 8, then 2x2 max-pool and store at half
        # coords (xx < x_end guard only; row y+1 always participates in the pool).
        # std::max(a,b) == maxss(b,a): NaN returns the first argument, unlike maxss(a,b).
        for n in list(range(TROWS, 2 * TROWS)) + list(range(TROWS)):
            L += hsum(t, isa, n)
            add(smax(t, isa, n, ZR))
        for k in range(0, TROWS, 2):
            add(i('lea', r('rdx'), m('rcx', disp=k)))
            add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
            add(i('jge', t.lbl(f'p{k}')))
            add(smax(t, isa, k + 1, k))                   # max(C0[k], C0[k+1])
            add(smax(t, isa, TROWS + k + 1, TROWS + k))  # max(C1[k], C1[k+1])
            add(smax(t, isa, TROWS + k + 1, k + 1))      # max of the two pairs
            add(smax(t, isa, TROWS + k + 1, ZR))         # std::max(0.0f, m)
            add(i('movss' if not isa.vex else 'vmovss',
                  t.dword(m('rbx', 'r8', disp=p.PIX * (k // 2))), t.r(isa.x(TROWS + k + 1))))
            add(f'{t.lbl(f"p{k}")}:')
        add(i('add', r('rsi'), str(p.W_ROW)))
        add(i('add', r('rbp'), '4'))
        add(i('add', r('r8'), '4'))
        add(i('jmp', 'ii_loop'))
        add('ii_done:')
        add(i('add', r('rcx'), str(TROWS)))
        add(i('jmp', 'x_loop'))
        add('x_done:')
        add(i('add', r('rax'), '2'))
        add(i('jmp', 'y_loop'))
        add('y_done:')
        return L

    if p.post_op == 'hdr':
        # hdr epilogue: spill all 8 hsum+ReLU scalars to slots first (powf/expf clobber xmm0-5
        # and every volatile GPR), then run the transfer chain per store site. Loop state is
        # saved to slots; const base -> r14, channel offset -> r13 (callee-saved).
        # rdi/rsi/r11 ALSO must be spilled: they are volatile on BOTH x86-64 ABIs (the prologue
        # pushes only preserve them across the kernel boundary) and stay live across the calls.
        add(i('lea', r('rdx'), m('rax', disp=1)))
        add(i('cmp', r('rdx'), m('rsp', disp=S_YE)))
        add(i('jge', 'row0_spill'))
        for k in range(TROWS):
            L += hsum(t, isa, TROWS + k)
            add(smax(t, isa, TROWS + k, ZR))
            add(i('movss' if not isa.vex else 'vmovss',
                  t.dword(m('rsp', disp=S_SC + 4 * (TROWS + k))), t.r(isa.x(TROWS + k))))
        add('row0_spill:')
        for k in range(TROWS):
            L += hsum(t, isa, k)
            add(smax(t, isa, k, ZR))
            add(i('movss' if not isa.vex else 'vmovss',
                  t.dword(m('rsp', disp=S_SC + 4 * k)), t.r(isa.x(k))))
        add(i('mov', m('rsp', disp=S_Y), r('rax')))
        add(i('mov', m('rsp', disp=S_X), r('rcx')))
        add(i('mov', m('rsp', disp=S_ST), r('r9')))
        add(i('mov', m('rsp', disp=S_II), r('r8')))
        add(i('mov', m('rsp', disp=S_DT), r('rdi')))
        add(i('mov', m('rsp', disp=S_WT), r('rsi')))
        add(i('mov', m('rsp', disp=S_R11), r('r11')))
        add(t.lea_sym('r14', p.tag + '_hdr_c'))
        add(i('mov', r('r13'), r('r8')))
        # store-phase guards read x/y from slots: powf/expf clobber rax/rcx (volatile on win64)
        add(i('mov', r('rdx'), m('rsp', disp=S_Y)))
        add(i('add', r('rdx'), '1'))
        add(i('cmp', r('rdx'), m('rsp', disp=S_YE)))
        add(i('jge', 'row0_store'))
        add(i('add', r('rbx'), r('r12')))                        # shift to row y+1
        for k in range(TROWS):
            add(i('mov', r('rdx'), m('rsp', disp=S_X)))
            add(i('add', r('rdx'), str(k)))
            add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
            add(i('jge', t.lbl(f'c1_{k}')))
            L += hdr_site(t, p, isa, TROWS + k, m('rbx', 'r13', disp=p.PIX * k), f'c1_{k}')
            add(f'{t.lbl(f"c1_{k}")}:')
        add(i('sub', r('rbx'), r('r12')))                        # back to row y
        add('row0_store:')
        for k in range(TROWS):
            add(i('mov', r('rdx'), m('rsp', disp=S_X)))
            add(i('add', r('rdx'), str(k)))
            add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
            add(i('jge', t.lbl(f'c0_{k}')))
            L += hdr_site(t, p, isa, k, m('rbx', 'r13', disp=p.PIX * k), f'c0_{k}')
            add(f'{t.lbl(f"c0_{k}")}:')
        add(i('mov', r('rax'), m('rsp', disp=S_Y)))              # restore loop state
        add(i('mov', r('rcx'), m('rsp', disp=S_X)))
        add(i('mov', r('r9'), m('rsp', disp=S_ST)))
        add(i('mov', r('r8'), m('rsp', disp=S_II)))
        add(i('mov', r('rdi'), m('rsp', disp=S_DT)))             # volatile ptrs (both ABIs)
        add(i('mov', r('rsi'), m('rsp', disp=S_WT)))
        add(i('mov', r('r11'), m('rsp', disp=S_R11)))
        if isa.fma and isa.V == 16 and ZR >= 16:
            add(i('vpxord', t.r(isa.v(ZR)), t.r(isa.v(ZR)), t.r(isa.v(ZR))))
        else:
            add(i('xorps' if not isa.vex else 'vpxor', t.r(isa.x(ZR)), t.r(isa.x(ZR)),
                  *(t.r(isa.x(ZR)),) if isa.vex else ()))        # ReLU zero: xmm6-15 are
        # r13/r14 doubled as const base / channel offset: restore the A tap row bases. r11 is
        # VOLATILE on both x86-64 ABIs (survived only because we spilled it), r15 is callee-saved.
        add(i('mov', r('r13'), r('r11')))
        add(i('sub', r('r13'), r('r15')))
        add(i('mov', r('r14'), r('r11')))
        add(i('add', r('r14'), r('r15')))
        add(i('mov', r('r10'), r('r14')))
        add(i('add', r('r10'), r('r15')))
        add(i('add', r('rsi'), str(p.W_ROW)))
        add(i('add', r('rbp'), '4'))
        add(i('add', r('r8'), '4'))
        add(i('jmp', 'ii_loop'))
        add('ii_done:')
        add(i('add', r('rcx'), str(TROWS)))
        add(i('jmp', 'x_loop'))
        add('x_done:')
        add(i('add', r('rax'), '2'))
        add(i('jmp', 'y_loop'))
        add('y_done:')
        return L

    # epilogue: hsum + ReLU + clipped scalar stores; A0 reg = hsum temp (A regs dead)
    add(i('lea', r('rdx'), m('rax', disp=1)))
    add(i('cmp', r('rdx'), m('rsp', disp=S_YE)))
    add(i('jge', 'row0_only'))
    add(i('add', r('rbx'), r('r12')))                        # shift to row y+1
    for k in range(TROWS):
        add(i('lea', r('rdx'), m('rcx', disp=k)))
        add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
        add(i('jge', t.lbl(f'c1_{k}')))
        L += hsum(t, isa, TROWS + k)
        add(smax(t, isa, TROWS + k, ZR))
        add(i('movss' if not isa.vex else 'vmovss',
              t.dword(m('rbx', 'r8', disp=p.PIX * k)), t.r(isa.x(TROWS + k))))
        add(f'{t.lbl(f"c1_{k}")}:')
    add(i('sub', r('rbx'), r('r12')))                        # back to row y
    add('row0_only:')
    for k in range(TROWS):
        add(i('lea', r('rdx'), m('rcx', disp=k)))
        add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
        add(i('jge', t.lbl(f'c0_{k}')))
        L += hsum(t, isa, k)
        add(smax(t, isa, k, ZR))
        add(i('movss' if not isa.vex else 'vmovss',
              t.dword(m('rbx', 'r8', disp=p.PIX * k)), t.r(isa.x(k))))
        add(f'{t.lbl(f"c0_{k}")}:')

    add(i('add', r('rsi'), str(p.W_ROW)))
    add(i('add', r('rbp'), '4'))
    add(i('add', r('r8'), '4'))
    add(i('jmp', 'ii_loop'))
    add('ii_done:')
    add(i('add', r('rcx'), str(TROWS)))
    add(i('jmp', 'x_loop'))
    add('x_done:')
    add(i('add', r('rax'), '2'))
    add(i('jmp', 'y_loop'))
    add('y_done:')
    return L


def body_cc2(t, p, isa):
    """2-input concat kernel body: j1 walks input1 with Upsample addressing (3 tap blocks,
    C[k] col ptrs in r10/r11/r13/r12, source row offset via r9 from the R slots), j2 walks
    input2 plain (kblock over COL2). Mirrors the C++ chain order per accumulator exactly."""
    r, m, i = t.r, t.m, t.i
    v = isa.v
    L = []
    add = L.append
    js = p.j_step(isa)
    ld = 'movups' if not isa.vex else 'vmovups'
    cp = 'movaps' if not isa.vex else 'vmovaps'
    st = 'movss' if not isa.vex else 'vmovss'
    vp = 'v' if isa.vex else ''  # VEX prefix for the non-FMA mul/add chains (avx1 ymm)

    def ma(op, dst, src):  # mul/add accumulate: 2-op legacy, 3-op VEX (masm needs 3-op VEX)
        return i('v' + op, dst, dst, src) if isa.vex else i(op, dst, src)
    CREG = (10, 11, 13, 12)  # C[k] col ptr regs for k = 0..3 (r12 = out row stride lives in S_OR12)

    add('y_loop:')
    add(i('cmp', r('rax'), m('rsp', disp=S_YE)))
    add(i('jge', 'y_done'))
    # Upsample source row offsets row_r * (stride1*COL1); row_0 = y==0 ? -1 : (y-1)/2
    add(i('mov', r('r15'), m('rsp', disp=S_ST1)))
    add(i('mov', r('r9'), r('rax')))
    add(i('shr', r('r9'), '1'))
    add(i('imul', r('r9'), r('r15')))
    add(i('mov', m('rsp', disp=S_R1), r('r9')))
    add(i('mov', r('r9'), r('rax')))
    add(i('add', r('r9'), '1'))
    add(i('shr', r('r9'), '1'))
    add(i('imul', r('r9'), r('r15')))
    add(i('mov', m('rsp', disp=S_R2), r('r9')))
    add(i('mov', r('r9'), r('rax')))
    add(i('add', r('r9'), '2'))
    add(i('shr', r('r9'), '1'))
    add(i('imul', r('r9'), r('r15')))
    add(i('mov', m('rsp', disp=S_R3), r('r9')))
    add(i('cmp', r('rax'), '1'))
    add(i('jb', t.lbl('r0neg')))
    add(i('mov', r('r9'), r('rax')))
    add(i('sub', r('r9'), '1'))
    add(i('shr', r('r9'), '1'))
    add(i('imul', r('r9'), r('r15')))
    add(i('jmp', t.lbl('r0')))
    add(f'{t.lbl("r0neg")}:')
    add(i('mov', r('r9'), r('r15')))
    add(i('neg', r('r9')))
    add(f'{t.lbl("r0")}:')
    add(i('mov', m('rsp', disp=S_R0), r('r9')))
    add(i('mov', r('rcx'), m('rsp', disp=S_RX)))

    add('x_loop:')
    add(i('cmp', r('rcx'), m('rsp', disp=S_XE)))
    add(i('jge', 'x_done'))
    add(i('mov', r('r12'), m('rsp', disp=S_OR12)))
    add(i('mov', r('rbx'), r('rax')))
    add(i('imul', r('rbx'), r('r12')))
    add(i('add', r('rbx'), m('rsp', disp=S_O)))
    add(i('imul', r('rdx'), r('rcx'), str(p.PIX)))
    add(i('add', r('rbx'), r('rdx')))

    add(i('xor', r('r8d'), r('r8d')))
    add(i('mov', r('rsi'), m('rsp', disp=S_W)))
    add(i('mov', r('rbp'), m('rsp', disp=S_B)))

    add('ii_loop:')
    add(i('cmp', r('r8'), str(p.II_END)))
    add(i('jge', 'ii_done'))
    add(i(st, t.r(isa.x(0)), t.dword(m('rbp'))))
    for k in list(ACC)[1:]:
        add(i(cp, v(k), v(0)))
    for k in ACC:
        add(i(cp, v(TROWS + k), v(k)))

    # ---- j1: input1, Upsample. tap = x_off is constant per block; C[k] = data1 + x_final(k)*COL1
    add(i('xor', r('edx'), r('edx')))
    for tap in range(3):
        add(i('mov', r('r9d'), r('ecx')))
        add(i('add', r('r9d'), str(tap + 1)))
        for k, reg in enumerate(CREG):
            if k:
                add(i('mov', r(f'r{reg}'), r('r9')))
                add(i('add', r(f'r{reg}'), str(k)))
            else:
                add(i('mov', r(f'r{reg}'), r('r9')))
            add(i('shr', r(f'r{reg}'), '1'))
            add(i('sub', r(f'r{reg}'), '1'))
            add(i('imul', r(f'r{reg}'), r(f'r{reg}'), str(p.COL1)))
            add(i('add', r(f'r{reg}'), r('rdi')))
        add(f'j1w{tap}:')
        add(i(cp, v(B0), m('rsi', 'rdx')))
        add(i(cp, v(B1), m('rsi', 'rdx', disp=p.TAP1)))
        add(i(cp, v(B2), m('rsi', 'rdx', disp=2 * p.TAP1)))
        # row groups: (R slot, B reg feeding C0 or None, B reg feeding C1 or None)
        for slot, c0b, c1b in (('S_R0', B0, None), ('S_R1', B1, B0), ('S_R2', B2, B1), ('S_R3', None, B2)):
            add(i('mov', r('r9'), m('rsp', disp=globals()[slot])))
            for k, reg in enumerate(CREG):
                add(i(ld, v(A0 + k), m(f'r{reg}', 'r9')))
            for k in range(TROWS):
                if c0b is not None and c1b is not None:
                    if isa.fma:
                        add(i('vfmadd231ps', v(k), v(A0 + k), v(c0b)))
                        add(i('vfmadd231ps', v(TROWS + k), v(A0 + k), v(c1b)))
                    else:
                        # A feeds both rows: copy through the ZR reg (re-zeroed before the epilogue)
                        add(i(cp, v(ZR), v(A0 + k)))
                        add(ma('mulps', v(ZR), v(c0b)))
                        add(ma('addps', v(k), v(ZR)))
                        add(ma('mulps', v(A0 + k), v(c1b)))
                        add(ma('addps', v(TROWS + k), v(A0 + k)))
                else:
                    b = c0b if c0b is not None else c1b
                    acc = k if c0b is not None else TROWS + k
                    if isa.fma:
                        add(i('vfmadd231ps', v(acc), v(A0 + k), v(b)))
                    else:
                        add(ma('mulps', v(A0 + k), v(b)))
                        add(ma('addps', v(acc), v(A0 + k)))
        add(i('add', r('rdx'), str(js)))
        for reg in CREG:
            add(i('add', r(f'r{reg}'), str(js)))
        add(i('cmp', r('rdx'), str((tap + 1) * p.in_ch * 4)))
        add(i('jl', f'j1w{tap}'))

    # ---- j2: input2, plain tap window (kblock over COL2) ----
    add(i('mov', r('r11'), m('rsp', disp=S_D2)))
    add(i('mov', r('rdx'), r('rax')))
    add(i('imul', r('rdx'), m('rsp', disp=S_ST2)))
    add(i('mov', r('r9'), r('rcx')))
    add(i('sub', r('r9'), '1'))
    add(i('imul', r('r9'), r('r9'), str(p.COL2)))
    add(i('add', r('rdx'), r('r9')))
    add(i('add', r('r11'), r('rdx')))                        # A1 base (row y)
    add(i('mov', r('r13'), r('r11')))
    add(i('sub', r('r13'), m('rsp', disp=S_ST2)))            # A0 base (row y-1)
    add(i('mov', r('r14'), r('r11')))
    add(i('add', r('r14'), m('rsp', disp=S_ST2)))            # A2 base (row y+1)
    add(i('mov', r('r10'), r('r14')))
    add(i('add', r('r10'), m('rsp', disp=S_ST2)))            # A3 base (row y+2)
    add(i('xor', r('edx'), r('edx')))
    add('j2_loop:')
    add(i(cp, v(B0), m('rsi', 'rdx', disp=p.W_OFF)))
    add(i(cp, v(B1), m('rsi', 'rdx', disp=p.W_OFF + p.TAP2)))
    add(i(cp, v(B2), m('rsi', 'rdx', disp=p.W_OFF + 2 * p.TAP2)))
    for k in range(TROWS):
        L += kblock(t, p, isa, k, col=p.COL2)
    add(i('add', r('rdx'), str(js)))
    add(i('cmp', r('rdx'), str(p.J2_END)))
    add(i('jl', 'j2_loop'))

    # ---- epilogue: same as the plain kernel (hsum + ReLU + clipped scalar stores) ----
    if not isa.fma:
        add(i('xorps', t.r(isa.x(ZR)), t.r(isa.x(ZR))))      # ZR was j1's mul scratch
    add(i('mov', r('r12'), m('rsp', disp=S_OR12)))
    add(i('lea', r('rdx'), m('rax', disp=1)))
    add(i('cmp', r('rdx'), m('rsp', disp=S_YE)))
    add(i('jge', 'row0_only'))
    add(i('add', r('rbx'), r('r12')))                        # shift to row y+1
    for k in range(TROWS):
        add(i('lea', r('rdx'), m('rcx', disp=k)))
        add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
        add(i('jge', t.lbl(f'c1_{k}')))
        L += hsum(t, isa, TROWS + k)
        add(smax(t, isa, TROWS + k, ZR))
        add(i(st, t.dword(m('rbx', 'r8', disp=p.PIX * k)), t.r(isa.x(TROWS + k))))
        add(f'{t.lbl(f"c1_{k}")}:')
    add(i('sub', r('rbx'), r('r12')))                        # back to row y
    add('row0_only:')
    for k in range(TROWS):
        add(i('lea', r('rdx'), m('rcx', disp=k)))
        add(i('cmp', r('rdx'), m('rsp', disp=S_XE)))
        add(i('jge', t.lbl(f'c0_{k}')))
        L += hsum(t, isa, k)
        add(smax(t, isa, k, ZR))
        add(i(st, t.dword(m('rbx', 'r8', disp=p.PIX * k)), t.r(isa.x(k))))
        add(f'{t.lbl(f"c0_{k}")}:')

    add(i('add', r('rsi'), str(p.W_ROW)))
    add(i('add', r('rbp'), '4'))
    add(i('add', r('r8'), '4'))
    add(i('jmp', 'ii_loop'))
    add('ii_done:')
    add(i('add', r('rcx'), str(TROWS)))
    add(i('jmp', 'x_loop'))
    add('x_done:')
    add(i('add', r('rax'), '2'))
    add(i('jmp', 'y_loop'))
    add('y_done:')
    return L


def epilogue(t, p, isa):
    r, m, i = t.r, t.m, t.i
    if isa.V == 16:
        restores = [i('vmovups', f'zmm{xr}', m('rsp', disp=64 * n)) for n, xr in enumerate(XMM_SAVE)]
    else:
        restores = [i('movups', f'xmm{xr}', m('rsp', disp=16 * n)) for n, xr in enumerate(XMM_SAVE)]
    pops = [i('pop', r(x)) for x in reversed(CALLEE_SAVED)]
    return restores + [i('add', r('rsp'), str(p.frame(t.abi, isa)))] + pops + [i('ret')]


def call_f(t, p, name):
    # powf/expf: the callee entry needs rsp == 8 (mod 16), i.e. rsp == 0 at the call.
    # hdr/win prologue leaves rsp == 8, so sub/add 40 (32 B shadow + 8 align) lands at
    # rsp == 0 with the shadow below the frame (clear of the xmm6-15 save area).
    # hdr/sysv prologue leaves rsp == 0 (FRAME=280), bare call.
    if t.abi == 'win':
        return [t.i('sub', 'rsp', '40'), t.i('call', t.csym(name)), t.i('add', 'rsp', '40')]
    return [t.i('call', t.csym(name))]


def hdr_site(t, p, isa, n, store_mem, tag):
    # transfer::output<HDRTransfer>(scalar slot n) -> store. Mirrors the C++ branch chain:
    # comiss+jbe takes the unordered case like MSVC's lowering (NaN can't reach here post-ReLU
    # anyway). Calls clobber volatile regs: const base is r14, channel offset r13 (callee-saved).
    i, r, m = t.i, t.r, t.m
    x0 = r(isa.x(0))
    ld = 'movss' if not isa.vex else 'vmovss'
    L = [i(ld, x0, t.dword(m('rsp', disp=S_SC + 4 * n))),
         i('mulss', x0, t.dword(m('r14', disp=0))),          # * norm_scale
         i('comiss', x0, t.dword(m('r14', disp=4))),         # <= x0?
         i('jbe', t.lbl(f'{tag}_lo')),
         i('comiss', x0, t.dword(m('r14', disp=8))),         # <= x1?
         i('jbe', t.lbl(f'{tag}_mi')),
         i('subss', x0, t.dword(m('r14', disp=12))),         # val - g
         i('divss', x0, t.dword(m('r14', disp=16)))]         # / e
    L += call_f(t, p, 'expf')
    L += [i('subss', x0, t.dword(m('r14', disp=20))),        # - f
          i('jmp', t.lbl(f'{tag}_dn')),
          f'{t.lbl(tag + "_mi")}:',
          i('subss', x0, t.dword(m('r14', disp=24))),        # val - d
          i('divss', x0, t.dword(m('r14', disp=28))),        # / b
          i(ld, r(isa.x(1)), t.dword(m('r14', disp=32)))]    # q = 1/c
    L += call_f(t, p, 'powf')
    L += [i('jmp', t.lbl(f'{tag}_dn')),
          f'{t.lbl(tag + "_lo")}:',
          i('divss', x0, t.dword(m('r14', disp=36))),        # / a
          f'{t.lbl(tag + "_dn")}:']
    L += [i(ld, t.dword(store_mem), x0)]
    return L


# ============================================================================
# ARM64 / NEON emitters (AArch64, AAPCS64 everywhere)
#
# Windows ARM64 assembles the .asm with armasm64.exe via a CMake custom command:
# CMake's ASM_MASM language rejects ARM64 ('MASM not supported on this platform') and
# ml64 has no ARM64 target. The _linux.s/_macos.s files are GAS AArch64 assembled by the
# non-MSVC CMake builds (Linux aarch64 / macOS arm64); Mach-O decorates C symbols with '_'
# on arm64 too, so the macOS variant prefixes function symbols and powf/expf with '_'.
# armasm64 quirks encoded below: labels have NO colons (a 'lbl:' line defines nothing!),
# forward label references are fine, no mul-with-immediate (mov #imm + reg mul instead).
#
# Bit-identity target: MSVC arm64's own codegen of the C++ NEON kernel (verified from a
# cl /FA listing): hsum = faddp v.4s + faddp s,v.2s (tree ((L0+L1)+(L2+L3))), fmadd = fmla,
# std::max(a,b) = fcmp a,b + fcselgt (same NaN/signed-zero picks as the x86 maxss mirror),
# hdr compares = fcmp + b.le (unordered takes the <= branch like comiss+jbe).
#
# GPR roles: x0=y x1=x x2=j-bytes | x4-x7 tap row bases (cc2 j1: 4 column ptrs) |
# x8-x11 kblock address scratch (cc2: R0..R3 source row offsets) | x12=4*ii |
# x13,x17 weight tap1/tap2 ptrs | x15 running bias ptr | x16 running weight ptr |
# callee-saved (all survive the hdr powf/expf calls): x19=data x20=out row stride B
# x21=in row stride B x22=w base x23=b base x24=out base x25=x_end x26=y_end x27=rect_x
# x28=out row ptr. cc2 spills data2/st1B to frame slots; frame: 128B q8-q15 + 80B x19-x28.
# Vectors: C0[k]=q(k) C1[k]=q(T+k), A q(2T..2T+3), B q(2T+4..2T+6), ReLU zero q(2T+7).
# ============================================================================

# hdr frame 304: [208..239] 8 hsum scalars, [240..287] loop state + x29/x30 pair
A_SC = 208
A_S_Y, A_S_X, A_S_II, A_S_W2, A_S_B2, A_S_FP = (A_SC + 32 + 8 * k for k in range(6))
# cc2 frame 272: data2 + st1B/st2B slots (x13-x15 stay weight/bias pointers + scratch).
# R0-R3 (Upsample source row offsets) also live in slots: j1w uses x8-x11 as row-offset
# registers, but the x_loop output-pointer math and kblock_arm temps clobber x8-x11, so
# the row offsets are parked at the top of every y and reloaded at the top of every x.
A_S_D2, A_S_ST1, A_S_ST2 = 208, 216, 224
A_S_R0, A_S_R1, A_S_R2, A_S_R3 = 232, 240, 248, 256


def frame_arm(p):
    if p.post_op == 'hdr':
        return 304
    return 272 if p.kind == 'concat2' else 208


def imul_imm(i, dst, src, imm, treg):
    # arm64 has no mul-by-immediate: materialize the constant, then register mul
    if imm == 1:
        return [i('mov', dst, src)]
    return [i('mov', treg, str(imm)), i('mul', dst, src, treg)]


def hsum_arm(t, isa, n):
    # fvec<4> hsum on AArch64: MSVC lowers vaddvq_f32 to exactly this two-step tree
    return [t.i('faddp', f'v{n}.4s', f'v{n}.4s', f'v{n}.4s'),
            t.i('faddp', f's{n}', f'v{n}.2s')]


def smax_arm(t, a, b):
    # std::max(a, b): finite inputs (ReLU outputs) make fmax bit-identical to the
    # fcmp+fcselgt MSVC emits and to the x86 maxss(a, b) mirror (NaN/+-0 picks agree).
    return t.i('fmax', f's{a}', f's{a}', f's{b}')


def kblock_arm(t, p, isa, k, col):
    # one tile column k + the 6 accumulator chains. The row base regs x4-x7 advance by
    # j_step with the channel window, so column k is addressed by the LOAD IMMEDIATE
    # [xrow, #k*col]: multiples of 16 -> scaled LDR, other strides (conv0/conv13's 36 B
    # staged pitch, offsets <= 108 B) -> the assemblers lower to LDUR unscaled loads.
    # This avoids the one-scalar-add-per-load temp form that made the plain walks
    # instruction-bound on Apple cores (Apple clang's walk: ~3 adds/iter, ours was 17).
    i, L = t.i, []
    for a, rowreg in ((A0, 'x4'), (A1, 'x5'), (A2, 'x6'), (A3, 'x7')):
        off = col * k
        L.append(i('ldr', f'q{a}', f'[{rowreg}, #{off}]') if off else i('ldr', f'q{a}', f'[{rowreg}]'))
    L += [i('fmla', f'v{k}.4s', f'v{A0}.4s', f'v{B0}.4s'),
          i('fmla', f'v{k}.4s', f'v{A1}.4s', f'v{B1}.4s'),
          i('fmla', f'v{k}.4s', f'v{A2}.4s', f'v{B2}.4s'),
          i('fmla', f'v{TROWS + k}.4s', f'v{A1}.4s', f'v{B0}.4s'),
          i('fmla', f'v{TROWS + k}.4s', f'v{A2}.4s', f'v{B1}.4s'),
          i('fmla', f'v{TROWS + k}.4s', f'v{A3}.4s', f'v{B2}.4s')]
    return L


def kblock_arm2(t, p, isa, k, col):
    # OB2 column block: the SAME 4 row loads feed both output channels' 4 accumulator
    # chains (12 FMA per 4 loads vs 6 per 4). Per-accumulator tap order is unchanged.
    i, L = t.i, []
    for a, rowreg in ((A0, 'x4'), (A1, 'x5'), (A2, 'x6'), (A3, 'x7')):
        off = col * k
        L.append(i('ldr', f'q{a}', f'[{rowreg}, #{off}]') if off else i('ldr', f'q{a}', f'[{rowreg}]'))
    T = TROWS
    L += [i('fmla', f'v{k}.4s', f'v{A0}.4s', f'v{B0}.4s'),
          i('fmla', f'v{k}.4s', f'v{A1}.4s', f'v{B1}.4s'),
          i('fmla', f'v{k}.4s', f'v{A2}.4s', f'v{B2}.4s'),
          i('fmla', f'v{T + k}.4s', f'v{A1}.4s', f'v{B0}.4s'),
          i('fmla', f'v{T + k}.4s', f'v{A2}.4s', f'v{B1}.4s'),
          i('fmla', f'v{T + k}.4s', f'v{A3}.4s', f'v{B2}.4s'),
          i('fmla', f'v{2 * T + k}.4s', f'v{A0}.4s', f'v{B3}.4s'),
          i('fmla', f'v{2 * T + k}.4s', f'v{A1}.4s', f'v{B4}.4s'),
          i('fmla', f'v{2 * T + k}.4s', f'v{A2}.4s', f'v{B5}.4s'),
          i('fmla', f'v{3 * T + k}.4s', f'v{A1}.4s', f'v{B3}.4s'),
          i('fmla', f'v{3 * T + k}.4s', f'v{A2}.4s', f'v{B4}.4s'),
          i('fmla', f'v{3 * T + k}.4s', f'v{A3}.4s', f'v{B5}.4s')]
    return L


def set_tap_rows(t, p, stride_reg, col):
    # x4=(y-1) x5=y x6=(y+1) x7=(y+2) row bases, all at pixel column (x-1)
    i = t.i
    return [i('mul', 'x8', 'x0', stride_reg), i('sub', 'x9', 'x1', '#1')] + \
        imul_imm(i, 'x9', 'x9', col, 'x10') + \
        [i('add', 'x8', 'x8', 'x9'), i('add', 'x5', 'x19', 'x8'),
         i('sub', 'x4', 'x5', stride_reg), i('add', 'x6', 'x5', stride_reg),
         i('add', 'x7', 'x6', stride_reg)]


def acc_init(t, p, isa, bias_reg):
    # acc = set<0>(biases[ii]) on all 2T accumulators: ldr s zeroes the other lanes
    i = t.i
    L = [i('ldr', 's0', f'[{bias_reg}]')]
    for k in range(1, 2 * TROWS):
        L.append(i('mov', f'v{k}.16b', 'v0.16b'))
    for k in range(min(2 * TROWS, TROWS + 4)):   # C1[k] mirrors, writes into A regs too (dead here)
        L.append(i('mov', f'v{TROWS + k}.16b', f'v{k}.16b'))
    return L


def acc_init2(t, p, isa, bias_reg):
    # OB2: 4T accumulators = bias[ii] (ch0, v0..v(2T-1)) and bias[ii+1] (ch1, v2T..v4T-1),
    # broadcast with zeroed upper lanes (ldr s semantics) exactly like acc_init.
    i, T = t.i, TROWS
    L = [i('ldr', 's0', f'[{bias_reg}]')]
    for k in range(1, 2 * T):
        L.append(i('mov', f'v{k}.16b', 'v0.16b'))
    L.append(i('ldr', f's{2 * T}', f'[{bias_reg}, #4]'))
    for k in range(1, 2 * T):
        L.append(i('mov', f'v{2 * T + k}.16b', f'v{2 * T}.16b'))
    return L


def prologue_arm(t, p, isa):
    i, F = t.i, frame_arm(p)
    cfi = t.syntax == 'gas_arm'
    L = [i('sub', 'sp', 'sp', str(F))]
    if cfi:
        L.append('    .cfi_def_cfa_offset ' + str(F))
    for a, d in ((8, 0), (10, 32), (12, 64), (14, 96)):
        L.append(i('stp', f'q{a}', f'q{a+1}', f'[sp, #{d}]'))
        if cfi:
            for r in (a, a + 1):
                L.append(f'    .cfi_offset d{r}, -{F - d - (8 if r == a + 1 else 0)}')
    for a, d in ((19, 128), (21, 144), (23, 160), (25, 176), (27, 192)):
        L.append(i('stp', f'x{a}', f'x{a+1}', f'[sp, #{d}]'))
        if cfi:
            for r in (a, a + 1):
                L.append(f'    .cfi_offset x{r}, -{F - d - (8 if r == a + 1 else 0)}')
    if p.kind == 'concat2':
        # x0=d1 x1=d2 w2=rx w3=ry w4=rw w5=rh w6=s1 w7=s2; stack: weights/biases/output/out_stride
        L += [
            i('mov', 'x19', 'x0'),                       # data1
            i('str', 'x1', f'[sp, #{A_S_D2}]'),          # data2 (slot: x13-x15 stay scratch)
            i('mov', 'w27', 'w2'),                       # rect_x
            i('add', 'w25', 'w2', 'w4'),                 # x_end
            i('add', 'w26', 'w3', 'w5'),                 # y_end
            i('mov', 'x16', str(p.COL1)), i('mul', 'x14', 'x6', 'x16'),      # st1 bytes
            i('str', 'x14', f'[sp, #{A_S_ST1}]'),
            i('mov', 'x16', str(p.COL2)), i('mul', 'x14', 'x7', 'x16'),      # st2 bytes
            i('str', 'x14', f'[sp, #{A_S_ST2}]'),
            i('ldr', 'x22', f'[sp, #{F}]'),              # weights
            i('ldr', 'x23', f'[sp, #{F+8}]'),            # biases
            i('ldr', 'x24', f'[sp, #{F+16}]'),           # output
            i('ldr', 'w17', f'[sp, #{F+24}]'),           # output_stride (32-bit arg)
            i('mov', 'x16', str(p.PIX)), i('mul', 'x20', 'x17', 'x16'),
            i('mov', 'w0', 'w3'),                        # y = rect_y
        ]
    else:
        # x0=data w1=rx w2=ry w3=rw w4=rh w5=stride x6=weights x7=biases; stack: output/out_stride
        L += [
            i('mov', 'x19', 'x0'),
            i('mov', 'w27', 'w1'),                       # rect_x
            i('add', 'w25', 'w1', 'w3'),                 # x_end
            i('add', 'w26', 'w2', 'w4'),                 # y_end
            i('mov', 'x16', str(p.COL)), i('mul', 'x21', 'x5', 'x16'),       # in row stride B
            i('mov', 'x22', 'x6'),                       # weights
            i('mov', 'x23', 'x7'),                       # biases
            i('ldr', 'x24', f'[sp, #{F}]'),              # output
            i('ldr', 'w17', f'[sp, #{F+8}]'),            # output_stride (32-bit arg)
            i('mov', 'x16', str(p.PIX)), i('mul', 'x20', 'x17', 'x16'),      # out row stride B
            i('mov', 'w0', 'w2'),                        # y = rect_y
        ]
    if p.post_op == 'hdr':
        L.append(i('stp', 'x29', 'x30', f'[sp, #{A_S_FP}]'))   # powf/expf calls clobber x30!
        if cfi:
            L += ['    .cfi_offset x29, -' + str(F - A_S_FP), '    .cfi_offset x30, -' + str(F - A_S_FP - 8)]
    L.append(i('fmov', f's{ZR}', 'wzr'))                 # ReLU zero (scalar reads only)
    return L


def epilogue_arm(t, p, isa):
    i, L = t.i, []
    for a, d in ((8, 0), (10, 32), (12, 64), (14, 96)):
        L.append(i('ldp', f'q{a}', f'q{a+1}', f'[sp, #{d}]'))
    for a, d in ((19, 128), (21, 144), (23, 160), (25, 176), (27, 192)):
        L.append(i('ldp', f'x{a}', f'x{a+1}', f'[sp, #{d}]'))
    if p.post_op == 'hdr':
        L.append(i('ldp', 'x29', 'x30', f'[sp, #{A_S_FP}]'))
    L.append(i('add', 'sp', 'sp', str(frame_arm(p))))
    L.append(i('ret'))
    return L


def call_arm(t, name):
    return [t.i('bl', t.csym(name))]  # '_' prefix on macOS (Mach-O decorates C symbols on arm64 too)


def hdr_site_arm(t, p, isa, n, col, tag):
    # transfer::output<HDRTransfer>(scalar slot n) -> store at [x14 = x28+PIX*col, x12=4*ii].
    # powf/expf clobber x0-x17 + v-regs: const base + store address are rematerialized
    # after the calls (adr range covers the pool placed right after the function).
    i = t.i
    L = [i('ldr', 's0', f'[sp, #{A_SC + 4 * n}]'),
         i('adr', 'x14', f'{p.tag}_hdr_c'),
                 i('ldr', 's1', '[x14, #0]'), i('fmul', 's0', 's0', 's1'),   # * norm_scale
                 i('ldr', 's1', '[x14, #4]'), i('fcmp', 's0', 's1'), i('b.le', f'L{tag}_lo'),
                 i('ldr', 's1', '[x14, #8]'), i('fcmp', 's0', 's1'), i('b.le', f'L{tag}_mi'),
                 i('ldr', 's1', '[x14, #12]'), i('fsub', 's0', 's0', 's1'),  # val - g
                 i('ldr', 's1', '[x14, #16]'), i('fdiv', 's0', 's0', 's1')]  # / e
    L += call_arm(t, 'expf')
    L += [i('adr', 'x14', f'{p.tag}_hdr_c'),
          i('ldr', 's1', '[x14, #20]'), i('fsub', 's0', 's0', 's1'),         # - f
          i('b', f'L{tag}_dn'),
          f'L{tag}_mi',
          i('ldr', 's1', '[x14, #24]'), i('fsub', 's0', 's0', 's1'),         # val - d
          i('ldr', 's1', '[x14, #28]'), i('fdiv', 's0', 's0', 's1'),         # / b
          i('ldr', 's1', '[x14, #32]')]                                      # q = 1/c
    L += call_arm(t, 'powf')
    L += [i('b', f'L{tag}_dn'),
          f'L{tag}_lo',
          i('ldr', 's1', '[x14, #36]'), i('fdiv', 's0', 's0', 's1'),         # / a
          f'L{tag}_dn',
          i('ldr', 'x12', f'[sp, #{A_S_II}]'),
          i('add', 'x14', 'x28', f'#{p.PIX * col}'),
          i('str', 's0', '[x14, x12]')]
    return L


def store_block_arm(t, p, isa, row_half):
    # hsum + ReLU + clipped scalar stores for one output row (row_half = 0/1 of the y pair);
    # caller has already shifted x28 to the right row.
    i, L, T = t.i, [], TROWS
    for k in range(T):
        L.append(i('add', 'x8', 'x1', f'#{k}'))
        L.append(i('cmp', 'x8', 'x25'))
        L.append(i('b.ge', f'Lc{row_half}_{k}'))
        L += hsum_arm(t, isa, T * row_half + k)
        L.append(smax_arm(t, T * row_half + k, ZR))
        L.append(i('add', 'x8', 'x28', f'#{p.PIX * k}'))
        L.append(i('str', f's{T * row_half + k}', '[x8, x12]'))
        L.append(f'Lc{row_half}_{k}')
    return L


def store_block_arm2(t, p, isa, row_half):
    # OB2 store: one x-guard serves both output channels; ch1 lands at +4 bytes (adjacent
    # channel slot), so no extra index register is needed.
    i, L, T = t.i, [], TROWS
    for k in range(T):
        L.append(i('add', 'x8', 'x1', f'#{k}'))
        L.append(i('cmp', 'x8', 'x25'))
        L.append(i('b.ge', f'Lc{row_half}_{k}'))
        a0 = T * row_half + k
        a1 = 2 * T + a0
        L += hsum_arm(t, isa, a0)
        L += hsum_arm(t, isa, a1)
        L.append(smax_arm(t, a0, ZR))
        L.append(smax_arm(t, a1, ZR))
        L.append(i('add', 'x8', 'x28', f'#{p.PIX * k}'))
        L.append(i('str', f's{a0}', '[x8, x12]'))
        L.append(i('add', 'x8', 'x8', '#4'))
        L.append(i('str', f's{a1}', '[x8, x12]'))
        L.append(f'Lc{row_half}_{k}')
    return L


def body_arm(t, p, isa):
    if p.kind == 'concat2':
        return body_cc2_arm(t, p, isa)
    i, L = t.i, []
    add = L.append
    js, T = p.j_step(isa), TROWS
    ds = 1 if p.downsample else 0

    add('y_loop')
    add(i('cmp', 'x0', 'x26')); add(i('b.ge', 'Ly_done'))
    add(i('mov', 'x1', 'x27'))
    add('x_loop')
    add(i('cmp', 'x1', 'x25')); add(i('b.ge', 'Lx_done'))
    # (set_tap_rows moved to ii top: x4-x7 now WALK the channel window inside j_loop)
    # x28 = output_g + out_row_stride*(y>>ds) + PIX*(x>>ds)   (ds = 1 for downsample passes)
    add(i('mov', 'x8', 'x0'))
    if ds:
        add(i('lsr', 'x8', 'x8', '#1'))
    add(i('mul', 'x8', 'x8', 'x20')); add(i('add', 'x28', 'x8', 'x24'))
    add(i('mov', 'x9', 'x1'))
    if ds:
        add(i('lsr', 'x9', 'x9', '#1'))
    L += imul_imm(i, 'x9', 'x9', p.PIX, 'x10')
    add(i('add', 'x28', 'x28', 'x9'))
    add(i('mov', 'x12', 'xzr'))                              # 4*ii
    add(i('mov', 'x16', 'x22'))                              # weight tap block ptrs
    add(i('add', 'x17', 'x16', f'#{p.TAP}'))
    add(i('add', 'x13', 'x17', f'#{p.TAP}'))
    add(i('mov', 'x15', 'x23'))                              # biases ptr

    add('ii_loop')
    add(i('cmp', 'x12', str(p.II_END))); add(i('b.ge', 'Lii_done'))
    L += set_tap_rows(t, p, 'x21', p.COL)                  # rebuild row bases per ii: j_loop walks x4-x7
    if OB2:
        # ch1 weight tap ptrs: x9/x10/x11 are scratch for set_tap_rows but stay free from
        # here to the store phase (which only uses x8) - re-derive from the ch0 cursors
        # each ii instead of advancing them, keeps the ii-tail cursor block untouched.
        L += [i('add', 'x9', 'x16', f'#{p.W_ROW}'), i('add', 'x10', 'x17', f'#{p.W_ROW}'),
              i('add', 'x11', 'x13', f'#{p.W_ROW}')]
    L += acc_init2(t, p, isa, 'x15') if OB2 else acc_init(t, p, isa, 'x15')
    add(i('mov', 'x2', 'xzr'))
    add('j_loop')
    for reg, base in ((B0, 'x16'), (B1, 'x17'), (B2, 'x13')):
        add(i('ldr', f'q{reg}', f'[{base}, x2]'))
    if OB2:
        for reg, base in ((B3, 'x9'), (B4, 'x10'), (B5, 'x11')):
            add(i('ldr', f'q{reg}', f'[{base}, x2]'))
    for k in range(T):
        L += kblock_arm2(t, p, isa, k, p.COL) if OB2 else kblock_arm(t, p, isa, k, p.COL)
    add(i('add', 'x2', 'x2', str(js)))
    for rr in ('x4', 'x5', 'x6', 'x7'):                    # channel window walks the row bases
        add(i('add', rr, rr, str(js)))                    # (loads use [xrow, #k*COL] immediates)
    add(i('cmp', 'x2', str(p.J_END))); add(i('b.lt', 'j_loop'))

    if p.downsample:
        # downsample epilogue: hsum + ReLU all, then 2x2 max-pool, store at half coords
        # (xx < x_end guard only; row y+1 always participates in the pool).
        if OB2:
            # OB2: pool both output channels (ch1 accumulators offset by 2T), one guard
            # per pair, ch1 store lands at +4 bytes.
            for base in (2 * T, 0):
                for n in list(range(base + T, base + 2 * T)) + list(range(base, base + T)):
                    L += hsum_arm(t, isa, n)
                    add(smax_arm(t, n, ZR))
            for k in range(0, T, 2):
                add(i('add', 'x8', 'x1', f'#{k}')); add(i('cmp', 'x8', 'x25')); add(i('b.ge', f'Lp{k}'))
                add(smax_arm(t, k + 1, k))                    # max(C0[k], C0[k+1])
                add(smax_arm(t, T + k + 1, T + k))            # max(C1[k], C1[k+1])
                add(smax_arm(t, T + k + 1, k + 1))            # max of the two pairs
                add(smax_arm(t, T + k + 1, ZR))               # std::max(0.0f, m)
                add(smax_arm(t, 2 * T + k + 1, 2 * T + k))
                add(smax_arm(t, 3 * T + k + 1, 3 * T + k))
                add(smax_arm(t, 3 * T + k + 1, 2 * T + k + 1))
                add(smax_arm(t, 3 * T + k + 1, ZR))
                add(i('add', 'x8', 'x28', f'#{p.PIX * (k // 2)}'))
                add(i('str', f's{T + k + 1}', '[x8, x12]'))
                add(i('add', 'x8', 'x8', '#4'))
                add(i('str', f's{3 * T + k + 1}', '[x8, x12]'))
                add(f'Lp{k}')
        else:
            for n in list(range(T, 2 * T)) + list(range(T)):
                L += hsum_arm(t, isa, n)
                add(smax_arm(t, n, ZR))
            for k in range(0, T, 2):
                add(i('add', 'x8', 'x1', f'#{k}')); add(i('cmp', 'x8', 'x25')); add(i('b.ge', f'Lp{k}'))
                add(smax_arm(t, k + 1, k))                    # max(C0[k], C0[k+1])
                add(smax_arm(t, T + k + 1, T + k))            # max(C1[k], C1[k+1])
                add(smax_arm(t, T + k + 1, k + 1))            # max of the two pairs
                add(smax_arm(t, T + k + 1, ZR))               # std::max(0.0f, m)
                add(i('add', 'x8', 'x28', f'#{p.PIX * (k // 2)}'))
                add(i('str', f's{T + k + 1}', '[x8, x12]'))
                add(f'Lp{k}')
    elif p.post_op == 'hdr':
        # hdr epilogue: spill all 2T hsum+ReLU scalars to slots first (powf/expf clobber
        # x0-x17 + v0-v31 except d8-15), then run the transfer chain per store site.
        # Loop state (y/x/ii/weight+bias ptrs) lives in frame slots across the calls;
        # x19-x28 (tensors, bounds, out ptr) are callee-saved and survive.
        add(i('add', 'x8', 'x0', '#1')); add(i('cmp', 'x8', 'x26')); add(i('b.ge', 'Lrow0_spill'))
        for k in range(T):
            L += hsum_arm(t, isa, T + k)
            add(smax_arm(t, T + k, ZR))
            add(i('str', f's{T+k}', f'[sp, #{A_SC + 4 * (T + k)}]'))
        add('Lrow0_spill')
        for k in range(T):
            L += hsum_arm(t, isa, k)
            add(smax_arm(t, k, ZR))
            add(i('str', f's{k}', f'[sp, #{A_SC + 4 * k}]'))
        add(i('str', 'x0', f'[sp, #{A_S_Y}]')); add(i('str', 'x1', f'[sp, #{A_S_X}]'))
        add(i('str', 'x12', f'[sp, #{A_S_II}]')); add(i('str', 'x16', f'[sp, #{A_S_W2}]'))
        add(i('str', 'x15', f'[sp, #{A_S_B2}]'))
        add(i('add', 'x8', 'x0', '#1')); add(i('cmp', 'x8', 'x26')); add(i('b.ge', 'Lrow0_store'))
        add(i('add', 'x28', 'x28', 'x20'))                     # shift to row y+1
        for k in range(T):
            # x1 is CALLER-SAVED and the first powf/expf bl clobbers it - guards read x back from
            # the slot saved above, they must not touch x1 after any call in this section
            add(i('ldr', 'x8', f'[sp, #{A_S_X}]')); add(i('add', 'x8', 'x8', f'#{k}'))
            add(i('cmp', 'x8', 'x25')); add(i('b.ge', f'Lc1_{k}'))
            L += hdr_site_arm(t, p, isa, T + k, k, f'c1_{k}')
            add(f'Lc1_{k}')
        add(i('sub', 'x28', 'x28', 'x20'))                     # back to row y
        add('Lrow0_store')
        for k in range(T):
            add(i('ldr', 'x8', f'[sp, #{A_S_X}]')); add(i('add', 'x8', 'x8', f'#{k}'))
            add(i('cmp', 'x8', 'x25')); add(i('b.ge', f'Lc0_{k}'))
            L += hdr_site_arm(t, p, isa, k, k, f'c0_{k}')
            add(f'Lc0_{k}')
        add(i('ldr', 'x0', f'[sp, #{A_S_Y}]')); add(i('ldr', 'x1', f'[sp, #{A_S_X}]'))
        add(i('ldr', 'x12', f'[sp, #{A_S_II}]')); add(i('ldr', 'x16', f'[sp, #{A_S_W2}]'))
        add(i('ldr', 'x15', f'[sp, #{A_S_B2}]'))
        add(i('fmov', f's{ZR}', 'wzr'))                        # ReLU zero: calls trash v0-v31 (d8-d15
                                                                # safe, s15 is NOT) - re-establish before
                                                                # the next ii's hsum/ReLU
        # x4-x7 clobbered by the calls: rebuilt by set_tap_rows at the top of the next ii
    else:
        add(i('add', 'x8', 'x0', '#1')); add(i('cmp', 'x8', 'x26')); add(i('b.ge', 'Lrow0'))
        add(i('add', 'x28', 'x28', 'x20'))                     # shift to row y+1
        L += store_block_arm2(t, p, isa, 1) if OB2 else store_block_arm(t, p, isa, 1)
        add(i('sub', 'x28', 'x28', 'x20'))                     # back to row y
        add('Lrow0')
        L += store_block_arm2(t, p, isa, 0) if OB2 else store_block_arm(t, p, isa, 0)

    add(i('mov', 'x8', str(2 * p.W_ROW if OB2 else p.W_ROW)))
    add(i('add', 'x16', 'x16', 'x8')); add(i('add', 'x17', 'x17', 'x8')); add(i('add', 'x13', 'x13', 'x8'))
    add(i('add', 'x15', 'x15', '#8' if OB2 else '#4'))
    add(i('add', 'x12', 'x12', '#8' if OB2 else '#4'))
    add(i('b', 'ii_loop'))
    add('Lii_done')
    add(i('add', 'x1', 'x1', str(T))); add(i('b', 'x_loop'))
    add('Lx_done')
    add(i('add', 'x0', 'x0', '#2')); add(i('b', 'y_loop'))
    add('Ly_done')
    return L


def body_cc2_arm(t, p, isa):
    """2-input concat kernel: j1 walks input1 with Upsample addressing (C[k] col ptrs in
    x4-x7, source row offsets R0-R3 reloaded from frame slots into x8-x11 at the top of every
    ii iteration - j2's kblock_arm temps scratch x8-x11 once per ii), j2 walks input2 plain
    (kblock over COL2). data2/st1B/st2B live in frame slots. x16 is the ONLY weight cursor
    (+W_ROW at the end of ii); its taps load as [x16, x2/x17/x13] with x17/x13 walking from
    +TAP1/2*TAP1, and j2 derives its base x15 = x16 + W_OFF per ii - never re-base x16/x17/x13,
    the earlier form (j2 re-issuing bases) corrupted every ii past the first."""
    i, L = t.i, []
    add = L.append
    js, T = p.j_step(isa), TROWS

    add('y_loop')
    add(i('cmp', 'x0', 'x26')); add(i('b.ge', 'Ly_done'))
    # Upsample source row offsets row_r * st1B; row_0 = y==0 ? -1 : (y-1)/2
    add(i('ldr', 'x12', f'[sp, #{A_S_ST1}]'))
    add(i('mov', 'x8', 'x0')); add(i('lsr', 'x8', 'x8', '#1')); add(i('mul', 'x8', 'x8', 'x12'))    # R1
    add(i('add', 'x9', 'x0', '#1')); add(i('lsr', 'x9', 'x9', '#1')); add(i('mul', 'x9', 'x9', 'x12'))  # R2
    add(i('add', 'x10', 'x0', '#2')); add(i('lsr', 'x10', 'x10', '#1')); add(i('mul', 'x10', 'x10', 'x12'))  # R3
    add(i('cmp', 'x0', '#1')); add(i('b.lo', 'LRneg'))
    add(i('sub', 'x11', 'x0', '#1')); add(i('lsr', 'x11', 'x11', '#1')); add(i('mul', 'x11', 'x11', 'x12'))  # R0
    add(i('b', 'LR0'))
    add('LRneg')
    add(i('neg', 'x11', 'x12'))                              # R0 = -st1B (clamped row)
    add('LR0')
    for reg, slot in (('x11', A_S_R0), ('x8', A_S_R1), ('x9', A_S_R2), ('x10', A_S_R3)):
        add(i('str', reg, f'[sp, #{slot}]'))                 # park R0-R3 (x8-x11 are header scratch)
    add(i('mov', 'x1', 'x27'))

    add('x_loop')
    add(i('cmp', 'x1', 'x25')); add(i('b.ge', 'Lx_done'))
    # x28 = output_g + out_row_stride*y + PIX*x
    add(i('mul', 'x8', 'x0', 'x20')); add(i('add', 'x28', 'x8', 'x24'))
    L += imul_imm(i, 'x9', 'x1', p.PIX, 'x10')
    add(i('add', 'x28', 'x28', 'x9'))
    add(i('mov', 'x12', 'xzr'))                              # 4*ii
    add(i('mov', 'x16', 'x22'))                              # weight base cursor (advanced W_ROW per ii)
    add(i('mov', 'x14', 'x23'))                              # biases ptr

    add('ii_loop')
    add(i('cmp', 'x12', str(p.II_END))); add(i('b.ge', 'Lii_done'))
    # weight taps load as [x16, x2] / [x16, x17] / [x16, x13] - x17/x13 are pure byte OFFSETS
    # (never pointers!) walking the same js as x2 from +TAP1/2*TAP1 (no displacement form for
    # reg-offset ldr, and j2 must not re-base x16: x16 is the only weight cursor advanced at the
    # end of ii)
    add(i('mov', 'x17', str(p.TAP1))); add(i('mov', 'x13', str(2 * p.TAP1)))
    L += acc_init2(t, p, isa, 'x14') if OB2 else acc_init(t, p, isa, 'x14')
    for reg, slot in (('x11', A_S_R0), ('x8', A_S_R1), ('x9', A_S_R2), ('x10', A_S_R3)):
        add(i('ldr', reg, f'[sp, #{slot}]'))                 # R0-R3 back: j2's kblock temps (x8-x11)
                                                             # scratch them once per ii, j1w needs them again
    # ---- j1: input1, Upsample. tap = x_off is constant per block; C[k] = data1 + x_final(k)*COL1
    add(i('mov', 'x2', 'xzr'))
    for tap in range(3):
        add(i('add', 'x3', 'x1', f'#{tap + 1}'))
        add(i('mov', 'x15', str(p.COL1)))
        for k in range(T):
            add(i('mov' if k == 0 else 'add', f'x{4 + k}', 'x3', *(f'#{k}',) if k else ()))
            add(i('lsr', f'x{4 + k}', f'x{4 + k}', '#1'))
            add(i('sub', f'x{4 + k}', f'x{4 + k}', '#1'))
            add(i('mul', f'x{4 + k}', f'x{4 + k}', 'x15'))
            add(i('add', f'x{4 + k}', f'x{4 + k}', 'x19'))
        if OB2:
            # ch1 j1 weight base: x15 is free once the column pointers are built (its COL1
            # stride job is done); ch1's three taps share it since x17/x13 are walking
            # byte offsets, not pointers.
            add(i('mov', 'x3', str(p.W_ROW))); add(i('add', 'x15', 'x16', 'x3'))
        add(f'j1w{tap}')
        add(i('ldr', f'q{B0}', '[x16, x2]'))
        add(i('ldr', f'q{B1}', '[x16, x17]'))
        add(i('ldr', f'q{B2}', '[x16, x13]'))
        if OB2:
            add(i('ldr', f'q{B3}', '[x15, x2]'))
            add(i('ldr', f'q{B4}', '[x15, x17]'))
            add(i('ldr', f'q{B5}', '[x15, x13]'))
        # row groups: (R reg, B reg feeding C0 or None, B reg feeding C1 or None)
        for rreg, c0b, c1b in (('x11', B0, None), ('x8', B1, B0), ('x9', B2, B1), ('x10', None, B2)):
            for k in range(T):
                add(i('ldr', f'q{A0 + k}', f'[x{4 + k}, {rreg}]'))
            for k in range(T):
                if c0b is not None:
                    add(i('fmla', f'v{k}.4s', f'v{A0 + k}.4s', f'v{c0b}.4s'))
                    if OB2:
                        add(i('fmla', f'v{2 * T + k}.4s', f'v{A0 + k}.4s', f'v{c0b + 3}.4s'))
                if c1b is not None:
                    add(i('fmla', f'v{T + k}.4s', f'v{A0 + k}.4s', f'v{c1b}.4s'))
                    if OB2:
                        add(i('fmla', f'v{3 * T + k}.4s', f'v{A0 + k}.4s', f'v{c1b + 3}.4s'))
        add(i('add', 'x2', 'x2', str(js))); add(i('add', 'x17', 'x17', str(js))); add(i('add', 'x13', 'x13', str(js)))
        for k in range(T):
            add(i('add', f'x{4 + k}', f'x{4 + k}', str(js)))
        add(i('cmp', 'x2', f'{(tap + 1) * p.in_ch * 4}')); add(i('b.lt', f'j1w{tap}'))

    # ---- j2: input2, plain tap window (kblock over COL2) ----
    add(i('ldr', 'x8', f'[sp, #{A_S_D2}]'))
    add(i('sub', 'x9', 'x1', '#1'))
    L += imul_imm(i, 'x9', 'x9', p.COL2, 'x10')
    add(i('add', 'x8', 'x8', 'x9'))
    add(i('ldr', 'x9', f'[sp, #{A_S_ST2}]'))
    add(i('mul', 'x10', 'x0', 'x9'))
    add(i('add', 'x8', 'x8', 'x10'))
    add(i('mov', 'x5', 'x8'))                                # A1 base (row y)
    add(i('sub', 'x4', 'x5', 'x9'))                          # A0 base (row y-1)
    add(i('add', 'x6', 'x5', 'x9'))                          # A2 base (row y+1)
    add(i('add', 'x7', 'x6', 'x9'))                          # A3 base (row y+2)
    add(i('add', 'x15', 'x16', f'#{p.W_OFF}'))               # x15 = j2 weight base (derived from the ii's x16)
    if OB2:
        add(i('mov', 'x3', str(p.W_ROW))); add(i('add', 'x3', 'x15', 'x3'))   # x3 = j2 ch1 weight base
    add(i('mov', 'x17', str(p.TAP2))); add(i('mov', 'x13', str(2 * p.TAP2)))
    add(i('mov', 'x2', 'xzr'))
    # non-16-B strides (conv13/conv0's 36 B staged pitch) become LDUR unscaled loads -
    # the assemblers accept ldr q,[xN,#36] and lower to ldur (offsets here <= 108 B)
    add('j2_loop')
    add(i('ldr', f'q{B0}', '[x15, x2]'))
    add(i('ldr', f'q{B1}', '[x15, x17]'))
    add(i('ldr', f'q{B2}', '[x15, x13]'))
    if OB2:
        add(i('ldr', f'q{B3}', '[x3, x2]'))
        add(i('ldr', f'q{B4}', '[x3, x17]'))
        add(i('ldr', f'q{B5}', '[x3, x13]'))
    for k in range(T):
        L += kblock_arm2(t, p, isa, k, p.COL2) if OB2 else kblock_arm(t, p, isa, k, p.COL2)
    add(i('add', 'x2', 'x2', str(js))); add(i('add', 'x17', 'x17', str(js))); add(i('add', 'x13', 'x13', str(js)))
    for rr in ('x4', 'x5', 'x6', 'x7'):                     # channel window walks the row bases
        add(i('add', rr, rr, str(js)))
    add(i('cmp', 'x2', str(p.J2_END))); add(i('b.lt', 'j2_loop'))

    # ---- epilogue: same as the plain kernel (hsum + ReLU + clipped scalar stores) ----
    add(i('add', 'x8', 'x0', '#1')); add(i('cmp', 'x8', 'x26')); add(i('b.ge', 'Lrow0'))
    add(i('add', 'x28', 'x28', 'x20'))                       # shift to row y+1
    L += store_block_arm2(t, p, isa, 1) if OB2 else store_block_arm(t, p, isa, 1)
    add(i('sub', 'x28', 'x28', 'x20'))                       # back to row y
    add('Lrow0')
    L += store_block_arm2(t, p, isa, 0) if OB2 else store_block_arm(t, p, isa, 0)

    add(i('mov', 'x8', str(2 * p.W_ROW if OB2 else p.W_ROW)))
    add(i('add', 'x16', 'x16', 'x8'))                        # single weight cursor; x17/x13 offsets are
    b8 = '#8' if OB2 else '#4'                               # re-materialized per ii
    add(i('add', 'x14', 'x14', b8)); add(i('add', 'x12', 'x12', b8))
    add(i('b', 'ii_loop'))
    add('Lii_done')
    add(i('add', 'x1', 'x1', str(T))); add(i('b', 'x_loop'))
    add('Lx_done')
    add(i('add', 'x0', 'x0', '#2')); add(i('b', 'y_loop'))
    add('Ly_done')
    return L


def header(t, p, isa):
    if p.kind == 'concat2':
        return """; generated by src/Ray/scripts/gen_conv_asm.py - do not edit by hand
; UNet conv pass %d (%s), %s kernel, bit-identical to
; NS::ConvolutionConcat3x3<%d, %d, %d, %d, Upsample, None, ReLU> (Convolution.h):
; same FMA chains (j1 over input1 with Upsample addressing, then j2 over input2), hsum, ReLU.
; void %s(const float* data1, const float* data2, int rect_x, int rect_y, int rect_w, int rect_h,
;         int stride1, int stride2, const float* weights, const float* biases,
;         float* output, int output_stride)
""" % (p.num, p.res_desc, isa.name.upper(), isa.V, p.in_ch, p.in2, p.out_ch, t.sym(p, isa))
    return """; generated by src/Ray/scripts/gen_conv_asm.py - do not edit by hand
; UNet conv pass %d (%s), %s kernel, bit-identical to
; NS::Convolution3x3<%d, %d, %d> (Convolution.h): same FMA chains, hsum, ReLU.
; void %s(const float* data, int rect_x, int rect_y, int rect_w, int rect_h,
;         int stride, const float* weights, const float* biases,
;         float* output, int output_stride)
""" % (p.num, p.res_desc, isa.name.upper(), isa.V, p.in_ch, p.out_ch, t.sym(p, isa))


def wrap_gas(t, p, isa):
    _set_plan(p, isa)
    macos = 'macos' in t.name
    lines = ['.intel_syntax noprefix']
    lines.append('.section __TEXT,__text,regular,pure_instructions' if macos else '.text')
    sym = t.sym(p, isa)
    lines += [f'.globl {sym}']
    if not macos:
        lines.append(f'.type {sym}, @function')
    lines.append(f'{sym}:')
    # macOS: no CFI at all - Mach-O's CFI object writer rejects these functions' label-difference
    # advance_loc expressions ("invalid CFI advance_loc expression"). Same posture as the ml64 /
    # armasm64 objects: these hot render kernels are never unwound through.
    if not macos:
        lines.append('.cfi_startproc')

    body_lines = prologue(t, p, isa) + body(t, p, isa) + epilogue(t, p, isa)
    # interleave CFI: the 8 pushes come first, then 'sub rsp, frame'; the epilogue tail
    # is [10 x movups restore, add rsp, frame, 8 x pop, ret] (movups don't move CFA).
    # sysv hdr passes make bare calls only (no mid-function rsp changes).
    # Use def_cfa_offset (absolute) - GNU as needs the CFA described across the frame alloc.
    n_push, n_epi, frame = len(CALLEE_SAVED), 10, p.frame(t.abi, isa)
    off = 0
    for n, line in enumerate(body_lines):
        lines.append(line)
        if macos:
            continue
        if n < n_push:
            off += 8
            reg = line.split()[-1]
            lines.append(f'    .cfi_def_cfa_offset {off}')
            lines.append(f'    .cfi_offset {reg}, -{off}')
        elif n == n_push:
            off += frame
            lines.append(f'    .cfi_def_cfa_offset {off}')
        elif n == len(body_lines) - n_epi:
            off -= frame
            lines.append(f'    .cfi_def_cfa_offset {off}')
        elif n > len(body_lines) - n_epi and n < len(body_lines) - 1:
            off -= 8
            lines.append(f'    .cfi_def_cfa_offset {off}')
    if not macos:
        lines.append('.cfi_endproc')
        lines.append(f'.size {sym}, .-{sym}')
    if p.post_op == 'hdr':
        lines += t.const_block(p.tag + '_hdr_c', HDR_CONSTS)
    if not macos:
        lines.append('.section .note.GNU-stack,"",@progbits')  # keep GNU ld from implying an executable stack
    return lines


def wrap_masm(t, p, isa):
    _set_plan(p, isa)
    sym = t.sym(p, isa)
    lines = [header(t, p, isa)]
    if p.post_op == 'hdr':
        lines += ['EXTERN powf:PROC', 'EXTERN expf:PROC']
    lines += ['.code', f'PUBLIC {sym}', f'{sym} PROC']
    lines += prologue(t, p, isa) + body(t, p, isa) + epilogue(t, p, isa)
    lines += [f'{sym} ENDP']
    if p.post_op == 'hdr':
        # ml64 auto-unwind only describes the prologue; the mid-function sub/add rsp,40 around
        # powf/expf is not encoded (these hot render kernels are never unwound through).
        lines += t.const_block(p.tag + '_hdr_c', HDR_CONSTS)
    # no END directive: the master TU conv_all_win_x64.asm includes all kernels and supplies END
    return lines


def wrap_armasm64(t, p, isa):
    _set_plan(p, isa)
    sym = p.symbol(isa)
    lines = [header(t, p, isa)]
    if p.post_op == 'hdr':
        lines += ['\tIMPORT powf', '\tIMPORT expf']
    lines += ['\tAREA .text, CODE, READONLY', f'\tEXPORT {sym}', f'{sym} PROC']
    lines += prologue_arm(t, p, isa) + body_arm(t, p, isa) + epilogue_arm(t, p, isa)
    lines += [f'{sym} ENDP']
    if p.post_op == 'hdr':
        # literal pool right after the function body: hdr_site reaches it with one adr
        lines += [f'{p.tag}_hdr_c'] + [f'\tDCD {v}' for v in HDR_CONSTS]
    # no END directive: CMake's asm_concat.cmake appends one after concatenating the 15 kernels
    return lines


def wrap_gas_arm(t, p, isa):
    import re
    _set_plan(p, isa)
    macos = 'macos' in t.name
    sym = t.sym(p, isa)  # '_' prefix on macOS: Mach-O decorates C symbols with '_' on arm64 too (proven by ld64: RendererNEON.cpp.o references _ray_convN_neon)
    lines = ['.section __TEXT,__text,regular,pure_instructions' if macos else '.text']
    lines.append(f'.globl {sym}')
    if not macos:
        lines.append(f'.type {sym}, @function')
    lines += [f'{sym}:']
    if not macos:
        lines.append('.cfi_startproc')
    epi = epilogue_arm(t, p, isa)
    lines += prologue_arm(t, p, isa) + body_arm(t, p, isa) + epi[:-1]
    if not macos:
        lines.append('    .cfi_def_cfa_offset 0')  # after 'add sp, sp, frame', before ret
    lines += [epi[-1]]
    if not macos:
        lines += ['.cfi_endproc', f'.size {sym}, .-{sym}']
    if p.post_op == 'hdr':
        lines += ['.p2align 2', f'{p.tag}_hdr_c:'] + [f'    .long {v}' for v in HDR_CONSTS]
    if not macos:
        lines.append('.section .note.GNU-stack,"",@progbits')  # keep GNU ld from implying an executable stack
    else:
        # prologue_arm emits inline CFI for the q8-q15 save region; strip it for Mach-O (see wrap_gas)
        lines = [ln for ln in lines if not ln.strip().startswith('.cfi_')]
    # label definitions are emitted bare (armasm64 needs no colons); append them for GAS
    lines = [re.sub(r'^([A-Za-z_][A-Za-z0-9_]*)$', r'\1:', ln)
             if ln and not ln[0].isspace() and ln[0] not in '.;' else ln for ln in lines]
    return lines


def ns_labels(lines, t, p, isa):
    # Give every internal label a per-kernel namespace so all kernels of one target can be
    # assembled in a single pass: the master TU .asm/.s includes the per-kernel sources and
    # neither ml64 nor armasm64 scopes labels per PROC (duplicate labels are hard errors;
    # same in GAS). Function symbols are already unique per file and are left as-is.
    ns = f'{p.tag}_{isa.name}_'
    skip = {t.sym(p, isa), p.symbol(isa), 'END'}
    if t.syntax == 'armasm64':
        pat_def = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)$')     # armasm labels carry no colon
    elif t.syntax == 'gas_arm':
        pat_def = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):$')    # colons added by wrap_gas_arm
    else:
        pat_def = re.compile(r'^([.A-Za-z_][A-Za-z0-9_.$]*):')  # 'y_loop:' / '.Lc1_0:'
    labels = []
    for ln in lines:
        m = pat_def.match(ln)
        if m and m.group(1) not in skip:
            labels.append(m.group(1))
    for tok in dict.fromkeys(labels):
        new = '.L' + ns + tok[2:] if tok.startswith('.L') else ns + tok
        pat = re.compile(r'(?<![\w.$])' + re.escape(tok) + r'(?![\w$])')
        lines = [pat.sub(new, ln) for ln in lines]
    return lines


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'internal', 'generated')
    os.makedirs(out_dir, exist_ok=True)

    def write_asm(path, lines):
        with open(path, 'w', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        print('wrote', os.path.normpath(path))

    targets = [
        Target('win_masm', 'masm', 'win'),
        Target('linux_gas', 'gas', 'sysv'),
        Target('macos_gas', 'gas', 'sysv'),
    ]
    suffix = {'win_masm': '.asm', 'linux_gas': '_linux.s', 'macos_gas': '_macos.s'}
    targets_arm = [
        Target('win_armasm64', 'armasm64', 'win'),
        Target('linux_gas_arm', 'gas_arm', 'sysv'),
        Target('macos_gas_arm', 'gas_arm', 'sysv'),
    ]
    suffix_arm = {'win_armasm64': '.asm', 'linux_gas_arm': '_linux.s', 'macos_gas_arm': '_macos.s'}
    for p in PASSES:
        for isa in ISAS:
            for t in targets:
                lines = wrap_masm(t, p, isa) if t.syntax == 'masm' else wrap_gas(t, p, isa)
                write_asm(os.path.join(out_dir, f'{p.tag}_{isa.name}{suffix[t.name]}'), ns_labels(lines, t, p, isa))
        for isa in ISAS_ARM:
            for t in targets_arm:
                lines = wrap_armasm64(t, p, isa) if t.syntax == 'armasm64' else wrap_gas_arm(t, p, isa)
                write_asm(os.path.join(out_dir, f'{p.tag}_{isa.name}{suffix_arm[t.name]}'), ns_labels(lines, t, p, isa))

    # Master TUs: one object per target/arch instead of 60/15. The per-kernel sources stay
    # separate files - the master only pulls them in (ml64 INCLUDE needs /I<generated dir>,
    # GAS .include needs -I<generated dir>; armasm64 has no include directive, so CMake
    # concatenates its 15 .asm at build time via scripts/asm_concat.cmake).
    def emit_master(fname, header, includes, footer):
        write_asm(os.path.join(out_dir, fname), [header] + includes + footer)

    emit_master('conv_all_win_x64.asm',
                '; master TU: assembles all 60 x86 kernels into one object (ml64 + /I<generated dir>;'
                ' per-kernel sources carry no END, it is supplied here)',
                [f'INCLUDE {p.tag}_{isa.name}.asm' for p in PASSES for isa in ISAS], ['END'])
    for fname, sfx, isas, isa0 in [('conv_all_linux_x86_64.s', '_linux.s', ISAS, None),
                                   ('conv_all_macos_x86_64.s', '_macos.s', ISAS, None),
                                   ('conv_all_linux_aarch64.s', '_linux.s', ISAS_ARM, None),
                                   ('conv_all_macos_aarch64.s', '_macos.s', ISAS_ARM, None)]:
        emit_master(fname,
                    '# master TU: assembles all kernels for this target/arch into one object '
                    '(assemble with -I<generated dir>)',
                    [f'.include "{p.tag}_{isa.name}{sfx}"' for p in PASSES for isa in isas], [])


if __name__ == '__main__':
    main()
