# Project 03: CUDA → PTX → SASS Pipeline — Analysis Report

## Overview

This project traces CUDA source code through the compilation pipeline:
```
.cu source → PTX (virtual ISA) → SASS (machine code for SM121)
```

We compare optimization levels, kernel variants, and GPU architectures.

---

## Part 1: Annotated PTX for saxpy_basic

Source:
```cpp
__global__ void saxpy_basic(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

PTX (compiled with `-arch=compute_121 -O2`):

```ptx
.visible .entry _Z11saxpy_basicPKfPffi(
    .param .u64 .ptr .align 1 param_0,   // x pointer
    .param .u64 .ptr .align 1 param_1,   // y pointer
    .param .f32 param_2,                  // a (scalar)
    .param .u32 param_3                   // n (count)
)
{
    // Register declarations
    .reg .pred %p<2>;      // 2 predicate registers (for branch)
    .reg .b32  %r<6>;      // 6 32-bit registers (integers)
    .reg .b32  %f<5>;      // 5 32-bit float registers
    .reg .b64  %rd<8>;     // 8 64-bit registers (pointers)

    // Load parameters from parameter memory
    ld.param.u64  %rd1, [param_0];    // rd1 = x (pointer)
    ld.param.u64  %rd2, [param_1];    // rd2 = y (pointer)
    ld.param.f32  %f1,  [param_2];    // f1 = a
    ld.param.u32  %r2,  [param_3];    // r2 = n

    // Compute thread index: i = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32  %r3, %ctaid.x;           // r3 = blockIdx.x
    mov.u32  %r4, %ntid.x;            // r4 = blockDim.x
    mov.u32  %r5, %tid.x;             // r5 = threadIdx.x
    mad.lo.s32  %r1, %r3, %r4, %r5;   // r1 = r3 * r4 + r5 = i

    // Bounds check: if (i >= n) skip to exit
    setp.ge.s32  %p1, %r1, %r2;       // p1 = (i >= n)
    @%p1 bra  $L__BB0_2;               // if p1, branch to exit

    // Convert to global addresses
    cvta.to.global.u64  %rd3, %rd1;   // rd3 = global address of x
    cvta.to.global.u64  %rd4, %rd2;   // rd4 = global address of y

    // Compute byte offset: i * 4 (sizeof(float))
    mul.wide.s32  %rd5, %r1, 4;       // rd5 = i * 4
    add.s64  %rd6, %rd3, %rd5;        // rd6 = &x[i]
    ld.global.f32  %f2, [%rd6];       // f2 = x[i]

    add.s64  %rd7, %rd4, %rd5;        // rd7 = &y[i]
    ld.global.f32  %f3, [%rd7];       // f3 = y[i]

    // Fused multiply-add: y[i] = a * x[i] + y[i]
    fma.rn.f32  %f4, %f1, %f2, %f3;   // f4 = a * f2 + f3

    // Store result
    st.global.f32  [%rd7], %f4;       // y[i] = f4

$L__BB0_2:
    ret;                               // return
}
```

### Key PTX observations:
- **Register allocation**: 2 predicates, 6 int32, 5 float32, 8 int64 = 21 registers
- **FMA**: The compiler uses `fma.rn.f32` (fused multiply-add) — one instruction instead of `mul.f32` + `add.f32`
- **Predicated branch**: `@%p1 bra` — the bounds check uses predication
- **64-bit addressing**: Pointers are 64-bit, byte offsets computed with `mul.wide.s32`

---

## Part 2: Annotated SASS for saxpy_basic (SM121, -Xptxas -O3)

```
/*0000*/  LDC R1, c[0x0][0x37c]           // Load stack pointer from constant memory
/*0010*/  S2R R0, SR_TID.X                // R0 = threadIdx.x (Special Register to Register)
/*0020*/  S2UR UR4, SR_CTAID.X            // UR4 = blockIdx.x (to Uniform Register)
/*0030*/  LDCU UR5, c[0x0][0x394]         // UR5 = n (uniform load from constant)
/*0040*/  LDC R7, c[0x0][0x360]           // R7 = blockDim.x
/*0050*/  IMAD R7, R7, UR4, R0            // R7 = blockDim.x * blockIdx.x + threadIdx.x = i
/*0060*/  ISETP.GE.AND P0, PT, R7, UR5, PT // P0 = (i >= n)
/*0070*/  @P0 EXIT                        // if (i >= n) exit — PREDICATED exit!
/*0080*/  LDC.64 R2, c[0x0][0x380]        // R2 = x pointer (64-bit load)
/*0090*/  LDCU.64 UR4, c[0x0][0x358]     // UR4 = memory descriptor (uniform)
/*00a0*/  LDCU UR6, c[0x0][0x390]        // UR6 = a (scalar, uniform)
/*00b0*/  LDC.64 R4, c[0x0][0x388]       // R4 = y pointer
/*00c0*/  IMAD.WIDE R2, R7, 0x4, R2      // R2 = x + i*4 = &x[i] (32x32→64 multiply-add)
/*00d0*/  LDG.E R2, desc[UR4][R2.64]     // R2 = x[i] (Global Load, 32-bit)
/*00e0*/  IMAD.WIDE R4, R7, 0x4, R4      // R4 = y + i*4 = &y[i]
/*00f0*/  LDG.E R7, desc[UR4][R4.64]     // R7 = y[i]
/*0100*/  FFMA R7, R2, UR6, R7           // R7 = a * x[i] + y[i] (Fused FMA)
/*0110*/  STG.E desc[UR4][R4.64], R7     // y[i] = R7 (Global Store)
/*0120*/  EXIT                            // Kernel exit
/*0130*/  BRA 0x130                       // Padding (never reached)
```

### Key SASS observations:
- **19 instructions** total (vs ~40 in O0)
- **FFMA**: Fused multiply-add in a single instruction
- **@P0 EXIT**: Predicated exit — no branch instruction needed for the bounds check
- **LDG.E / STG.E**: 32-bit load/store (not vectorized — single float)
- **desc[UR4]**: Memory descriptor addressing (SM121 uses descriptor-based addressing)
- **Uniform registers (UR4, UR5, UR6)**: Used for values uniform across all threads (blockIdx, n, a, memory descriptor) — saves regular register pressure
- **IMAD.WIDE**: 32×32→64 integer multiply-add for address computation

---

## Part 3: O0 vs O3 Comparison

| Metric | -Xptxas -O0 | -Xptxas -O3 |
|--------|-------------|-------------|
| Total SASS lines | 1173 | 661 |
| saxpy_basic instructions | ~40 | ~19 |
| Redundant MOV instructions | Many (`MOV R2, R2`) | None |
| Register usage | Higher (no reuse) | Lower (aggressive reuse) |
| Predicated exit | No (uses `@P0 BRA`) | Yes (`@P0 EXIT`) |
| Parameter loading | All params loaded upfront | Lazy loading (only when needed) |
| Dead code | Present (e.g., `MOV.64 R2, R2`) | Eliminated |

### What the optimizer changed:
1. **Eliminated redundant MOVs**: O0 has many `MOV Rx, Rx` (no-op moves) from unoptimized register allocation
2. **Predicated exit**: O3 uses `@P0 EXIT` directly instead of `@P0 BRA` to a separate exit block
3. **Lazy parameter loading**: O3 loads parameters only when needed, O0 loads all upfront
4. **Register reuse**: O3 reuses registers aggressively (e.g., R7 for both index and y[i])
5. **Instruction scheduling**: O3 reorders instructions for better pipeline utilization

---

## Part 4: Kernel Variant Comparison

### saxpy_basic vs saxpy_restrict
The SASS is **identical**. The `__restrict__` hint tells the compiler that x and y don't alias, but for this simple kernel the compiler already generates optimal code. `__restrict__` matters more for complex kernels with multiple loads/stores.

### saxpy_vec4 (float4 vectorized loads)
Key SASS differences:
```
LDG.E.128.CONSTANT R8, desc[UR4][R2.64]   // 128-bit load (4 floats at once!)
LDG.E.128 R12, desc[UR4][R4.64]           // 128-bit load for y
FFMA R8, R8, UR6, R12                      // 4 FFMAs (one per float4 component)
FFMA R9, R9, UR6, R13
FFMA R10, R10, UR6, R14
FFMA R11, R11, UR6, R15
STG.E.128 desc[UR4][R4.64], R8            // 128-bit store (4 floats at once!)
```
- **LDG.E.128**: 128-bit vectorized load — reads 4 floats in one transaction
- **STG.E.128**: 128-bit vectorized store
- **4 FFMA instructions**: One per float4 component
- **Same total instruction count** but 4x fewer memory transactions

### saxpy_shared (shared memory tiling)
- Uses `LDS` (Load Shared) and `STS` (Store Shared) instructions
- Includes `BAR` (barrier) instructions for `__syncthreads()`
- Slower due to shared memory overhead for this simple kernel

### saxpy_launchbounds (256, 4)
- `__launch_bounds__(256, 4)` hints: max 256 threads/block, 4 blocks/SM
- SASS is identical to saxpy_restrict — the hint affects register allocation, not instruction selection
- The compiler may use fewer registers to allow 4 blocks/SM

### saxpy_unrolled (manual unroll)
- The `#pragma unroll` loop is unrolled into 4 separate load/compute/store sequences
- Each iteration has its own predicated bounds check (`@!P1`, `@!P0`)
- More instructions but potentially better instruction-level parallelism

---

## Part 5: Architecture Comparison (sm_80 vs sm_90 vs sm_121)

| Feature | SM80 (Ampere) | SM90 (Hopper) | SM121 (Blackwell) |
|---------|---------------|---------------|-------------------|
| Parameter loading | `c[0x0][offset]` direct | `LDC` + `ULDC` | `LDC` + `LDCU` |
| Uniform regs | No | Yes (UR4+) | Yes (UR4+) |
| Memory addressing | `[R2.64]` direct | `desc[UR4]` | `desc[UR4]` |
| Load instruction | `LDG.E R2, [R2.64]` | `LDG.E R2, desc[UR4][R2.64]` | `LDG.E R2, desc[UR4][R2.64]` |
| Thread ID | `S2R R4, SR_TID.X` | `S2R R0, SR_TID.X` | `S2R R0, SR_TID.X` |
| Block ID | `S2R R4, SR_CTAID.X` | `S2UR UR4, SR_CTAID.X` | `S2UR UR4, SR_CTAID.X` |
| Bounds check | `ISETP.GE.AND P0, PT, R4, c[...], PT` | `ISETP.GE.AND P0, PT, R7, UR4, PT` | `ISETP.GE.AND P0, PT, R7, UR5, PT` |
| Index computation | `IMAD R4, R4, c[0x0][0x0], R3` | `IMAD R7, R7, UR4, R0` | `IMAD R7, R7, UR4, R0` |

### Key architectural differences:
1. **SM80 (Ampere)**: No uniform registers. All values use regular registers. Direct memory addressing `[R2.64]`.
2. **SM90 (Hopper)**: Introduces uniform registers (UR4+) for values uniform across all threads. Uses descriptor-based addressing `desc[UR4]`.
3. **SM121 (Blackwell)**: Same uniform register concept as SM90. Uses `LDCU` (uniform load) for parameters that are the same for all threads. Descriptor-based addressing.

### Why uniform registers matter:
- `blockIdx.x`, `n`, `a`, and memory descriptors are the **same for all threads** in a block
- Storing them in uniform registers (UR) saves regular register (R) pressure
- Uniform registers are shared across the warp, reducing per-thread register count
- This allows higher occupancy (more warps per SM)

---

## Part 6: Performance Results

| Kernel | Time (ms) | BW (GB/s) | Notes |
|--------|-----------|-----------|-------|
| saxpy_basic | 0.010 | 1248 | L2-cached (1MB working set) |
| saxpy_restrict | 0.010 | 1248 | Same as basic (compiler already optimizes) |
| saxpy_vec4 | 0.010 | 1248 | Same speed but 4x fewer memory transactions |
| saxpy_shared | 0.012 | 1037 | Shared memory overhead for simple kernel |
| saxpy_launchbounds | 0.010 | 1248 | Same SASS as basic |
| saxpy_unrolled | 0.026 | 475 | Slower — more instructions, no benefit at this size |

Note: All kernels run at ~1248 GB/s because the 1M-element (4MB) working set fits entirely in the 24MB L2 cache. The differences would appear at larger sizes where DRAM bandwidth is the bottleneck.

---

## Key Takeaways

1. **The compiler is smart**: It automatically uses FMA, predication, and uniform registers. You don't need to manually optimize simple kernels.

2. **`__restrict__` is often a no-op**: For simple kernels, the compiler already knows pointers don't alias. It matters for complex kernels with multiple data streams.

3. **Vectorization works**: `float4` loads produce `LDG.E.128` instructions — 4x wider memory transactions. This is the easiest way to improve memory bandwidth utilization.

4. **`-Xptxas -O3` matters, `-O3` doesn't**: The nvcc `-O` flag controls host code. Use `-Xptxas -O3` to optimize device code.

5. **Uniform registers are a Hopper+ feature**: SM80 (Ampere) doesn't have them. SM90+ uses UR registers for block-uniform values, reducing register pressure.

6. **Descriptor-based addressing**: SM90+ uses `desc[UR]` instead of direct `[R]` addressing. This is related to the memory descriptor system for unified memory.

7. **Shared memory isn't always faster**: For bandwidth-bound kernels that don't reuse data, shared memory adds overhead (barriers, extra load/store) without benefit.

8. **Manual unrolling can hurt**: The compiler already unrolls effectively. Manual unrolling can increase instruction count and register pressure without improving performance.
