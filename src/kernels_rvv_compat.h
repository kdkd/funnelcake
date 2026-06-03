/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/* --------------------------------------------------------------------------
 * kernels_rvv_compat.h - compatibility shim for the RVV intrinsic spec
 *                        transition between GCC 13 (v0.11) and GCC 14+
 *                        (v1.0).
 *
 * Two categories of difference are bridged here:
 *
 * 1. Rounded-average instruction vaaddu and the rounding-mode CSR:
 *
 *    v0.11 (__riscv_v_intrinsic == 11000, GCC 13):
 *      vxrm is a global CSR set via vwrite_csr(RVV_VXRM, ...) at function
 *      entry; vaaddu takes (vs1, vs2, vl).
 *
 *    v1.0  (__riscv_v_intrinsic >= 12000, GCC 14):
 *      vxrm is a per-instruction argument named __RISCV_VXRM_RNU et al.;
 *      vaaddu takes (vs1, vs2, vxrm, vl).
 *
 * 2. Segment loads/stores (vlseg2e8 / vsseg2e8 / vlseg3e8 / vsseg3e8):
 *
 *    v1.0  ships these intrinsics with tuple types (vuint8m1x2_t etc.)
 *    and vget / vcreate accessors.  They map to a single hardware
 *    instruction that interleaves N stride-N elements per cycle - faster
 *    than emitting N separate strided loads.
 *
 *    v0.11 in GCC 13 does NOT ship segment-load intrinsics.  We fall back
 *    to N strided loads/stores, which is what the original kernels did
 *    before this header existed.  Performance is measurably worse on the
 *    horizontal halve, h_filter_3x, and h_filter_1_5x paths under GCC 13.
 *
 * Kernel call sites use the fused_*() wrappers below and stay uniform
 * across both compilers.
 * -------------------------------------------------------------------------- */

#ifndef FUNNELCAKE_KERNELS_RVV_COMPAT_H
#define FUNNELCAKE_KERNELS_RVV_COMPAT_H

#if defined(__riscv) && (__riscv_xlen == 64)

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

#if !defined(__riscv_v_intrinsic) || __riscv_v_intrinsic < 12000
# define FUSED_RVV_HAS_SEGMENT 0
/* #pragma message rather than #warning so -Werror builds don't fail.  This is
 * informational - the build still works, just slower than it would on a
 * compiler with v1.0 intrinsics that includes segment-load support. */
# pragma message "funnelcake RVV: GCC 13 / RVV intrinsic v0.11 detected - segment loads unavailable, falling back to strided loads.  GCC 14+ is recommended for best performance on RISC-V."
#else
# define FUSED_RVV_HAS_SEGMENT 1
#endif

/* --------------------------------------------------------------------------
 * vaaddu / vxrm shim (see file header for spec details).
 * -------------------------------------------------------------------------- */

#if !FUSED_RVV_HAS_SEGMENT
/* v0.11 - global vxrm via CSR write at function entry. */
# define FUSED_RVV_SET_VXRM_RNU()         vwrite_csr(RVV_VXRM, 0)
# define fused_vaaddu_vv_u8m1(a, b, vl)   __riscv_vaaddu_vv_u8m1((a), (b), (vl))
# define fused_vaaddu_vv_u16m1(a, b, vl)  __riscv_vaaddu_vv_u16m1((a), (b), (vl))
#else
/* v1.0+ - vxrm is a per-instruction argument; the global CSR write is a no-op. */
# define FUSED_RVV_SET_VXRM_RNU()         ((void)0)
# define fused_vaaddu_vv_u8m1(a, b, vl)   __riscv_vaaddu_vv_u8m1((a), (b), __RISCV_VXRM_RNU, (vl))
# define fused_vaaddu_vv_u16m1(a, b, vl)  __riscv_vaaddu_vv_u16m1((a), (b), __RISCV_VXRM_RNU, (vl))
#endif

/* --------------------------------------------------------------------------
 * Segment load/store shims.
 *
 * Output vectors are returned via pointer outputs so the caller doesn't
 * need to know about the v1.0 tuple types.  Compilers inline these and
 * the indirection disappears.
 * -------------------------------------------------------------------------- */

#if FUSED_RVV_HAS_SEGMENT

/* 8-bit: load 2 interleaved bytes per output element ("UVUV..." layout). */
static inline void fused_load2_u8m1(const uint8_t *src, size_t vl,
                                    vuint8m1_t *out0, vuint8m1_t *out1)
{
    vuint8m1x2_t pair = __riscv_vlseg2e8_v_u8m1x2(src, vl);
    *out0 = __riscv_vget_v_u8m1x2_u8m1(pair, 0);
    *out1 = __riscv_vget_v_u8m1x2_u8m1(pair, 1);
}

/* 8-bit: load 3 interleaved bytes per output element. */
static inline void fused_load3_u8m1(const uint8_t *src, size_t vl,
                                    vuint8m1_t *out0,
                                    vuint8m1_t *out1,
                                    vuint8m1_t *out2)
{
    vuint8m1x3_t trip = __riscv_vlseg3e8_v_u8m1x3(src, vl);
    *out0 = __riscv_vget_v_u8m1x3_u8m1(trip, 0);
    *out1 = __riscv_vget_v_u8m1x3_u8m1(trip, 1);
    *out2 = __riscv_vget_v_u8m1x3_u8m1(trip, 2);
}

/* 8-bit: interleave 2 vectors and store. */
static inline void fused_store2_u8m1(uint8_t *dst, size_t vl,
                                     vuint8m1_t v0, vuint8m1_t v1)
{
    vuint8m1x2_t pair = __riscv_vcreate_v_u8m1x2(v0, v1);
    __riscv_vsseg2e8_v_u8m1x2(dst, pair, vl);
}

/* 8-bit: interleave 3 vectors and store. */
static inline void fused_store3_u8m1(uint8_t *dst, size_t vl,
                                     vuint8m1_t v0, vuint8m1_t v1, vuint8m1_t v2)
{
    vuint8m1x3_t trip = __riscv_vcreate_v_u8m1x3(v0, v1, v2);
    __riscv_vsseg3e8_v_u8m1x3(dst, trip, vl);
}

/* 16-bit equivalents. */
static inline void fused_load2_u16m1(const uint16_t *src, size_t vl,
                                     vuint16m1_t *out0, vuint16m1_t *out1)
{
    vuint16m1x2_t pair = __riscv_vlseg2e16_v_u16m1x2(src, vl);
    *out0 = __riscv_vget_v_u16m1x2_u16m1(pair, 0);
    *out1 = __riscv_vget_v_u16m1x2_u16m1(pair, 1);
}

static inline void fused_load3_u16m1(const uint16_t *src, size_t vl,
                                     vuint16m1_t *out0,
                                     vuint16m1_t *out1,
                                     vuint16m1_t *out2)
{
    vuint16m1x3_t trip = __riscv_vlseg3e16_v_u16m1x3(src, vl);
    *out0 = __riscv_vget_v_u16m1x3_u16m1(trip, 0);
    *out1 = __riscv_vget_v_u16m1x3_u16m1(trip, 1);
    *out2 = __riscv_vget_v_u16m1x3_u16m1(trip, 2);
}

static inline void fused_store2_u16m1(uint16_t *dst, size_t vl,
                                      vuint16m1_t v0, vuint16m1_t v1)
{
    vuint16m1x2_t pair = __riscv_vcreate_v_u16m1x2(v0, v1);
    __riscv_vsseg2e16_v_u16m1x2(dst, pair, vl);
}

static inline void fused_store3_u16m1(uint16_t *dst, size_t vl,
                                      vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2)
{
    vuint16m1x3_t trip = __riscv_vcreate_v_u16m1x3(v0, v1, v2);
    __riscv_vsseg3e16_v_u16m1x3(dst, trip, vl);
}

#else  /* !FUSED_RVV_HAS_SEGMENT - GCC 13 strided fallbacks. */

static inline void fused_load2_u8m1(const uint8_t *src, size_t vl,
                                    vuint8m1_t *out0, vuint8m1_t *out1)
{
    *out0 = __riscv_vlse8_v_u8m1(src + 0, 2, vl);
    *out1 = __riscv_vlse8_v_u8m1(src + 1, 2, vl);
}

static inline void fused_load3_u8m1(const uint8_t *src, size_t vl,
                                    vuint8m1_t *out0,
                                    vuint8m1_t *out1,
                                    vuint8m1_t *out2)
{
    *out0 = __riscv_vlse8_v_u8m1(src + 0, 3, vl);
    *out1 = __riscv_vlse8_v_u8m1(src + 1, 3, vl);
    *out2 = __riscv_vlse8_v_u8m1(src + 2, 3, vl);
}

static inline void fused_store2_u8m1(uint8_t *dst, size_t vl,
                                     vuint8m1_t v0, vuint8m1_t v1)
{
    __riscv_vsse8_v_u8m1(dst + 0, 2, v0, vl);
    __riscv_vsse8_v_u8m1(dst + 1, 2, v1, vl);
}

static inline void fused_store3_u8m1(uint8_t *dst, size_t vl,
                                     vuint8m1_t v0, vuint8m1_t v1, vuint8m1_t v2)
{
    __riscv_vsse8_v_u8m1(dst + 0, 3, v0, vl);
    __riscv_vsse8_v_u8m1(dst + 1, 3, v1, vl);
    __riscv_vsse8_v_u8m1(dst + 2, 3, v2, vl);
}

static inline void fused_load2_u16m1(const uint16_t *src, size_t vl,
                                     vuint16m1_t *out0, vuint16m1_t *out1)
{
    *out0 = __riscv_vlse16_v_u16m1(src + 0, sizeof(uint16_t) * 2, vl);
    *out1 = __riscv_vlse16_v_u16m1(src + 1, sizeof(uint16_t) * 2, vl);
}

static inline void fused_load3_u16m1(const uint16_t *src, size_t vl,
                                     vuint16m1_t *out0,
                                     vuint16m1_t *out1,
                                     vuint16m1_t *out2)
{
    *out0 = __riscv_vlse16_v_u16m1(src + 0, sizeof(uint16_t) * 3, vl);
    *out1 = __riscv_vlse16_v_u16m1(src + 1, sizeof(uint16_t) * 3, vl);
    *out2 = __riscv_vlse16_v_u16m1(src + 2, sizeof(uint16_t) * 3, vl);
}

static inline void fused_store2_u16m1(uint16_t *dst, size_t vl,
                                      vuint16m1_t v0, vuint16m1_t v1)
{
    __riscv_vsse16_v_u16m1(dst + 0, sizeof(uint16_t) * 2, v0, vl);
    __riscv_vsse16_v_u16m1(dst + 1, sizeof(uint16_t) * 2, v1, vl);
}

static inline void fused_store3_u16m1(uint16_t *dst, size_t vl,
                                      vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2)
{
    __riscv_vsse16_v_u16m1(dst + 0, sizeof(uint16_t) * 3, v0, vl);
    __riscv_vsse16_v_u16m1(dst + 1, sizeof(uint16_t) * 3, v1, vl);
    __riscv_vsse16_v_u16m1(dst + 2, sizeof(uint16_t) * 3, v2, vl);
}

#endif  /* FUSED_RVV_HAS_SEGMENT */

#endif /* __riscv && __riscv_xlen == 64 */

#endif /* FUNNELCAKE_KERNELS_RVV_COMPAT_H */
