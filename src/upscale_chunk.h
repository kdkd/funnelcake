/*
 * upscale_chunk.h - shared upscale helpers for scalar, NEON, and AVX2.
 *
 * The upscale primitives are:
 *   1. Horizontal 2x: each source pixel becomes two output pixels.
 *          dst[2i]   = src[i]                         (exact copy)
 *          dst[2i+1] = avg(src[i], src[i+1])          (midpoint, bilinear)
 *      At the right edge src[w-1] is replicated as the "next" pixel.
 *
 *   2. Vertical 2x: each source row becomes two output rows.  Row 2r is
 *      a horizontal-doubled copy of src[r]; row 2r+1 is a horizontal-
 *      doubled copy of vert_avg(src[r], src[r+1]).  The bottom edge
 *      replicates src[h-1] as "row h".
 *
 *   3. 1.5x tail (2->3 upscale).  Uses the same 171/85 weights as the
 *      thirds-family downscale blend, traversed in the reverse direction:
 *      downscale blends 3 samples into 2, upscale blends 2 samples into 3.
 *      For source samples A = src[2i], B = src[2i+1], C = src[2i+2]:
 *          dst[3i  ] = A                                 (exact)
 *          dst[3i+1] = (A * 85 + B * 171 + 128) >> 8     (1/3 A, 2/3 B)
 *          dst[3i+2] = (B * 171 + C * 85 + 128) >> 8     (2/3 B, 1/3 C)
 *      At the right edge C is replicated as src[w-1].  The source width
 *      must be even and the output width is in_w * 3 / 2.
 *
 * Bit-depth variants: 8-bit (uint8_t) for SDR, 10-bit (uint16_t) for HDR.
 * The HDR helpers use the same integer math - the 10-bit value range
 * (0..1023) still fits in 16-bit intermediates after the weighted sum.
 */

#ifndef FUNNELCAKE_UPSCALE_CHUNK_H
#define FUNNELCAKE_UPSCALE_CHUNK_H

#include <stdint.h>

/* -----------------------------------------------------------------------
 * Scalar primitives (SDR 8-bit)
 * ----------------------------------------------------------------------- */

/* Rounded average of two bytes (matches vrhaddq_u8 / _mm_avg_epu8). */
static inline uint8_t up_avg_u8(uint8_t a, uint8_t b)
{
    return (uint8_t)(((uint16_t)a + (uint16_t)b + 1) >> 1);
}

/* Bilinear blend at 2/3 (toward b), using 171/85 weights.  Identical to
 * the existing `blend_2_1` helper in kernels_neon.c - duplicated here so
 * the scalar upscale path does not depend on that file. */
static inline uint8_t up_blend_21_u8(uint8_t a, uint8_t b)
{
    return (uint8_t)(((uint16_t)a * 85 + (uint16_t)b * 171 + 128) >> 8);
}

/* Horizontal 2x upscale of one row:
 *   dst[2i]   = src[i]
 *   dst[2i+1] = avg(src[i], src[i+1])      (src[w-1] replicated at edge)
 * dst must have room for 2*w pixels.
 */
static inline void up_h_2x_row_u8(const uint8_t *src, int w, uint8_t *dst)
{
    for (int i = 0; i < w; i++) {
        uint8_t a = src[i];
        uint8_t b = (i + 1 < w) ? src[i + 1] : a;
        dst[2 * i + 0] = a;
        dst[2 * i + 1] = up_avg_u8(a, b);
    }
}

/* Horizontal 1.5x (2:3) upscale of one row:
 *   dst[3i  ] = src[2i]
 *   dst[3i+1] = (src[2i]*85   + src[2i+1]*171 + 128) >> 8
 *   dst[3i+2] = (src[2i+1]*171 + src[2i+2]*85 + 128) >> 8
 * The source width must be even; out_w = in_w * 3 / 2.
 * dst must have room for in_w * 3 / 2 pixels.
 */
static inline void up_h_1_5x_row_u8(const uint8_t *src, int w, uint8_t *dst)
{
    int pairs = w / 2;
    for (int i = 0; i < pairs; i++) {
        uint8_t a = src[2 * i + 0];
        uint8_t b = src[2 * i + 1];
        uint8_t c = (2 * i + 2 < w) ? src[2 * i + 2] : b;
        dst[3 * i + 0] = a;
        dst[3 * i + 1] = up_blend_21_u8(a, b);
        dst[3 * i + 2] = up_blend_21_u8(c, b);
    }
}

/* -----------------------------------------------------------------------
 * Scalar primitives (HDR 10-bit)
 * ----------------------------------------------------------------------- */

static inline uint16_t up_avg_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a + (uint32_t)b + 1) >> 1);
}

static inline uint16_t up_blend_21_u16(uint16_t a, uint16_t b)
{
    /* Same 171/85 weight scheme; values are 10-bit so result stays <=1023. */
    return (uint16_t)(((uint32_t)a * 85 + (uint32_t)b * 171 + 128) >> 8);
}

static inline void up_h_2x_row_u16(const uint16_t *src, int w, uint16_t *dst)
{
    for (int i = 0; i < w; i++) {
        uint16_t a = src[i];
        uint16_t b = (i + 1 < w) ? src[i + 1] : a;
        dst[2 * i + 0] = a;
        dst[2 * i + 1] = up_avg_u16(a, b);
    }
}

static inline void up_h_1_5x_row_u16(const uint16_t *src, int w, uint16_t *dst)
{
    int pairs = w / 2;
    for (int i = 0; i < pairs; i++) {
        uint16_t a = src[2 * i + 0];
        uint16_t b = src[2 * i + 1];
        uint16_t c = (2 * i + 2 < w) ? src[2 * i + 2] : b;
        dst[3 * i + 0] = a;
        dst[3 * i + 1] = up_blend_21_u16(a, b);
        dst[3 * i + 2] = up_blend_21_u16(c, b);
    }
}

#endif /* FUNNELCAKE_UPSCALE_CHUNK_H */
