#ifndef TEST_PATTERNS_H
#define TEST_PATTERNS_H

#include <stdint.h>

/* --------------------------------------------------------------------------
 * Test frame: holds a synthetic YUV420 (I420) image
 * -------------------------------------------------------------------------- */

typedef struct {
    int      width;
    int      height;
    int      y_stride;
    int      uv_stride;
    uint8_t *plane_y;
    uint8_t *plane_u;
    uint8_t *plane_v;
} test_frame_t;

/* --------------------------------------------------------------------------
 * Pattern enumeration
 * -------------------------------------------------------------------------- */

typedef enum {
    PATTERN_SOLID        = 0,
    PATTERN_HGRADIENT    = 1,
    PATTERN_VGRADIENT    = 2,
    PATTERN_CHECKERBOARD = 3,
    PATTERN_RANDOM       = 4,
    PATTERN_COUNT        = 5
} test_pattern_t;

/* Human-readable pattern names — defined in test_patterns.c */
extern const char *pattern_names[];

/* --------------------------------------------------------------------------
 * API
 * -------------------------------------------------------------------------- */

/*
 * Allocate and fill a test frame.
 * Returns 0 on success, -1 on allocation failure.
 * seed is used for PATTERN_RANDOM; ignored by other patterns.
 */
int  test_frame_create(test_frame_t *frame, int width, int height,
                       test_pattern_t pattern, uint32_t seed);

/*
 * Free all plane allocations and zero the struct.
 */
void test_frame_free(test_frame_t *frame);

/* --------------------------------------------------------------------------
 * 10-bit HDR frame for testing (I010 layout: separate Y, U, V planes)
 * -------------------------------------------------------------------------- */

typedef struct {
    int       width;
    int       height;
    int       y_stride;      /* bytes per row */
    int       uv_stride;     /* bytes per row */
    uint16_t *plane_y;
    uint16_t *plane_u;
    uint16_t *plane_v;
} test_hdr_frame_t;

/* HDR-specific patterns */
#define PATTERN_PQ_RAMP         (PATTERN_COUNT + 0)   /* PQ-encoded luminance ramp */
#define PATTERN_HDR_HIGHLIGHT   (PATTERN_COUNT + 1)   /* bright spot on mid-tone bg */
#define PATTERN_SATURATED_2020  (PATTERN_COUNT + 2)   /* high-saturation BT.2020 chroma */
#define PATTERN_PQ_COLORBARS    (PATTERN_COUNT + 3)   /* SMPTE bars in PQ BT.2020 YCbCr */
#define PATTERN_HDR_COUNT       (PATTERN_COUNT + 4)

int  test_hdr_frame_create(test_hdr_frame_t *frame, int width, int height,
                           test_pattern_t pattern, uint32_t seed);
void test_hdr_frame_free(test_hdr_frame_t *frame);

/* --------------------------------------------------------------------------
 * P010-format frame (Y planar + UV interleaved)
 * -------------------------------------------------------------------------- */

typedef struct {
    int       width, height;
    int       y_stride, uv_stride;
    uint16_t *plane_y;
    uint16_t *plane_uv;    /* interleaved U0V0U1V1... */
} test_p010_frame_t;

int  test_p010_frame_create(test_p010_frame_t *frame, int width, int height,
                            test_pattern_t pattern, uint32_t seed);
void test_p010_frame_free(test_p010_frame_t *frame);

#endif /* TEST_PATTERNS_H */
