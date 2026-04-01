#ifndef TEST_PATTERNS_H
#define TEST_PATTERNS_H

#include <stdint.h>

/* --------------------------------------------------------------------------
 * Test frame: holds a synthetic planar YUV image
 * -------------------------------------------------------------------------- */

typedef struct {
    int      width;
    int      height;
    int      y_stride;
    int      uv_stride;
    int      chroma_format;
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
int  test_frame_create_ex(test_frame_t *frame, int width, int height,
                          int chroma_format, test_pattern_t pattern, uint32_t seed);

/*
 * Free all plane allocations and zero the struct.
 */
void test_frame_free(test_frame_t *frame);

#endif /* TEST_PATTERNS_H */
