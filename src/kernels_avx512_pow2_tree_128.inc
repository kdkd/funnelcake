/*
 * Shared 128-byte power-of-two tree body.
 *
 * This is included in two distinct functions so GCC retains independent
 * PGO histories before identical-code folding. Keep this as statements
 * only; the including functions provide the common parameter names.
 */
    const int group_rows = 2 << deepest;
    const int num_groups = src_h / group_rows;
    const int emit0 = (active_outputs & (1u << 1)) != 0;
    const int emit1 = (active_outputs & (1u << 3)) != 0;
    const int emit2 = (active_outputs & (1u << 5)) != 0;
    const int emit3 = (active_outputs & (1u << 7)) != 0;

    for (int g = 0; g < num_groups; g++) {
        const uint8_t *grp = src
            + (size_t)g * (size_t)group_rows * (size_t)src_stride;
        uint8_t *out0 = emit0 ? dst_planes[0]
            + (size_t)g * (size_t)(group_rows >> 1)
              * (size_t)dst_strides[0] : NULL;
        uint8_t *out1 = emit1 ? dst_planes[1]
            + (size_t)g * (size_t)(group_rows >> 2)
              * (size_t)dst_strides[1] : NULL;
        uint8_t *out2 = emit2 ? dst_planes[2]
            + (size_t)g * (size_t)(group_rows >> 3)
              * (size_t)dst_strides[2] : NULL;
        uint8_t *out3 = emit3 ? dst_planes[3]
            + (size_t)g * (size_t)(group_rows >> 4)
              * (size_t)dst_strides[3] : NULL;

        for (int x = 0; x < src_w; x += 128) {
            __m512i n0_lo, n0_hi, n1_lo, n1_hi;
            __m512i p0_lo, p0_hi;

            avg_rows_128_avx512(grp,
                grp + (size_t)src_stride, x, &n0_lo, &n0_hi);
            if (emit0)
                store_pow2_128_avx512(n0_lo, n0_hi, 0,
                    out0 + (size_t)x / 2);
            if (deepest == 0)
                continue;

            avg_rows_128_avx512(
                grp + (size_t)2 * (size_t)src_stride,
                grp + (size_t)3 * (size_t)src_stride,
                x, &n1_lo, &n1_hi);
            if (emit0)
                store_pow2_128_avx512(n1_lo, n1_hi, 0,
                    out0 + (size_t)dst_strides[0] + (size_t)x / 2);
            avg_nodes_128_avx512(n0_lo, n0_hi, n1_lo, n1_hi,
                                 &p0_lo, &p0_hi);
            if (emit1)
                store_pow2_128_avx512(p0_lo, p0_hi, 1,
                    out1 + (size_t)x / 4);
            if (deepest == 1)
                continue;

            __m512i n2_lo, n2_hi, n3_lo, n3_hi;
            __m512i p1_lo, p1_hi, q0_lo, q0_hi;
            avg_rows_128_avx512(
                grp + (size_t)4 * (size_t)src_stride,
                grp + (size_t)5 * (size_t)src_stride,
                x, &n2_lo, &n2_hi);
            if (emit0)
                store_pow2_128_avx512(n2_lo, n2_hi, 0,
                    out0 + (size_t)2 * (size_t)dst_strides[0]
                         + (size_t)x / 2);
            avg_rows_128_avx512(
                grp + (size_t)6 * (size_t)src_stride,
                grp + (size_t)7 * (size_t)src_stride,
                x, &n3_lo, &n3_hi);
            if (emit0)
                store_pow2_128_avx512(n3_lo, n3_hi, 0,
                    out0 + (size_t)3 * (size_t)dst_strides[0]
                         + (size_t)x / 2);
            avg_nodes_128_avx512(n2_lo, n2_hi, n3_lo, n3_hi,
                                 &p1_lo, &p1_hi);
            if (emit1)
                store_pow2_128_avx512(p1_lo, p1_hi, 1,
                    out1 + (size_t)dst_strides[1] + (size_t)x / 4);
            avg_nodes_128_avx512(p0_lo, p0_hi, p1_lo, p1_hi,
                                 &q0_lo, &q0_hi);
            if (emit2)
                store_pow2_128_avx512(q0_lo, q0_hi, 2,
                    out2 + (size_t)x / 8);
            if (deepest == 2)
                continue;

            __m512i n4_lo, n4_hi, n5_lo, n5_hi;
            __m512i n6_lo, n6_hi, n7_lo, n7_hi;
            __m512i p2_lo, p2_hi, p3_lo, p3_hi;
            __m512i q1_lo, q1_hi, root_lo, root_hi;
            avg_rows_128_avx512(
                grp + (size_t)8 * (size_t)src_stride,
                grp + (size_t)9 * (size_t)src_stride,
                x, &n4_lo, &n4_hi);
            if (emit0)
                store_pow2_128_avx512(n4_lo, n4_hi, 0,
                    out0 + (size_t)4 * (size_t)dst_strides[0]
                         + (size_t)x / 2);
            avg_rows_128_avx512(
                grp + (size_t)10 * (size_t)src_stride,
                grp + (size_t)11 * (size_t)src_stride,
                x, &n5_lo, &n5_hi);
            if (emit0)
                store_pow2_128_avx512(n5_lo, n5_hi, 0,
                    out0 + (size_t)5 * (size_t)dst_strides[0]
                         + (size_t)x / 2);
            avg_nodes_128_avx512(n4_lo, n4_hi, n5_lo, n5_hi,
                                 &p2_lo, &p2_hi);
            if (emit1)
                store_pow2_128_avx512(p2_lo, p2_hi, 1,
                    out1 + (size_t)2 * (size_t)dst_strides[1]
                         + (size_t)x / 4);

            avg_rows_128_avx512(
                grp + (size_t)12 * (size_t)src_stride,
                grp + (size_t)13 * (size_t)src_stride,
                x, &n6_lo, &n6_hi);
            if (emit0)
                store_pow2_128_avx512(n6_lo, n6_hi, 0,
                    out0 + (size_t)6 * (size_t)dst_strides[0]
                         + (size_t)x / 2);
            avg_rows_128_avx512(
                grp + (size_t)14 * (size_t)src_stride,
                grp + (size_t)15 * (size_t)src_stride,
                x, &n7_lo, &n7_hi);
            if (emit0)
                store_pow2_128_avx512(n7_lo, n7_hi, 0,
                    out0 + (size_t)7 * (size_t)dst_strides[0]
                         + (size_t)x / 2);
            avg_nodes_128_avx512(n6_lo, n6_hi, n7_lo, n7_hi,
                                 &p3_lo, &p3_hi);
            if (emit1)
                store_pow2_128_avx512(p3_lo, p3_hi, 1,
                    out1 + (size_t)3 * (size_t)dst_strides[1]
                         + (size_t)x / 4);
            avg_nodes_128_avx512(p2_lo, p2_hi, p3_lo, p3_hi,
                                 &q1_lo, &q1_hi);
            if (emit2)
                store_pow2_128_avx512(q1_lo, q1_hi, 2,
                    out2 + (size_t)dst_strides[2] + (size_t)x / 8);
            avg_nodes_128_avx512(q0_lo, q0_hi, q1_lo, q1_hi,
                                 &root_lo, &root_hi);
            if (emit3)
                store_pow2_128_avx512(root_lo, root_hi, 3,
                    out3 + (size_t)x / 16);
        }
    }
