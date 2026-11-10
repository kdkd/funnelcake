/*
 * Shared 128-byte power-of-two tree body.
 *
 * This is included in two distinct functions so GCC retains independent
 * PGO histories before identical-code folding. Keep this as statements
 * only; the including functions provide the common parameter names.
 */
    const int emit0 = (active_outputs & (1u << 1)) != 0;
    const int emit1 = (active_outputs & (1u << 3)) != 0;
    const int emit2 = (active_outputs & (1u << 5)) != 0;
    const int dense = (deepest == 0)
                   || (deepest == 1 && emit0)
                   || (deepest == 2 && emit0 && emit1)
                   || (deepest == 3 && emit0 && emit1 && emit2);

#define STEP_PTR(p, step) do { (p) += (step); __asm__("" : "+r"(p)); } while (0)

#define POW2_TREE_128_LOOP(D, DENSE) do {                                      \
    const int group_rows_ = 2 << (D);                                          \
    const int num_groups_ = src_h / group_rows_;                               \
    const ptrdiff_t ss_    = (ptrdiff_t)src_stride;                            \
    const ptrdiff_t ss2_   = ss_ * 2;                                          \
    const ptrdiff_t ds0_   = (ptrdiff_t)dst_strides[0];                        \
    const ptrdiff_t ds1_   = (ptrdiff_t)dst_strides[1];                        \
    const ptrdiff_t ds2_   = (ptrdiff_t)dst_strides[2];                        \
    const ptrdiff_t ds3_   = (ptrdiff_t)dst_strides[3];                        \
    const int e0_ = ((D) == 0 || (DENSE)) ? 1 : emit0;                         \
    const int e1_ = ((D) == 1 || (DENSE)) ? 1 : emit1;                         \
    const int e2_ = ((D) == 2 || (DENSE)) ? 1 : emit2;                         \
    const uint8_t *grp_ = src;                                                 \
    uint8_t *out0_ = e0_ ? dst_planes[0] : NULL;                               \
    uint8_t *out1_ = ((D) >= 1 && e1_) ? dst_planes[1] : NULL;                 \
    uint8_t *out2_ = ((D) >= 2 && e2_) ? dst_planes[2] : NULL;                 \
    uint8_t *out3_ = ((D) >= 3) ? dst_planes[3] : NULL;                        \
    (void)ss2_; (void)ds1_; (void)ds2_; (void)ds3_;                            \
    (void)out1_; (void)out2_; (void)out3_;                                     \
                                                                               \
    for (int g_ = 0; g_ < num_groups_; g_++) {                                 \
        for (int x_ = 0; x_ < src_w; x_ += 128) {                              \
            const uint8_t *sr_ = grp_ + x_;                                    \
            uint8_t *o0_ = e0_ ? (out0_ + ((size_t)x_ >> 1)) : NULL;           \
            if ((D) >= 2) {                                                    \
                __asm__("" : "+r"(sr_));                                       \
                if (e0_) __asm__("" : "+r"(o0_));                              \
            }                                                                  \
            __m512i n0_lo_, n0_hi_;                                            \
                                                                               \
            avg_rows_128_avx512(sr_, sr_ + ss_, 0, &n0_lo_, &n0_hi_);          \
            if (e0_)                                                           \
                store_pow2_128_avx512(n0_lo_, n0_hi_, 0, o0_);                 \
                                                                               \
            if ((D) >= 1) {                                                    \
                __m512i n1_lo_, n1_hi_, p0_lo_, p0_hi_;                        \
                uint8_t *o1_ = e1_ ? (out1_ + ((size_t)x_ >> 2)) : NULL;       \
                if ((D) >= 3 && e1_) __asm__("" : "+r"(o1_));                  \
                if ((D) >= 2) {                                                \
                    STEP_PTR(sr_, ss2_);                                       \
                    if (e0_) STEP_PTR(o0_, ds0_);                              \
                    avg_rows_128_avx512(sr_, sr_ + ss_, 0, &n1_lo_, &n1_hi_);  \
                    if (e0_) store_pow2_128_avx512(n1_lo_, n1_hi_, 0, o0_);    \
                } else {                                                       \
                    avg_rows_128_avx512(sr_ + ss2_, sr_ + ss_ * 3, 0,          \
                                        &n1_lo_, &n1_hi_);                     \
                    if (e0_)                                                   \
                        store_pow2_128_avx512(n1_lo_, n1_hi_, 0, o0_ + ds0_);  \
                }                                                              \
                avg_nodes_128_avx512(n0_lo_, n0_hi_, n1_lo_, n1_hi_,           \
                                     &p0_lo_, &p0_hi_);                        \
                if (e1_)                                                       \
                    store_pow2_128_avx512(p0_lo_, p0_hi_, 1, o1_);             \
                                                                               \
                if ((D) >= 2) {                                                \
                    __m512i n2_lo_, n2_hi_, n3_lo_, n3_hi_;                    \
                    __m512i p1_lo_, p1_hi_, q0_lo_, q0_hi_;                    \
                    STEP_PTR(sr_, ss2_);                                       \
                    if (e0_) STEP_PTR(o0_, ds0_);                              \
                    avg_rows_128_avx512(sr_, sr_ + ss_, 0, &n2_lo_, &n2_hi_);  \
                    if (e0_) store_pow2_128_avx512(n2_lo_, n2_hi_, 0, o0_);    \
                                                                               \
                    STEP_PTR(sr_, ss2_);                                       \
                    if (e0_) STEP_PTR(o0_, ds0_);                              \
                    avg_rows_128_avx512(sr_, sr_ + ss_, 0, &n3_lo_, &n3_hi_);  \
                    if (e0_) store_pow2_128_avx512(n3_lo_, n3_hi_, 0, o0_);    \
                                                                               \
                    avg_nodes_128_avx512(n2_lo_, n2_hi_, n3_lo_, n3_hi_,       \
                                         &p1_lo_, &p1_hi_);                    \
                    if (e1_) {                                                 \
                        if ((D) >= 3) STEP_PTR(o1_, ds1_);                     \
                        else o1_ += ds1_;                                      \
                        store_pow2_128_avx512(p1_lo_, p1_hi_, 1, o1_);         \
                    }                                                          \
                    avg_nodes_128_avx512(p0_lo_, p0_hi_, p1_lo_, p1_hi_,       \
                                         &q0_lo_, &q0_hi_);                    \
                    if (e2_)                                                   \
                        store_pow2_128_avx512(q0_lo_, q0_hi_, 2,               \
                                              out2_ + ((size_t)x_ >> 3));      \
                                                                               \
                    if ((D) >= 3) {                                            \
                        __m512i n4_lo_, n4_hi_, n5_lo_, n5_hi_;                \
                        __m512i n6_lo_, n6_hi_, n7_lo_, n7_hi_;                \
                        __m512i p2_lo_, p2_hi_, p3_lo_, p3_hi_;                \
                        __m512i q1_lo_, q1_hi_, root_lo_, root_hi_;            \
                        STEP_PTR(sr_, ss2_);                                   \
                        if (e0_) STEP_PTR(o0_, ds0_);                          \
                        avg_rows_128_avx512(sr_, sr_ + ss_, 0,                 \
                                            &n4_lo_, &n4_hi_);                 \
                        if (e0_)                                               \
                            store_pow2_128_avx512(n4_lo_, n4_hi_, 0, o0_);     \
                                                                               \
                        STEP_PTR(sr_, ss2_);                                   \
                        if (e0_) STEP_PTR(o0_, ds0_);                          \
                        avg_rows_128_avx512(sr_, sr_ + ss_, 0,                 \
                                            &n5_lo_, &n5_hi_);                 \
                        if (e0_)                                               \
                            store_pow2_128_avx512(n5_lo_, n5_hi_, 0, o0_);     \
                                                                               \
                        avg_nodes_128_avx512(n4_lo_, n4_hi_, n5_lo_, n5_hi_,   \
                                             &p2_lo_, &p2_hi_);                \
                        if (e1_) {                                             \
                            STEP_PTR(o1_, ds1_);                               \
                            store_pow2_128_avx512(p2_lo_, p2_hi_, 1, o1_);     \
                        }                                                      \
                                                                               \
                        STEP_PTR(sr_, ss2_);                                   \
                        if (e0_) STEP_PTR(o0_, ds0_);                          \
                        avg_rows_128_avx512(sr_, sr_ + ss_, 0,                 \
                                            &n6_lo_, &n6_hi_);                 \
                        if (e0_)                                               \
                            store_pow2_128_avx512(n6_lo_, n6_hi_, 0, o0_);     \
                                                                               \
                        STEP_PTR(sr_, ss2_);                                   \
                        if (e0_) STEP_PTR(o0_, ds0_);                          \
                        avg_rows_128_avx512(sr_, sr_ + ss_, 0,                 \
                                            &n7_lo_, &n7_hi_);                 \
                        if (e0_)                                               \
                            store_pow2_128_avx512(n7_lo_, n7_hi_, 0, o0_);     \
                                                                               \
                        avg_nodes_128_avx512(n6_lo_, n6_hi_, n7_lo_, n7_hi_,   \
                                             &p3_lo_, &p3_hi_);                \
                        if (e1_) {                                             \
                            STEP_PTR(o1_, ds1_);                               \
                            store_pow2_128_avx512(p3_lo_, p3_hi_, 1, o1_);     \
                        }                                                      \
                        avg_nodes_128_avx512(p2_lo_, p2_hi_, p3_lo_, p3_hi_,   \
                                             &q1_lo_, &q1_hi_);                \
                        if (e2_)                                               \
                            store_pow2_128_avx512(q1_lo_, q1_hi_, 2,           \
                                                  out2_ + ds2_                 \
                                                  + ((size_t)x_ >> 3));        \
                        avg_nodes_128_avx512(q0_lo_, q0_hi_, q1_lo_, q1_hi_,   \
                                             &root_lo_, &root_hi_);            \
                        store_pow2_128_avx512(root_lo_, root_hi_, 3,           \
                                              out3_ + ((size_t)x_ >> 4));      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        grp_ += ss_ * group_rows_;                                             \
        if (e0_) out0_ += ds0_ * (group_rows_ >> 1);                           \
        if ((D) >= 1 && e1_) out1_ += ds1_ * (group_rows_ >> 2);               \
        if ((D) >= 2 && e2_) out2_ += ds2_ * (group_rows_ >> 3);               \
        if ((D) >= 3) out3_ += ds3_;                                           \
    }                                                                          \
} while (0)

    switch (deepest) {
    case 0:
        POW2_TREE_128_LOOP(0, 1);
        break;
    case 1:
        if (dense) POW2_TREE_128_LOOP(1, 1);
        else       POW2_TREE_128_LOOP(1, 0);
        break;
    case 2:
        if (dense) POW2_TREE_128_LOOP(2, 1);
        else       POW2_TREE_128_LOOP(2, 0);
        break;
    default:
        if (dense) POW2_TREE_128_LOOP(3, 1);
        else       POW2_TREE_128_LOOP(3, 0);
        break;
    }

#undef POW2_TREE_128_LOOP
#undef STEP_PTR
