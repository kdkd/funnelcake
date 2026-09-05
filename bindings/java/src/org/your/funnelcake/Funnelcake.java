/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

/**
 * Constants and capability queries for the funnelcake binding.
 *
 * <p>Scale/upscale/option values are bitmasks, combined with {@code |}. The
 * pixel-format, transfer, range and tone-map values are plain enumerated ints
 * passed to the relevant config records. They mirror the {@code FUSED_*}
 * macros in {@code funnelcake.h} exactly.
 */
public final class Funnelcake {
    private Funnelcake() {}

    // Downscale step flags. A single scaler must use flags from ONE family.
    public static final int SCALE_1_5X = 1 << 0; // thirds family
    public static final int SCALE_2X   = 1 << 1; // pow2 family
    public static final int SCALE_3X   = 1 << 2; // thirds family
    public static final int SCALE_4X   = 1 << 3; // pow2 family
    public static final int SCALE_6X   = 1 << 4; // thirds family
    public static final int SCALE_8X   = 1 << 5; // pow2 family
    public static final int SCALE_12X  = 1 << 6; // thirds family
    public static final int SCALE_16X  = 1 << 7; // pow2 family

    public static final int SCALE_THIRDS_MASK = SCALE_1_5X | SCALE_3X | SCALE_6X | SCALE_12X;
    public static final int SCALE_POW2_MASK   = SCALE_2X | SCALE_4X | SCALE_8X | SCALE_16X;

    // Upscale cascade levels. The requested set must be a contiguous prefix.
    public static final int UPSCALE_2X  = 1 << 0;
    public static final int UPSCALE_4X  = 1 << 1;
    public static final int UPSCALE_8X  = 1 << 2;
    public static final int UPSCALE_16X = 1 << 3;
    public static final int UPSCALE_32X = 1 << 4;

    // Option flags.
    public static final int OPT_NO_CROP     = 1 << 0;
    public static final int OPT_NO_FALLBACK = 1 << 1;

    // 10-bit input pixel formats.
    public static final int PIX_I010 = 0; // 4:2:0 planar
    public static final int PIX_P010 = 1; // 4:2:0 semi-planar (interleaved UV)
    public static final int PIX_I210 = 2; // 4:2:2 planar (decimated to 4:2:0)
    public static final int PIX_P210 = 3; // 4:2:2 semi-planar (decimated to 4:2:0)

    // Transfer functions.
    public static final int TRC_PQ  = 0; // SMPTE ST 2084 (HDR10)
    public static final int TRC_HLG = 1; // Hybrid Log-Gamma

    // Quantization ranges.
    public static final int RANGE_LIMITED = 0; // video range (default)
    public static final int RANGE_FULL    = 1; // full / PC range

    // Tone-mapping curve presets.
    public static final int TONEMAP_HABLE    = 0; // filmic (default)
    public static final int TONEMAP_REINHARD = 1; // simple, lower contrast
    public static final int TONEMAP_BT2390   = 2; // broadcast reference
    public static final int TONEMAP_CUSTOM   = 3; // use TonemapConfig.customLut

    /**
     * Reports whether the scalers will use vectorized kernels on this machine.
     * When {@code false}, every output's {@code fallback} is true and a scalar
     * warning is expected.
     */
    public static boolean simdAvailable() {
        return Native.simdAvailable() != 0;
    }
    /** Version of the loaded native library. */
    public static String version() { return Native.diagnostic(true); }
    /** Preferred backend; individual outputs can use scalar fallback. */
    public static String backend() { return Native.diagnostic(false); }
}
