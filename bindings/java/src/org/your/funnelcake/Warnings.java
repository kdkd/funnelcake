/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

/**
 * The composable set of non-fatal conditions reported by a successful scaler
 * init. {@link #perfect()} is true when none are set (a FUSED_OK result).
 *
 * @param bits the raw positive return code from init
 */
public record Warnings(int bits) {
    /** A step fell back to the scalar kernel. */
    public boolean scalar() {
        return (bits & (1 << 0)) != 0;
    }

    /** At least one requested step was rejected. */
    public boolean partial() {
        return (bits & (1 << 1)) != 0;
    }

    /** The source was cropped to satisfy dimension constraints. */
    public boolean cropped() {
        return (bits & (1 << 2)) != 0;
    }

    /** Every requested output was produced with SIMD and no cropping. */
    public boolean perfect() {
        return bits == 0;
    }
}
