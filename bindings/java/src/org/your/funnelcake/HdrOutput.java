/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

import java.lang.foreign.MemorySegment;

/**
 * A read-only view of one 10-bit output plane set. Samples are unsigned 16-bit
 * values (10 significant bits) laid out little-endian. Strides are in bytes.
 * The plane segments alias scaler-owned memory and are valid only until the
 * producing scaler's next {@code run} or {@code close}.
 *
 * @param width    output luma width
 * @param height   output luma height
 * @param yStride  bytes per luma row
 * @param uvStride bytes per chroma row
 * @param fallback true if the scalar kernel was used for this step
 * @param y        luma plane (16-bit samples)
 * @param u        Cb plane
 * @param v        Cr plane
 */
public record HdrOutput(
        int width,
        int height,
        int yStride,
        int uvStride,
        boolean fallback,
        MemorySegment y,
        MemorySegment u,
        MemorySegment v) {
}
