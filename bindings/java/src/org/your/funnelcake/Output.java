/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

import java.lang.foreign.MemorySegment;

/**
 * A read-only view of one 8-bit output plane set. The plane segments alias
 * memory owned by the producing scaler and are valid only until that scaler's
 * next {@code run} or {@code close}. Each plane is a native segment sized to
 * {@code stride * height}; copy out anything you need to retain.
 *
 * @param width    output luma width
 * @param height   output luma height
 * @param yStride  bytes per luma row
 * @param uvStride bytes per chroma row
 * @param fallback true if the scalar kernel was used for this step
 * @param y        luma plane (8-bit samples)
 * @param u        Cb plane
 * @param v        Cr plane
 */
public record Output(
        int width,
        int height,
        int yStride,
        int uvStride,
        boolean fallback,
        MemorySegment y,
        MemorySegment u,
        MemorySegment v) {
}
