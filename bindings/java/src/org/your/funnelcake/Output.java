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
    public MemorySegment yRow(int row) {
        if (row < 0 || row >= height) throw new IndexOutOfBoundsException(row);
        return y.asSlice((long) row * yStride, (long) (width) * 1).asReadOnly();
    }
    public MemorySegment uRow(int row) {
        if (row < 0 || row >= (height + 1) / 2) throw new IndexOutOfBoundsException(row);
        return u.asSlice((long) row * uvStride, (long) (width / 2) * 1).asReadOnly();
    }
    public MemorySegment vRow(int row) {
        if (row < 0 || row >= (height + 1) / 2) throw new IndexOutOfBoundsException(row);
        return v.asSlice((long) row * uvStride, (long) (width / 2) * 1).asReadOnly();
    }
    /** Independent tightly packed copy backed by Java arrays. */
    public Output copy() {
        int ch = (height + 1) / 2;
        MemorySegment py = MemorySegment.ofArray(new byte[Math.multiplyExact(width * 1, height)]);
        MemorySegment pu = MemorySegment.ofArray(new byte[Math.multiplyExact(width / 2 * 1, ch)]);
        MemorySegment pv = MemorySegment.ofArray(new byte[Math.multiplyExact(width / 2 * 1, ch)]);
        for (int r=0; r<height; ++r) py.asSlice((long) r * width * 1, (long) width * 1).copyFrom(yRow(r));
        for (int r=0; r<ch; ++r) {
            pu.asSlice((long) r * (width / 2) * 1, (long) (width / 2) * 1).copyFrom(uRow(r));
            pv.asSlice((long) r * (width / 2) * 1, (long) (width / 2) * 1).copyFrom(vRow(r));
        }
        return new Output(width, height, width * 1, width / 2 * 1, fallback,
                py.asReadOnly(), pu.asReadOnly(), pv.asReadOnly());
    }
}
