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
    public MemorySegment yRow(int row) {
        if (row < 0 || row >= height) throw new IndexOutOfBoundsException(row);
        return y.asSlice((long) row * yStride, (long) (width) * 2).asReadOnly();
    }
    public MemorySegment uRow(int row) {
        if (row < 0 || row >= (height + 1) / 2) throw new IndexOutOfBoundsException(row);
        return u.asSlice((long) row * uvStride, (long) (width / 2) * 2).asReadOnly();
    }
    public MemorySegment vRow(int row) {
        if (row < 0 || row >= (height + 1) / 2) throw new IndexOutOfBoundsException(row);
        return v.asSlice((long) row * uvStride, (long) (width / 2) * 2).asReadOnly();
    }
    /** Independent tightly packed copy backed by Java arrays. */
    public HdrOutput copy() {
        int ch = (height + 1) / 2;
        MemorySegment py = MemorySegment.ofArray(new byte[Math.multiplyExact(width * 2, height)]);
        MemorySegment pu = MemorySegment.ofArray(new byte[Math.multiplyExact(width / 2 * 2, ch)]);
        MemorySegment pv = MemorySegment.ofArray(new byte[Math.multiplyExact(width / 2 * 2, ch)]);
        for (int r=0; r<height; ++r) py.asSlice((long) r * width * 2, (long) width * 2).copyFrom(yRow(r));
        for (int r=0; r<ch; ++r) {
            pu.asSlice((long) r * (width / 2) * 2, (long) (width / 2) * 2).copyFrom(uRow(r));
            pv.asSlice((long) r * (width / 2) * 2, (long) (width / 2) * 2).copyFrom(vRow(r));
        }
        return new HdrOutput(width, height, width * 2, width / 2 * 2, fallback,
                py.asReadOnly(), pu.asReadOnly(), pv.asReadOnly());
    }
}
