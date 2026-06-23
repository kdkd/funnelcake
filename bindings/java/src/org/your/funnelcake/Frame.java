/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * An 8-bit I420 (YUV 4:2:0 planar) input frame. Its three planes are allocated
 * from a dedicated {@link Arena} with 32-byte alignment and 32-byte-aligned
 * strides — exactly what the SIMD kernels require — so a Frame is always safe
 * to hand to {@link Scaler#run}.
 *
 * <p>Fill the planes through {@link #y()}, {@link #u()}, {@link #v()}. Frame is
 * {@link AutoCloseable}; use try-with-resources or call {@link #close()} to free
 * the native memory.
 */
public final class Frame implements AutoCloseable {
    private final Arena arena;
    private final int width;
    private final int height;
    private final int yStride;
    private final int uvStride;
    private final int chromaH;
    final MemorySegment y;
    final MemorySegment u;
    final MemorySegment v;

    /**
     * Allocates an aligned I420 frame. {@code width} and {@code height} should
     * be even (4:2:0 chroma is half-resolution on each axis).
     *
     * @throws IllegalArgumentException if either dimension is not positive
     */
    public Frame(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("frame dimensions must be positive");
        }
        this.arena = Arena.ofShared();
        this.width = width;
        this.height = height;
        this.yStride = Native.align32(width);
        this.uvStride = Native.align32(width / 2);
        this.chromaH = (height + 1) / 2;
        this.y = arena.allocate((long) yStride * height, 32);
        this.u = arena.allocate((long) uvStride * chromaH, 32);
        this.v = arena.allocate((long) uvStride * chromaH, 32);
    }

    public int width() {
        return width;
    }

    public int height() {
        return height;
    }

    public int yStride() {
        return yStride;
    }

    public int uvStride() {
        return uvStride;
    }

    /** Writable luma plane, {@code yStride * height} bytes. */
    public MemorySegment y() {
        return y;
    }

    /** Writable Cb plane, {@code uvStride * chromaHeight} bytes. */
    public MemorySegment u() {
        return u;
    }

    /** Writable Cr plane, {@code uvStride * chromaHeight} bytes. */
    public MemorySegment v() {
        return v;
    }

    @Override
    public void close() {
        arena.close();
    }
}
