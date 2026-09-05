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
 * A 10-bit input frame (samples in the low 10 bits of each 16-bit element).
 * Layout follows the {@code Funnelcake.PIX_*} format: planar formats
 * (I010/I210) use three planes; semi-planar formats (P010/P210) use a luma
 * plane plus one interleaved-UV plane, and {@link #v()} is {@code NULL}.
 * All planes are 32-byte aligned.
 */
public final class HdrFrame implements AutoCloseable {
    private final Arena arena;
    private final int width;
    private final int height;
    private final int format;
    private final int yStride;
    private final int uvStride;
    private final int chromaH;
    final MemorySegment y;
    final MemorySegment u;
    final MemorySegment v;

    /**
     * Allocates an aligned 10-bit frame for the given {@code Funnelcake.PIX_*}
     * format.
     *
     * @throws IllegalArgumentException if dimensions are not positive or the
     *                                  format is unknown
     */
    public HdrFrame(int width, int height, int format) {
        Native.validateDimensions(width, height);
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("frame dimensions must be positive");
        }
        if (format < Funnelcake.PIX_I010 || format > Funnelcake.PIX_P210) {
            throw new IllegalArgumentException("unknown pixel format: " + format);
        }
        boolean semiPlanar = format == Funnelcake.PIX_P010 || format == Funnelcake.PIX_P210;
        boolean is422 = format == Funnelcake.PIX_I210 || format == Funnelcake.PIX_P210;

        this.arena = Arena.ofShared();
        this.width = width;
        this.height = height;
        this.format = format;
        this.yStride = Native.align32(width * 2);
        this.chromaH = is422 ? height : (height + 1) / 2;

        this.y = arena.allocate((long) yStride * height, 32);
        if (semiPlanar) {
            // Interleaved UV row is the same byte width as a luma row.
            this.uvStride = yStride;
            this.u = arena.allocate((long) uvStride * chromaH, 32);
            this.v = MemorySegment.NULL;
        } else {
            this.uvStride = Native.align32((width / 2) * 2);
            this.u = arena.allocate((long) uvStride * chromaH, 32);
            this.v = arena.allocate((long) uvStride * chromaH, 32);
        }
    }

    public int width() {
        return width;
    }

    public int height() {
        return height;
    }

    public int format() {
        return format;
    }

    public int yStride() {
        return yStride;
    }

    public int uvStride() {
        return uvStride;
    }

    /** Writable luma plane (16-bit samples). */
    public MemorySegment y() {
        return y;
    }

    /** Cb plane (I010/I210) or interleaved UV plane (P010/P210). */
    public MemorySegment u() {
        return u;
    }

    /** Cr plane for planar formats; {@code MemorySegment.NULL} for P010/P210. */
    public MemorySegment v() {
        return v;
    }

    @Override
    public void close() {
        arena.close();
    }
}
