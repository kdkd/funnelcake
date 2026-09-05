/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.Optional;

import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * An initialized 8-bit scaling context. Owns its output buffers, which are
 * scoped to this scaler's lifetime: the {@link Output} segments are valid until
 * {@link #close} (after which accessing them throws {@code IllegalStateException}
 * rather than reading freed memory). A {@link Cleaner} frees the native memory
 * if {@code close} is never called, but prefer try-with-resources.
 */
public final class Scaler implements AutoCloseable {

    /**
     * Configuration for a {@link Scaler}. Source strides are derived from
     * {@code srcWidth}, so a {@link Frame} built with the same width/height
     * matches automatically.
     */
    public record Config(
            int srcWidth,
            int srcHeight,
            int flags,
            int upscaleFlags,
            boolean upscaleTail15,
            int options) {

        /** Convenience for a plain downscale with no upscaling or options. */
        public Config(int srcWidth, int srcHeight, int flags) {
            this(srcWidth, srcHeight, flags, 0, false, 0);
        }
    }

    private static final Cleaner CLEANER = Cleaner.create();

    // Cleanup state holds only native resources (no reference to Scaler), so the
    // Cleaner can run it once the Scaler becomes unreachable.
    private record FreeState(Arena arena, MemorySegment ctx) implements Runnable {
        @Override
        public void run() {
            Native.scalerFree(ctx);
            arena.close();
        }
    }

    private final Arena arena;
    private final MemorySegment ctx;
    private final Warnings warnings;
    private final int srcWidth;
    private final int srcHeight;
    private final Cleaner.Cleanable cleanable;

    /**
     * Validates the configuration and allocates output buffers.
     *
     * @throws FunnelcakeException if the library returns a hard error
     */
    public Scaler(Config cfg) {
        Native.validateDimensions(cfg.srcWidth(), cfg.srcHeight());
        this.arena = Arena.ofShared();
        this.ctx = arena.allocate(Native.SCALER_CTX); // zero-filled

        ctx.set(JAVA_INT, Native.SC_src_width, cfg.srcWidth());
        ctx.set(JAVA_INT, Native.SC_src_height, cfg.srcHeight());
        ctx.set(JAVA_INT, Native.SC_src_y_stride, Native.align32(cfg.srcWidth()));
        ctx.set(JAVA_INT, Native.SC_src_uv_stride, Native.align32(cfg.srcWidth() / 2));
        ctx.set(JAVA_INT, Native.SC_requested_flags, cfg.flags());
        ctx.set(JAVA_INT, Native.SC_options, cfg.options());
        ctx.set(JAVA_INT, Native.SC_upscale_flags, cfg.upscaleFlags());
        ctx.set(JAVA_INT, Native.SC_upscale_tail, cfg.upscaleTail15() ? 1 : 0);

        int rc = Native.scalerInit(ctx);
        if (rc < 0) {
            arena.close();
            throw new FunnelcakeException(rc);
        }
        this.warnings = new Warnings(rc);
        this.srcWidth = cfg.srcWidth();
        this.srcHeight = cfg.srcHeight();
        this.cleanable = CLEANER.register(this, new FreeState(arena, ctx));
    }

    /** Non-fatal conditions reported by init. */
    public Warnings warnings() {
        return warnings;
    }

    /**
     * Scales one frame, filling all achieved outputs.
     *
     * @throws IllegalArgumentException if the frame's dimensions differ from the
     *                                  scaler's configured source size
     */
    public void run(Frame f) {
        if (f.width() != srcWidth || f.height() != srcHeight) {
            throw new IllegalArgumentException(String.format(
                    "frame %dx%d does not match scaler source %dx%d",
                    f.width(), f.height(), srcWidth, srcHeight));
        }
        Native.scalerRun(ctx, f.y, f.u, f.v);
    }

    /** Source luma width actually used (after any crop). */
    public int effectiveWidth() {
        return ctx.get(JAVA_INT, Native.SC_effective_width);
    }

    /** Source luma height actually used (after any crop). */
    public int effectiveHeight() {
        return ctx.get(JAVA_INT, Native.SC_effective_height);
    }

    /** Bitmask of downscale steps that were produced. */
    public int achievedFlags() {
        return ctx.get(JAVA_INT, Native.SC_achieved_flags);
    }

    /** The downscale output for a single flag, and whether it was produced. */
    public Optional<Output> output(int flag) {
        if ((achievedFlags() & flag) == 0) {
            return Optional.empty();
        }
        int idx = Integer.numberOfTrailingZeros(flag);
        return Optional.of(Native.readOutput(arena, ctx, Native.SC_outputs + idx * Native.OUT_SIZE));
    }

    /** An upscale-cascade output for a single flag. */
    public Optional<Output> upscaleOutput(int flag) {
        if ((ctx.get(JAVA_INT, Native.SC_achieved_upscale_flags) & flag) == 0) {
            return Optional.empty();
        }
        int idx = Integer.numberOfTrailingZeros(flag);
        return Optional.of(Native.readOutput(arena, ctx, Native.SC_upscale_outputs + idx * Native.OUT_SIZE));
    }

    /** The 1.5x upscale tail output, if produced. */
    public Optional<Output> upscaleTail() {
        if (ctx.get(JAVA_INT, Native.SC_achieved_upscale_tail) == 0) {
            return Optional.empty();
        }
        return Optional.of(Native.readOutput(arena, ctx, Native.SC_upscale_outputs + Native.UP_IDX_TAIL * Native.OUT_SIZE));
    }

    /** Frees all native buffers. Idempotent. */
    @Override
    public void close() {
        cleanable.clean();
    }
}
