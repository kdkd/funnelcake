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

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * An initialized 10-bit scaling / tone-mapping context. Owns its output
 * buffers, scoped to this scaler's lifetime (accessing a view after
 * {@link #close} throws {@code IllegalStateException} rather than reading freed
 * memory). A {@link Cleaner} frees the native memory if {@code close} is never
 * called.
 */
public final class HdrScaler implements AutoCloseable {

    /**
     * Tone-mapping configuration. The default instance ({@link #defaults()})
     * selects the Hable curve, library-default nits, and limited range.
     */
    public record Tonemap(
            int curve,
            int peakNits,
            int targetNits,
            int srcRange,
            int dstRange,
            byte[] customLut) {

        public static Tonemap defaults() {
            return new Tonemap(Funnelcake.TONEMAP_HABLE, 0, 0,
                    Funnelcake.RANGE_LIMITED, Funnelcake.RANGE_LIMITED, null);
        }

        /** A built-in curve with default nits and limited range. */
        public static Tonemap of(int curve) {
            return new Tonemap(curve, 0, 0,
                    Funnelcake.RANGE_LIMITED, Funnelcake.RANGE_LIMITED, null);
        }
    }

    /**
     * Configuration for an {@link HdrScaler}. {@code hdrFlags} and
     * {@code sdrFlags} must each be a subset of {@code flags}.
     */
    public record Config(
            int srcWidth,
            int srcHeight,
            int format,
            int transfer,
            int flags,
            int hdrFlags,
            int sdrFlags,
            int options,
            boolean tonemap1x,
            Tonemap tonemap,
            int upscaleFlags,
            boolean upscaleTail15,
            int upscaleSdrFlags,
            boolean upscaleSdrTail15) {

        /** Minimal config: one source, a step set, HDR + SDR copies, default tone map. */
        public Config(int srcWidth, int srcHeight, int format, int transfer,
                      int flags, int hdrFlags, int sdrFlags) {
            this(srcWidth, srcHeight, format, transfer, flags, hdrFlags, sdrFlags,
                    0, false, Tonemap.defaults(), 0, false, 0, false);
        }
    }

    private static final Cleaner CLEANER = Cleaner.create();

    private record FreeState(Arena arena, MemorySegment ctx) implements Runnable {
        @Override
        public void run() {
            Native.hdrFree(ctx);
            arena.close();
        }
    }

    private final Arena arena;
    private final MemorySegment ctx;
    private final Warnings warnings;
    private final int srcWidth;
    private final int srcHeight;
    private final int format;
    private final Cleaner.Cleanable cleanable;

    /**
     * Builds tone-mapping LUTs and allocates outputs.
     *
     * @throws FunnelcakeException      if the library returns a hard error
     * @throws IllegalArgumentException if a custom LUT is not exactly 1024 bytes
     */
    public HdrScaler(Config cfg) {
        Native.validateDimensions(cfg.srcWidth(), cfg.srcHeight());
        Tonemap tm = cfg.tonemap() != null ? cfg.tonemap() : Tonemap.defaults();
        if (tm.customLut() != null && tm.customLut().length != 1024) {
            throw new IllegalArgumentException(
                    "custom tone-map LUT must be exactly 1024 bytes, got " + tm.customLut().length);
        }

        this.arena = Arena.ofShared();
        this.ctx = arena.allocate(Native.HDR_CTX); // zero-filled

        boolean semiPlanar = cfg.format() == Funnelcake.PIX_P010 || cfg.format() == Funnelcake.PIX_P210;
        int yStride = Native.align32(cfg.srcWidth() * 2);
        int uvStride = semiPlanar ? yStride : Native.align32((cfg.srcWidth() / 2) * 2);

        ctx.set(JAVA_INT, Native.HC_src_width, cfg.srcWidth());
        ctx.set(JAVA_INT, Native.HC_src_height, cfg.srcHeight());
        ctx.set(JAVA_INT, Native.HC_src_y_stride, yStride);
        ctx.set(JAVA_INT, Native.HC_src_uv_stride, uvStride);
        ctx.set(JAVA_INT, Native.HC_src_format, cfg.format());
        ctx.set(JAVA_INT, Native.HC_src_transfer, cfg.transfer());
        ctx.set(JAVA_INT, Native.HC_requested_flags, cfg.flags());
        ctx.set(JAVA_INT, Native.HC_hdr_flags, cfg.hdrFlags());
        ctx.set(JAVA_INT, Native.HC_sdr_flags, cfg.sdrFlags());
        ctx.set(JAVA_INT, Native.HC_options, cfg.options());
        ctx.set(JAVA_INT, Native.HC_tonemap_1x, cfg.tonemap1x() ? 1 : 0);

        ctx.set(JAVA_INT, Native.HC_tm_curve, tm.curve());
        ctx.set(JAVA_INT, Native.HC_tm_peak, tm.peakNits());
        ctx.set(JAVA_INT, Native.HC_tm_target, tm.targetNits());
        ctx.set(JAVA_INT, Native.HC_tm_src_range, tm.srcRange());
        ctx.set(JAVA_INT, Native.HC_tm_dst_range, tm.dstRange());
        if (tm.customLut() != null) {
            // Copy into native memory owned by this scaler's arena.
            MemorySegment lut = arena.allocate(tm.customLut().length);
            MemorySegment.copy(tm.customLut(), 0, lut, JAVA_BYTE, 0, tm.customLut().length);
            ctx.set(ADDRESS, Native.HC_tm_custom_lut, lut);
        }

        ctx.set(JAVA_INT, Native.HC_upscale_flags, cfg.upscaleFlags());
        ctx.set(JAVA_INT, Native.HC_upscale_tail, cfg.upscaleTail15() ? 1 : 0);
        ctx.set(JAVA_INT, Native.HC_upscale_sdr_flags, cfg.upscaleSdrFlags());
        ctx.set(JAVA_INT, Native.HC_upscale_sdr_tail, cfg.upscaleSdrTail15() ? 1 : 0);

        int rc = Native.hdrInit(ctx);
        if (rc < 0) {
            arena.close();
            throw new FunnelcakeException(rc);
        }
        this.warnings = new Warnings(rc);
        this.srcWidth = cfg.srcWidth();
        this.srcHeight = cfg.srcHeight();
        this.format = cfg.format();
        this.cleanable = CLEANER.register(this, new FreeState(arena, ctx));
    }

    /** Non-fatal conditions reported by init. */
    public Warnings warnings() {
        return warnings;
    }

    /**
     * Scales and tone-maps one 10-bit frame.
     *
     * @throws IllegalArgumentException if the frame's dimensions or format
     *                                  differ from the scaler's configuration
     */
    public void run(HdrFrame f) {
        if (f.width() != srcWidth || f.height() != srcHeight || f.format() != format) {
            throw new IllegalArgumentException(String.format(
                    "frame %dx%d fmt=%d does not match scaler %dx%d fmt=%d",
                    f.width(), f.height(), f.format(), srcWidth, srcHeight, format));
        }
        Native.hdrRun(ctx, f.y, f.u, f.v);
    }

    public int effectiveWidth() {
        return ctx.get(JAVA_INT, Native.HC_effective_width);
    }

    public int effectiveHeight() {
        return ctx.get(JAVA_INT, Native.HC_effective_height);
    }

    /** The 10-bit HDR output for a downscale flag. */
    public Optional<HdrOutput> hdrOutput(int flag) {
        if (Integer.bitCount(flag) != 1 || ((ctx.get(JAVA_INT, Native.HC_achieved_hdr_flags) & flag) == 0)) {
            return Optional.empty();
        }
        int idx = Integer.numberOfTrailingZeros(flag);
        return Optional.of(Native.readHdrOutput(arena, ctx, Native.HC_hdr_outputs + idx * Native.OUT_SIZE));
    }

    /** The tone-mapped 8-bit output for a downscale flag. */
    public Optional<Output> sdrOutput(int flag) {
        if (Integer.bitCount(flag) != 1 || ((ctx.get(JAVA_INT, Native.HC_achieved_sdr_flags) & flag) == 0)) {
            return Optional.empty();
        }
        int idx = Integer.numberOfTrailingZeros(flag);
        return Optional.of(Native.readOutput(arena, ctx, Native.HC_sdr_outputs + idx * Native.OUT_SIZE));
    }

    /** The 1:1 tone-mapped SDR copy, if it was produced. */
    public Optional<Output> tonemap1xOutput() {
        if (ctx.get(ADDRESS, Native.HC_output_1x + Native.OUT_plane_y).address() == 0) {
            return Optional.empty();
        }
        return Optional.of(Native.readOutput(arena, ctx, Native.HC_output_1x));
    }

    /** A 10-bit upscale-cascade output for a flag. */
    public Optional<HdrOutput> upscaleHdrOutput(int flag) {
        if (Integer.bitCount(flag) != 1 || ((ctx.get(JAVA_INT, Native.HC_achieved_upscale_flags) & flag) == 0)) {
            return Optional.empty();
        }
        int idx = Integer.numberOfTrailingZeros(flag);
        return Optional.of(Native.readHdrOutput(arena, ctx, Native.HC_upscale_hdr_outputs + idx * Native.OUT_SIZE));
    }

    /** A tone-mapped 8-bit upscale-cascade output for a flag. */
    public Optional<Output> upscaleSdrOutput(int flag) {
        if (Integer.bitCount(flag) != 1 || ((ctx.get(JAVA_INT, Native.HC_achieved_upscale_sdr_flags) & flag) == 0)) {
            return Optional.empty();
        }
        int idx = Integer.numberOfTrailingZeros(flag);
        return Optional.of(Native.readOutput(arena, ctx, Native.HC_upscale_sdr_outputs + idx * Native.OUT_SIZE));
    }

    /** Frees all native buffers. Idempotent. */
    @Override
    public void close() {
        cleanable.clean();
    }
}
