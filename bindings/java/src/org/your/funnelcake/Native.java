/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

import static java.lang.foreign.MemoryLayout.PathElement.groupElement;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Internal FFM glue: native struct layouts mirroring {@code funnelcake.h},
 * downcall handles, library loading, and read helpers. Package-private.
 *
 * <p>Since there is no struct generator (no jextract), the layouts below are
 * hand-written. They are validated at class-init against the real C
 * {@code sizeof} via {@code fused_*_ctx_sizeof}, so any drift fails loudly
 * here rather than corrupting memory at runtime.
 *
 * <p>The shared library is located via the {@code funnelcake.libdir} system
 * property (default {@code "."}): {@code libfunnelcake.<ext>}, which provides
 * both the scaler API and the binding helpers.
 */
final class Native {
    private Native() {}

    // ----- nested struct layouts (mirror include/funnelcake.h) -----

    static final StructLayout LOG_CONFIG = MemoryLayout.structLayout(
            JAVA_INT.withName("target"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("file"),
            ADDRESS.withName("callback"),
            ADDRESS.withName("callback_ctx")
    ).withName("fused_log_config_t");

    static final StructLayout OUTPUT = MemoryLayout.structLayout(
            JAVA_INT.withName("width"),
            JAVA_INT.withName("height"),
            JAVA_INT.withName("y_stride"),
            JAVA_INT.withName("uv_stride"),
            ADDRESS.withName("plane_y"),
            ADDRESS.withName("plane_u"),
            ADDRESS.withName("plane_v"),
            JAVA_INT.withName("fallback"),
            MemoryLayout.paddingLayout(4)
    ).withName("fused_scale_output_t");

    static final StructLayout TONEMAP = MemoryLayout.structLayout(
            JAVA_INT.withName("curve"),
            JAVA_INT.withName("peak_nits"),
            JAVA_INT.withName("target_nits"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("custom_lut"),
            JAVA_INT.withName("src_range"),
            JAVA_INT.withName("dst_range")
    ).withName("fused_tonemap_config_t");

    static final StructLayout SCALER_CTX = MemoryLayout.structLayout(
            JAVA_INT.withName("src_width"),
            JAVA_INT.withName("src_height"),
            JAVA_INT.withName("src_y_stride"),
            JAVA_INT.withName("src_uv_stride"),
            JAVA_INT.withName("requested_flags"),
            JAVA_INT.withName("options"),
            LOG_CONFIG.withName("log_errors"),
            LOG_CONFIG.withName("log_warnings"),
            JAVA_INT.withName("upscale_flags"),
            JAVA_INT.withName("upscale_tail_1_5x"),
            JAVA_INT.withName("achieved_flags"),
            JAVA_INT.withName("rejected_flags"),
            JAVA_INT.withName("effective_width"),
            JAVA_INT.withName("effective_height"),
            MemoryLayout.sequenceLayout(8, OUTPUT).withName("outputs"),
            JAVA_INT.withName("achieved_upscale_flags"),
            JAVA_INT.withName("achieved_upscale_tail"),
            MemoryLayout.sequenceLayout(6, OUTPUT).withName("upscale_outputs"),
            ADDRESS.withName("_internal")
    ).withName("fused_scaler_ctx_t");

    static final StructLayout HDR_CTX = MemoryLayout.structLayout(
            JAVA_INT.withName("src_width"),
            JAVA_INT.withName("src_height"),
            JAVA_INT.withName("src_y_stride"),
            JAVA_INT.withName("src_uv_stride"),
            JAVA_INT.withName("src_format"),
            JAVA_INT.withName("src_transfer"),
            JAVA_INT.withName("requested_flags"),
            JAVA_INT.withName("hdr_flags"),
            JAVA_INT.withName("sdr_flags"),
            JAVA_INT.withName("options"),
            JAVA_INT.withName("tonemap_1x"),
            MemoryLayout.paddingLayout(4),
            TONEMAP.withName("tonemap"),
            LOG_CONFIG.withName("log_errors"),
            LOG_CONFIG.withName("log_warnings"),
            JAVA_INT.withName("achieved_hdr_flags"),
            JAVA_INT.withName("achieved_sdr_flags"),
            JAVA_INT.withName("rejected_flags"),
            JAVA_INT.withName("effective_width"),
            JAVA_INT.withName("effective_height"),
            MemoryLayout.paddingLayout(4),
            MemoryLayout.sequenceLayout(8, OUTPUT).withName("hdr_outputs"),
            MemoryLayout.sequenceLayout(8, OUTPUT).withName("sdr_outputs"),
            OUTPUT.withName("output_1x"),
            JAVA_INT.withName("upscale_flags"),
            JAVA_INT.withName("upscale_tail_1_5x"),
            JAVA_INT.withName("upscale_sdr_flags"),
            JAVA_INT.withName("upscale_sdr_tail_1_5x"),
            JAVA_INT.withName("achieved_upscale_flags"),
            JAVA_INT.withName("achieved_upscale_tail"),
            JAVA_INT.withName("achieved_upscale_sdr_flags"),
            JAVA_INT.withName("achieved_upscale_sdr_tail"),
            MemoryLayout.sequenceLayout(6, OUTPUT).withName("upscale_hdr_outputs"),
            MemoryLayout.sequenceLayout(6, OUTPUT).withName("upscale_sdr_outputs"),
            ADDRESS.withName("_internal")
    ).withName("fused_hdr_ctx_t");

    // ----- field offsets (derived from the layouts above) -----

    private static long off(StructLayout l, String name) {
        return l.byteOffset(groupElement(name));
    }

    // fused_scale_output_t
    static final long OUT_SIZE     = OUTPUT.byteSize();
    static final long OUT_width    = off(OUTPUT, "width");
    static final long OUT_height   = off(OUTPUT, "height");
    static final long OUT_y_stride = off(OUTPUT, "y_stride");
    static final long OUT_uv_stride = off(OUTPUT, "uv_stride");
    static final long OUT_plane_y  = off(OUTPUT, "plane_y");
    static final long OUT_plane_u  = off(OUTPUT, "plane_u");
    static final long OUT_plane_v  = off(OUTPUT, "plane_v");
    static final long OUT_fallback = off(OUTPUT, "fallback");

    // fused_scaler_ctx_t
    static final long SC_src_width      = off(SCALER_CTX, "src_width");
    static final long SC_src_height     = off(SCALER_CTX, "src_height");
    static final long SC_src_y_stride   = off(SCALER_CTX, "src_y_stride");
    static final long SC_src_uv_stride  = off(SCALER_CTX, "src_uv_stride");
    static final long SC_requested_flags = off(SCALER_CTX, "requested_flags");
    static final long SC_options        = off(SCALER_CTX, "options");
    static final long SC_upscale_flags  = off(SCALER_CTX, "upscale_flags");
    static final long SC_upscale_tail   = off(SCALER_CTX, "upscale_tail_1_5x");
    static final long SC_achieved_flags = off(SCALER_CTX, "achieved_flags");
    static final long SC_effective_width  = off(SCALER_CTX, "effective_width");
    static final long SC_effective_height = off(SCALER_CTX, "effective_height");
    static final long SC_outputs        = off(SCALER_CTX, "outputs");
    static final long SC_achieved_upscale_flags = off(SCALER_CTX, "achieved_upscale_flags");
    static final long SC_achieved_upscale_tail  = off(SCALER_CTX, "achieved_upscale_tail");
    static final long SC_upscale_outputs = off(SCALER_CTX, "upscale_outputs");

    // fused_hdr_ctx_t
    static final long HC_src_width   = off(HDR_CTX, "src_width");
    static final long HC_src_height  = off(HDR_CTX, "src_height");
    static final long HC_src_y_stride = off(HDR_CTX, "src_y_stride");
    static final long HC_src_uv_stride = off(HDR_CTX, "src_uv_stride");
    static final long HC_src_format  = off(HDR_CTX, "src_format");
    static final long HC_src_transfer = off(HDR_CTX, "src_transfer");
    static final long HC_requested_flags = off(HDR_CTX, "requested_flags");
    static final long HC_hdr_flags   = off(HDR_CTX, "hdr_flags");
    static final long HC_sdr_flags   = off(HDR_CTX, "sdr_flags");
    static final long HC_options     = off(HDR_CTX, "options");
    static final long HC_tonemap_1x  = off(HDR_CTX, "tonemap_1x");
    static final long HC_tm_curve    = HDR_CTX.byteOffset(groupElement("tonemap"), groupElement("curve"));
    static final long HC_tm_peak     = HDR_CTX.byteOffset(groupElement("tonemap"), groupElement("peak_nits"));
    static final long HC_tm_target   = HDR_CTX.byteOffset(groupElement("tonemap"), groupElement("target_nits"));
    static final long HC_tm_custom_lut = HDR_CTX.byteOffset(groupElement("tonemap"), groupElement("custom_lut"));
    static final long HC_tm_src_range = HDR_CTX.byteOffset(groupElement("tonemap"), groupElement("src_range"));
    static final long HC_tm_dst_range = HDR_CTX.byteOffset(groupElement("tonemap"), groupElement("dst_range"));
    static final long HC_achieved_hdr_flags = off(HDR_CTX, "achieved_hdr_flags");
    static final long HC_achieved_sdr_flags = off(HDR_CTX, "achieved_sdr_flags");
    static final long HC_effective_width  = off(HDR_CTX, "effective_width");
    static final long HC_effective_height = off(HDR_CTX, "effective_height");
    static final long HC_hdr_outputs = off(HDR_CTX, "hdr_outputs");
    static final long HC_sdr_outputs = off(HDR_CTX, "sdr_outputs");
    static final long HC_output_1x   = off(HDR_CTX, "output_1x");
    static final long HC_upscale_flags = off(HDR_CTX, "upscale_flags");
    static final long HC_upscale_tail  = off(HDR_CTX, "upscale_tail_1_5x");
    static final long HC_upscale_sdr_flags = off(HDR_CTX, "upscale_sdr_flags");
    static final long HC_upscale_sdr_tail  = off(HDR_CTX, "upscale_sdr_tail_1_5x");
    static final long HC_achieved_upscale_flags = off(HDR_CTX, "achieved_upscale_flags");
    static final long HC_achieved_upscale_sdr_flags = off(HDR_CTX, "achieved_upscale_sdr_flags");
    static final long HC_upscale_hdr_outputs = off(HDR_CTX, "upscale_hdr_outputs");
    static final long HC_upscale_sdr_outputs = off(HDR_CTX, "upscale_sdr_outputs");

    /** Slot index of the 1.5x upscale tail (FUSED_UP_IDX_TAIL). */
    static final int UP_IDX_TAIL = 5;

    // ----- downcall handles -----

    private static MethodHandle MH_scaler_init;
    private static MethodHandle MH_scaler_run;
    private static MethodHandle MH_scaler_free;
    private static MethodHandle MH_hdr_init;
    private static MethodHandle MH_hdr_run;
    private static MethodHandle MH_hdr_free;
    private static MethodHandle MH_simd;
    private static MethodHandle MH_version, MH_backend;
    private static MethodHandle MH_scaler_sizeof;
    private static MethodHandle MH_hdr_sizeof;

    static {
        try {
            String libdir = System.getProperty("funnelcake.libdir", ".");
            String ext = osExt();
            // Process-lifetime arena: the libraries stay loaded for the JVM.
            Arena arena = Arena.ofShared();
            Linker linker = Linker.nativeLinker();
            SymbolLookup core = SymbolLookup.libraryLookup(Path.of(libdir, "libfunnelcake." + ext), arena);

            FunctionDescriptor initDesc = FunctionDescriptor.of(JAVA_INT, ADDRESS);
            FunctionDescriptor runDesc = FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS);
            FunctionDescriptor freeDesc = FunctionDescriptor.ofVoid(ADDRESS);

            MH_scaler_init = linker.downcallHandle(core.find("fused_scaler_init").orElseThrow(), initDesc);
            MH_scaler_run = linker.downcallHandle(core.find("fused_scaler_run").orElseThrow(), runDesc);
            MH_scaler_free = linker.downcallHandle(core.find("fused_scaler_free").orElseThrow(), freeDesc);
            MH_hdr_init = linker.downcallHandle(core.find("fused_hdr_init").orElseThrow(), initDesc);
            MH_hdr_run = linker.downcallHandle(core.find("fused_hdr_run").orElseThrow(), runDesc);
            MH_hdr_free = linker.downcallHandle(core.find("fused_hdr_free").orElseThrow(), freeDesc);
            MH_version = linker.downcallHandle(core.find("fused_version").orElseThrow(), FunctionDescriptor.of(ADDRESS));
            MH_backend = linker.downcallHandle(core.find("fused_backend").orElseThrow(), FunctionDescriptor.of(ADDRESS));
            MH_simd = linker.downcallHandle(core.find("fused_simd_available").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT));
            MH_scaler_sizeof = linker.downcallHandle(core.find("fused_scaler_ctx_sizeof").orElseThrow(),
                    FunctionDescriptor.of(JAVA_LONG));
            MH_hdr_sizeof = linker.downcallHandle(core.find("fused_hdr_ctx_sizeof").orElseThrow(),
                    FunctionDescriptor.of(JAVA_LONG));

            long cScaler = (long) MH_scaler_sizeof.invokeExact();
            long cHdr = (long) MH_hdr_sizeof.invokeExact();
            if (cScaler != SCALER_CTX.byteSize()) {
                throw new IllegalStateException("fused_scaler_ctx_t layout mismatch: C sizeof="
                        + cScaler + " Java byteSize=" + SCALER_CTX.byteSize());
            }
            if (cHdr != HDR_CTX.byteSize()) {
                throw new IllegalStateException("fused_hdr_ctx_t layout mismatch: C sizeof="
                        + cHdr + " Java byteSize=" + HDR_CTX.byteSize());
            }

            // Win the library's one-time CPU-detection race before any user
            // thread can call init concurrently.
            int ignored = (int) MH_simd.invokeExact();
        } catch (Throwable t) {
            throw new ExceptionInInitializerError(t);
        }
    }

    // ----- invokers -----

    static int scalerInit(MemorySegment ctx) {
        try {
            return (int) MH_scaler_init.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void scalerRun(MemorySegment ctx, MemorySegment y, MemorySegment u, MemorySegment v) {
        try {
            MH_scaler_run.invokeExact(ctx, y, u, v);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void scalerFree(MemorySegment ctx) {
        try {
            MH_scaler_free.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static int hdrInit(MemorySegment ctx) {
        try {
            return (int) MH_hdr_init.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void hdrRun(MemorySegment ctx, MemorySegment y, MemorySegment u, MemorySegment v) {
        try {
            MH_hdr_run.invokeExact(ctx, y, u, v);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void hdrFree(MemorySegment ctx) {
        try {
            MH_hdr_free.invokeExact(ctx);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static String diagnostic(boolean version) {
        try {
            MemorySegment text = (MemorySegment) (version ? MH_version : MH_backend).invokeExact();
            return text.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) { throw rethrow(t); }
    }

    static int simdAvailable() {
        try {
            return (int) MH_simd.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // ----- read helpers -----

    /**
     * Reinterpret a plane pointer at {@code off} to a readable segment scoped to
     * {@code arena}, or NULL. Scoping to the scaler's arena means that once the
     * scaler is closed, accessing this segment throws IllegalStateException
     * rather than reading freed memory.
     */
    private static MemorySegment plane(Arena arena, MemorySegment ctx, long off, long size) {
        MemorySegment p = ctx.get(ADDRESS, off);
        return p.address() == 0 ? MemorySegment.NULL : p.reinterpret(size, arena, null).asReadOnly();
    }

    static Output readOutput(Arena arena, MemorySegment ctx, long base) {
        int w = ctx.get(JAVA_INT, base + OUT_width);
        int h = ctx.get(JAVA_INT, base + OUT_height);
        int ys = ctx.get(JAVA_INT, base + OUT_y_stride);
        int uvs = ctx.get(JAVA_INT, base + OUT_uv_stride);
        boolean fb = ctx.get(JAVA_INT, base + OUT_fallback) != 0;
        int chromaH = (h + 1) / 2;
        return new Output(w, h, ys, uvs, fb,
                plane(arena, ctx, base + OUT_plane_y, (long) ys * h),
                plane(arena, ctx, base + OUT_plane_u, (long) uvs * chromaH),
                plane(arena, ctx, base + OUT_plane_v, (long) uvs * chromaH));
    }

    static HdrOutput readHdrOutput(Arena arena, MemorySegment ctx, long base) {
        int w = ctx.get(JAVA_INT, base + OUT_width);
        int h = ctx.get(JAVA_INT, base + OUT_height);
        int ys = ctx.get(JAVA_INT, base + OUT_y_stride);
        int uvs = ctx.get(JAVA_INT, base + OUT_uv_stride);
        boolean fb = ctx.get(JAVA_INT, base + OUT_fallback) != 0;
        int chromaH = (h + 1) / 2;
        return new HdrOutput(w, h, ys, uvs, fb,
                plane(arena, ctx, base + OUT_plane_y, (long) ys * h),
                plane(arena, ctx, base + OUT_plane_u, (long) uvs * chromaH),
                plane(arena, ctx, base + OUT_plane_v, (long) uvs * chromaH));
    }

    // ----- misc -----

    static void validateDimensions(int width, int height) {
        if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0
                || width > Integer.MAX_VALUE / 64 || height > Integer.MAX_VALUE / 64
                || (((long) width * 2 + 31) & ~31L) * height > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("invalid frame dimensions");
        }
    }

    static int align32(int n) {
        return (n + 31) & ~31;
    }

    private static String osExt() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) {
            return "dylib";
        }
        if (os.contains("win")) {
            return "dll";
        }
        return "so";
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException re) {
            return re;
        }
        return new RuntimeException(t);
    }
}
