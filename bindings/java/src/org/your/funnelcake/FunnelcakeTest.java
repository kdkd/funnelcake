/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

import java.lang.foreign.MemorySegment;
import java.util.Optional;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Self-contained smoke test (no JUnit dependency). Run via {@code make test-java}.
 * Exits 0 if all checks pass, 1 otherwise.
 */
public final class FunnelcakeTest {

    private static int failures;

    public static void main(String[] args) {
        sdrRoundTrip();
        hdrRoundTrip();
        invalidFlags();
        flatFieldValues();
        reuseScaler();
        hdrP010();
        customLutLength();
        frameMismatch();

        if (failures == 0) {
            System.out.println("OK  org.your.funnelcake  (all checks passed)");
            System.exit(0);
        } else {
            System.out.println("FAIL  " + failures + " check(s) failed");
            System.exit(1);
        }
    }

    /** 256x256 -> 2x downscale; with SIMD the step must not fall back. */
    private static void sdrRoundTrip() {
        final int w = 256, h = 256;
        try (Frame f = new Frame(w, h);
             Scaler s = new Scaler(new Scaler.Config(w, h, Funnelcake.SCALE_2X))) {
            f.y().fill((byte) 128);
            f.u().fill((byte) 128);
            f.v().fill((byte) 128);

            s.run(f);

            Optional<Output> opt = s.output(Funnelcake.SCALE_2X);
            check("SDR 2x produced", opt.isPresent());
            if (opt.isEmpty()) {
                return;
            }
            Output o = opt.get();
            check("SDR 2x width", o.width() == w / 2);
            check("SDR 2x height", o.height() == h / 2);
            check("SDR y stride 32-aligned", o.yStride() % 32 == 0);
            check("SDR uv stride 32-aligned", o.uvStride() % 32 == 0);
            check("SDR Y plane sized", o.y().byteSize() == (long) o.yStride() * o.height());
            if (Funnelcake.simdAvailable()) {
                check("SDR 2x used SIMD (alignment held)", !o.fallback());
                check("SDR result perfect", s.warnings().perfect());
            } else {
                System.out.println("note: SIMD unavailable; scalar path exercised");
            }
        } catch (Throwable t) {
            fail("SDR round-trip threw: " + t);
        }
    }

    /** 10-bit I010 -> 2x with HDR + tone-mapped SDR + 1:1 copy. */
    private static void hdrRoundTrip() {
        final int w = 256, h = 256;
        HdrScaler.Config cfg = new HdrScaler.Config(
                w, h, Funnelcake.PIX_I010, Funnelcake.TRC_PQ,
                Funnelcake.SCALE_2X, Funnelcake.SCALE_2X, Funnelcake.SCALE_2X,
                0, true, HdrScaler.Tonemap.of(Funnelcake.TONEMAP_BT2390),
                0, false, 0, false);
        try (HdrFrame f = new HdrFrame(w, h, Funnelcake.PIX_I010);
             HdrScaler s = new HdrScaler(cfg)) {

            // Fill luma with a mid 10-bit value.
            long lumaSamples = (long) (f.yStride() / 2) * h;
            MemorySegment y = f.y();
            for (long i = 0; i < lumaSamples; i++) {
                y.setAtIndex(JAVA_SHORT, i, (short) 512);
            }

            s.run(f);

            Optional<HdrOutput> hdr = s.hdrOutput(Funnelcake.SCALE_2X);
            check("HDR 2x produced", hdr.isPresent());
            hdr.ifPresent(o -> {
                check("HDR 2x width", o.width() == w / 2);
                check("HDR 2x height", o.height() == h / 2);
                check("HDR Y plane sized", o.y().byteSize() == (long) o.yStride() * o.height());
            });

            Optional<Output> sdr = s.sdrOutput(Funnelcake.SCALE_2X);
            check("SDR(tone-mapped) 2x produced", sdr.isPresent());
            sdr.ifPresent(o -> {
                check("SDR 2x width", o.width() == w / 2);
                check("SDR 2x height", o.height() == h / 2);
            });

            Optional<Output> one = s.tonemap1xOutput();
            check("tonemap 1x produced", one.isPresent());
            one.ifPresent(o -> {
                check("tonemap 1x width", o.width() == w);
                check("tonemap 1x height", o.height() == h);
            });
        } catch (Throwable t) {
            fail("HDR round-trip threw: " + t);
        }
    }

    /** Mixing thirds and pow2 families is a hard error. */
    private static void invalidFlags() {
        try {
            new Scaler(new Scaler.Config(256, 256, Funnelcake.SCALE_2X | Funnelcake.SCALE_3X)).close();
            fail("expected FunnelcakeException for mixed-family flags");
        } catch (FunnelcakeException e) {
            check("invalid flags -> code -1", e.code() == -1);
        } catch (Throwable t) {
            fail("invalid-flags test threw unexpected: " + t);
        }
    }

    /** Constant input must yield constant output, per-plane — exercises the data path. */
    private static void flatFieldValues() {
        final int w = 256, h = 256;
        try (Frame f = new Frame(w, h);
             Scaler s = new Scaler(new Scaler.Config(w, h, Funnelcake.SCALE_2X))) {
            f.y().fill((byte) 128);
            f.u().fill((byte) 64);
            f.v().fill((byte) 192);
            s.run(f);
            Output o = s.output(Funnelcake.SCALE_2X).orElseThrow();
            checkRowConst("Y", o.y(), o.yStride(), o.width(), o.height(), (byte) 128);
            checkRowConst("U", o.u(), o.uvStride(), o.width() / 2, o.height() / 2, (byte) 64);
            checkRowConst("V", o.v(), o.uvStride(), o.width() / 2, o.height() / 2, (byte) 192);
        } catch (Throwable t) {
            fail("flat-field threw: " + t);
        }
    }

    private static void checkRowConst(String name, MemorySegment p, int stride, int width, int height, byte want) {
        for (int row : new int[]{0, height - 1}) {
            long base = (long) row * stride;
            for (int col = 0; col < width; col++) {
                if (p.get(JAVA_BYTE, base + col) != want) {
                    fail(name + " plane row " + row + " col " + col);
                    return;
                }
            }
        }
        check(name + " plane constant", true);
    }

    /** A scaler can be reused across frames. */
    private static void reuseScaler() {
        final int w = 256, h = 256;
        try (Frame f = new Frame(w, h);
             Scaler s = new Scaler(new Scaler.Config(w, h, Funnelcake.SCALE_2X))) {
            f.y().fill((byte) 10);
            s.run(f);
            check("reuse run #1", s.output(Funnelcake.SCALE_2X).orElseThrow().y().get(JAVA_BYTE, 0) == 10);
            f.y().fill((byte) 200);
            s.run(f);
            check("reuse run #2", (s.output(Funnelcake.SCALE_2X).orElseThrow().y().get(JAVA_BYTE, 0) & 0xFF) == 200);
        } catch (Throwable t) {
            fail("reuse threw: " + t);
        }
    }

    /** Semi-planar P010 path produces correctly-sized output. */
    private static void hdrP010() {
        final int w = 256, h = 256;
        try (HdrFrame f = new HdrFrame(w, h, Funnelcake.PIX_P010);
             HdrScaler s = new HdrScaler(new HdrScaler.Config(
                     w, h, Funnelcake.PIX_P010, Funnelcake.TRC_PQ,
                     Funnelcake.SCALE_2X, Funnelcake.SCALE_2X, 0))) {
            s.run(f);
            HdrOutput o = s.hdrOutput(Funnelcake.SCALE_2X).orElseThrow();
            check("P010 2x dims", o.width() == w / 2 && o.height() == h / 2);
        } catch (Throwable t) {
            fail("P010 threw: " + t);
        }
    }

    /** A wrong-sized custom LUT is rejected before it reaches C. */
    private static void customLutLength() {
        try {
            new HdrScaler(new HdrScaler.Config(
                    256, 256, Funnelcake.PIX_I010, Funnelcake.TRC_PQ,
                    Funnelcake.SCALE_2X, 0, Funnelcake.SCALE_2X,
                    0, false,
                    new HdrScaler.Tonemap(Funnelcake.TONEMAP_CUSTOM, 0, 0,
                            Funnelcake.RANGE_LIMITED, Funnelcake.RANGE_LIMITED, new byte[100]),
                    0, false, 0, false)).close();
            fail("expected IllegalArgumentException for short custom LUT");
        } catch (IllegalArgumentException e) {
            check("custom LUT length rejected", true);
        } catch (Throwable t) {
            fail("custom LUT test threw unexpected: " + t);
        }
    }

    /** A mismatched frame size is rejected, not read out of bounds. */
    private static void frameMismatch() {
        try (Scaler s = new Scaler(new Scaler.Config(256, 256, Funnelcake.SCALE_2X));
             Frame f = new Frame(128, 128)) {
            s.run(f);
            fail("expected IllegalArgumentException for frame/scaler mismatch");
        } catch (IllegalArgumentException e) {
            check("frame mismatch rejected", true);
        } catch (Throwable t) {
            fail("frame mismatch test threw unexpected: " + t);
        }
    }

    private static void check(String name, boolean ok) {
        if (ok) {
            System.out.println("  pass: " + name);
        } else {
            fail(name);
        }
    }

    private static void fail(String name) {
        failures++;
        System.out.println("  FAIL: " + name);
    }
}
