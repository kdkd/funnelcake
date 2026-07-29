# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""Smoke tests for the funnelcake Python binding. Run with:

    make test-python
    # or: PYTHONPATH=bindings/python FUNNELCAKE_LIBDIR=bindings/python \
    #       python3 -m unittest discover -s bindings/python
"""

import array
import unittest

import funnelcake as fc


class TestFunnelcake(unittest.TestCase):
    def test_sdr_round_trip(self):
        """256x256 -> 2x; with SIMD the step must not fall back (proves alignment)."""
        w, h = 256, 256
        with fc.Frame(w, h) as f:
            f.y()[:] = b"\x80" * len(f.y())
            f.u()[:] = b"\x80" * len(f.u())
            f.v()[:] = b"\x80" * len(f.v())

            with fc.Scaler(fc.ScalerConfig(w, h, fc.Scale.X2)) as s:
                s.run(f)
                out = s.output(fc.Scale.X2)
                self.assertIsNotNone(out)
                self.assertEqual((out.width, out.height), (w // 2, h // 2))
                self.assertEqual(out.y_stride % 32, 0)
                self.assertEqual(out.uv_stride % 32, 0)
                self.assertEqual(len(out.y), out.y_stride * out.height)
                if fc.simd_available():
                    self.assertFalse(out.fallback, "alignment broken: fell back to scalar")
                    self.assertTrue(s.warnings.perfect())

    def test_hdr_round_trip(self):
        """10-bit I010 -> 2x with HDR + tone-mapped SDR + a 1:1 copy."""
        w, h = 256, 256
        with fc.HdrFrame(w, h, fc.PixelFormat.I010) as f:
            # Fill luma with a mid 10-bit value (uint16 samples).
            y = f.y()
            y[:] = array.array("H", [512]) * len(y)

            cfg = fc.HdrConfig(
                src_width=w,
                src_height=h,
                format=fc.PixelFormat.I010,
                transfer=fc.Transfer.PQ,
                flags=fc.Scale.X2,
                hdr_flags=fc.Scale.X2,
                sdr_flags=fc.Scale.X2,
                tonemap_1x=True,
                tonemap=fc.TonemapConfig(curve=fc.TonemapCurve.BT2390),
            )
            with fc.HdrScaler(cfg) as s:
                s.run(f)

                hdr = s.hdr_output(fc.Scale.X2)
                self.assertIsNotNone(hdr)
                self.assertEqual((hdr.width, hdr.height), (w // 2, h // 2))
                self.assertEqual(len(hdr.y), (hdr.y_stride // 2) * hdr.height)

                sdr = s.sdr_output(fc.Scale.X2)
                self.assertIsNotNone(sdr)
                self.assertEqual((sdr.width, sdr.height), (w // 2, h // 2))

                one = s.tonemap_1x_output()
                self.assertIsNotNone(one)
                self.assertEqual((one.width, one.height), (w, h))

    def test_invalid_flags(self):
        """Mixing thirds and pow2 families is a hard error."""
        with self.assertRaises(fc.FunnelcakeError) as cm:
            fc.Scaler(fc.ScalerConfig(256, 256, fc.Scale.X2 | fc.Scale.X3))
        self.assertEqual(cm.exception.code, -1)

    def _assert_row_const(self, name, plane, stride, width, height, want):
        for row in (0, height - 1):
            base = row * stride
            for col in range(width):
                self.assertEqual(plane[base + col], want, f"{name} at row {row} col {col}")

    def test_flat_field_values(self):
        """Constant input -> constant output, per plane (exercises the data path)."""
        w, h = 256, 256
        with fc.Frame(w, h) as f:
            f.y()[:] = b"\x80" * len(f.y())
            f.u()[:] = b"\x40" * len(f.u())
            f.v()[:] = b"\xc0" * len(f.v())
            with fc.Scaler(fc.ScalerConfig(w, h, fc.Scale.X2)) as s:
                s.run(f)
                o = s.output(fc.Scale.X2)
                self._assert_row_const("Y", o.y, o.y_stride, o.width, o.height, 0x80)
                self._assert_row_const("U", o.u, o.uv_stride, o.width // 2, o.height // 2, 0x40)
                self._assert_row_const("V", o.v, o.uv_stride, o.width // 2, o.height // 2, 0xC0)

    def test_reuse_scaler(self):
        w, h = 256, 256
        with fc.Frame(w, h) as f, fc.Scaler(fc.ScalerConfig(w, h, fc.Scale.X2)) as s:
            f.y()[:] = b"\x0a" * len(f.y())
            s.run(f)
            self.assertEqual(s.output(fc.Scale.X2).y[0], 10)
            f.y()[:] = b"\xc8" * len(f.y())
            s.run(f)
            self.assertEqual(s.output(fc.Scale.X2).y[0], 200)

    def test_hdr_p010(self):
        """Semi-planar P010 path produces correctly-sized output; v() is None."""
        w, h = 256, 256
        with fc.HdrFrame(w, h, fc.PixelFormat.P010) as f:
            self.assertIsNone(f.v())
            cfg = fc.HdrConfig(
                src_width=w, src_height=h, format=fc.PixelFormat.P010,
                transfer=fc.Transfer.PQ, flags=fc.Scale.X2, hdr_flags=fc.Scale.X2,
            )
            with fc.HdrScaler(cfg) as s:
                s.run(f)
                hdr = s.hdr_output(fc.Scale.X2)
                self.assertEqual((hdr.width, hdr.height), (w // 2, h // 2))

    def test_custom_lut_length(self):
        """A wrong-sized custom LUT is rejected before it reaches C."""
        cfg = fc.HdrConfig(
            src_width=256, src_height=256, flags=fc.Scale.X2, sdr_flags=fc.Scale.X2,
            tonemap=fc.TonemapConfig(curve=fc.TonemapCurve.CUSTOM, custom_lut=bytes(100)),
        )
        with self.assertRaises(ValueError):
            fc.HdrScaler(cfg)

    def test_frame_mismatch(self):
        """A mismatched frame size is rejected, not read out of bounds."""
        with fc.Scaler(fc.ScalerConfig(256, 256, fc.Scale.X2)) as s, fc.Frame(128, 128) as f:
            with self.assertRaises(ValueError):
                s.run(f)

    def test_use_after_close(self):
        """A closed scaler raises instead of passing a null context to C."""
        s = fc.Scaler(fc.ScalerConfig(256, 256, fc.Scale.X2))
        s.close()
        with fc.Frame(256, 256) as f:
            with self.assertRaises(RuntimeError):
                s.run(f)

    def test_run_closed_frame(self):
        """A closed frame is rejected, not passed as null planes to C."""
        with fc.Scaler(fc.ScalerConfig(256, 256, fc.Scale.X2)) as s:
            f = fc.Frame(256, 256)
            f.close()
            with self.assertRaises(RuntimeError):
                s.run(f)


if __name__ == "__main__":
    unittest.main()
