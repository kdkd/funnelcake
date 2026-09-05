import unittest
import funnelcake as fc

class BoundsTests(unittest.TestCase):
    def test_reject_invalid_dimensions(self):
        for w, h in [(0, 2), (1, 2), (128, 3), (2**32 + 128, 64), (32768, 32768)]:
            with self.subTest(w=w, h=h):
                with self.assertRaises(ValueError):
                    fc.Frame(w, h)
                with self.assertRaises(ValueError):
                    fc.HdrFrame(w, h, fc.PixelFormat.P010)
                with self.assertRaises(ValueError):
                    fc.Scaler(fc.ScalerConfig(w, h, fc.Scale.X2))
