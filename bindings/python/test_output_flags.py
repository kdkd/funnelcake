import unittest
import funnelcake as fc

class OutputFlagTests(unittest.TestCase):
    def test_invalid_selectors(self):
        with fc.Scaler(fc.ScalerConfig(256, 64, fc.Scale.X4)) as scaler:
            for flag in [0, -1, fc.Scale.X2 | fc.Scale.X4, (1 << 32) | fc.Scale.X4]:
                self.assertIsNone(scaler.output(flag))
            self.assertEqual(scaler.output(fc.Scale.X4).width, 64)
