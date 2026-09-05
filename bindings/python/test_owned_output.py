import unittest
import funnelcake as fc

class OwnedOutputTests(unittest.TestCase):
    def test_packed_copy(self):
        with fc.Frame(136,64) as frame, fc.Scaler(fc.ScalerConfig(136,64,fc.Scale.X2)) as scaler:
            frame.y()[:] = bytes([91]) * len(frame.y())
            scaler.run(frame)
            out=scaler.output(fc.Scale.X2)
            self.assertEqual(len(out.y_row(0)),out.width)
            owned=out.copy()
            frame.y()[:] = bytes(len(frame.y()))
            scaler.run(frame)
            self.assertTrue(all(x==91 for x in owned.y))
            with self.assertRaises(IndexError):
                out.y_row(out.height)
        self.assertEqual(len(owned.y),owned.width*owned.height)
        self.assertTrue(owned.y.readonly)
