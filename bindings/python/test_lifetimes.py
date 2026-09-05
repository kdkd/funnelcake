import gc
import weakref
import unittest
import funnelcake as fc

class LifetimeTests(unittest.TestCase):
    def test_frame_view_retains_storage_after_close(self):
        for ctor in [lambda: fc.Frame(128, 64), lambda: fc.HdrFrame(128, 64, fc.PixelFormat.P010)]:
            frame = ctor()
            view = frame.y()[3:11]
            view[0] = 42
            storage = weakref.ref(frame._storage)
            frame.close()
            del frame
            gc.collect()
            self.assertIsNotNone(storage())
            self.assertEqual(view[0], 42)
            view.release()
            del view
            gc.collect()
            self.assertIsNone(storage())

    def test_output_view_retains_storage(self):
        frame = fc.Frame(128, 64)
        frame.y()[:] = bytes([80]) * (128 * 64)
        scaler = fc.Scaler(fc.ScalerConfig(128, 64, fc.Scale.X2))
        scaler.run(frame)
        view = scaler.output(fc.Scale.X2).y[0:8]
        storage = weakref.ref(scaler._storage)
        self.assertTrue(view.readonly)
        scaler.close()
        del scaler
        gc.collect()
        self.assertIsNotNone(storage())
        self.assertEqual(view[0], 80)
        view.release()
        del view
        gc.collect()
        self.assertIsNone(storage())
        frame.close()

    def test_metadata_is_read_only(self):
        with fc.Frame(128, 64) as frame:
            with self.assertRaises(AttributeError):
                frame.height = 128
