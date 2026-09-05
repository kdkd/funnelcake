import unittest
import funnelcake as fc
class DiagnosticTests(unittest.TestCase):
    def test_loaded_library(self):
        self.assertTrue(fc.version())
        self.assertIn(fc.backend(),("scalar","avx2","avx512","neon","rvv"))
