"""Exact rewrites must retain every real channel, including the last one."""
import unittest
import numpy as np
import shapes

class ShapesTest(unittest.TestCase):
    def test_every_rewrite_preserves_projection(self):
        rng = np.random.default_rng(42)
        x = rng.integers(-2, 3, (7, 64)).astype(np.float32)
        w = rng.integers(-2, 3, (128, 64)).astype(np.float32)
        for variant in shapes.VARIANTS:
            with self.subTest(variant=variant):
                padded_x, weights = shapes.rewrite(x, w, variant)
                actual = np.concatenate([padded_x @ part.T for part in weights], axis=1)[:, :128]
                np.testing.assert_array_equal(actual, x @ w.T)

    def test_notch_lattice_and_safe_rewrites(self):
        self.assertEqual(shapes.coefficient_bytes(2048, 4096), 1048576)
        self.assertEqual(shapes.coefficient_bytes(2080, 4096), 1064960)
        self.assertEqual(shapes.coefficient_bytes(2048, 4160), 1064960)
        self.assertEqual(shapes.coefficient_bytes(2048, 2048), 524288)

    def test_bad_input_fails(self):
        with self.assertRaises(ValueError):
            shapes.rewrite(np.zeros((2, 3)), np.zeros((4, 5)), 'original')
        with self.assertRaises(ValueError):
            shapes.rewrite(np.zeros((2, 3)), np.zeros((4, 3)), 'truncate')

if __name__ == '__main__':
    unittest.main()
