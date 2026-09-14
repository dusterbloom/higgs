import unittest

import numpy as np

from replayssm_probe import affine_chain, gates, serial_chain


class ReplaySsmProbeTests(unittest.TestCase):
    def test_l1_affine_is_exact_anchor(self):
        rng = np.random.default_rng(3)
        k = rng.normal(size=(1, 2, 4)).astype(np.float32)
        v = rng.normal(size=(1, 4, 3)).astype(np.float32)
        q = rng.normal(size=(1, 2, 4)).astype(np.float32)
        g, beta = gates(rng.normal(size=(1, 4)).astype(np.float32),
                         rng.normal(size=(1, 4)).astype(np.float32))
        state = np.zeros((4, 3, 4), dtype=np.float32)
        serial = serial_chain(k, v, q, g, beta, state)
        affine = affine_chain(k, v, q, g, beta, state)
        np.testing.assert_array_equal(serial[0], affine[0])
        np.testing.assert_array_equal(serial[1], affine[1])

    def test_affine_chain_preserves_production_shapes(self):
        rng = np.random.default_rng(4)
        k = rng.normal(size=(2, 2, 4)).astype(np.float32)
        v = rng.normal(size=(2, 4, 3)).astype(np.float32)
        q = rng.normal(size=(2, 2, 4)).astype(np.float32)
        g, beta = gates(np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32))
        out, state = affine_chain(k, v, q, g, beta, np.zeros((4, 3, 4), np.float32))
        self.assertEqual(out.shape, (2, 4, 3))
        self.assertEqual(state.shape, (4, 3, 4))


if __name__ == "__main__":
    unittest.main()
