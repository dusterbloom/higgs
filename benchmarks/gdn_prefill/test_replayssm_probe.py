import unittest

import numpy as np

from replayssm_probe import (
    affine_chain,
    forward_with_tape,
    gates,
    low_rank_chain,
    replay_scan,
    replay_serial,
    serial_chain,
)


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
        low_rank = low_rank_chain(k, v, q, g, beta, state)
        np.testing.assert_array_equal(serial[0], affine[0])
        np.testing.assert_array_equal(serial[1], affine[1])
        np.testing.assert_array_equal(serial[0], low_rank[0])
        np.testing.assert_array_equal(serial[1], low_rank[1])

    def test_affine_chain_preserves_production_shapes(self):
        rng = np.random.default_rng(4)
        k = rng.normal(size=(2, 2, 4)).astype(np.float32)
        v = rng.normal(size=(2, 4, 3)).astype(np.float32)
        q = rng.normal(size=(2, 2, 4)).astype(np.float32)
        g, beta = gates(np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32))
        out, state = affine_chain(k, v, q, g, beta, np.zeros((4, 3, 4), np.float32))
        self.assertEqual(out.shape, (2, 4, 3))
        self.assertEqual(state.shape, (4, 3, 4))

    def test_replay_scan_matches_serial_replay(self):
        rng = np.random.default_rng(5)
        k = rng.normal(size=(4, 2, 4)).astype(np.float32)
        v = rng.normal(size=(4, 4, 3)).astype(np.float32)
        g, beta = gates(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32))
        state = np.zeros((4, 3, 4), np.float32)
        delta, _ = forward_with_tape(k, v, g, beta, state)
        serial = replay_serial(k, delta, g, state)
        scan = replay_scan(k, delta, g, state)
        np.testing.assert_allclose(serial, scan, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
