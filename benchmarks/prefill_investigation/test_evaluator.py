import unittest

import numpy as np

from evaluator import dense_attention, indexless_mean_attention


class EvaluatorTest(unittest.TestCase):
    def test_dense_mode_matches_causal_oracle_and_ignores_future_keys(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(4, 4, 8)).astype(np.float32)
        k = rng.normal(size=(1, 9, 8)).astype(np.float32)
        v = rng.normal(size=(1, 9, 8)).astype(np.float32)
        positions = np.array([1, 3, 6, 8], dtype=np.int32)

        expected = dense_attention(q, k, v, positions, scale=8**-0.5)
        actual, stats = indexless_mean_attention(
            q, k, v, positions, scale=8**-0.5, alpha=0.0, block_size=4
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(stats["selected_tokens"], stats["visible_tokens"])

        changed_k = k.copy()
        changed_v = v.copy()
        changed_k[:, 7:] += 1000
        changed_v[:, 7:] -= 1000
        changed, _ = indexless_mean_attention(
            q[:3], changed_k, changed_v, positions[:3], scale=8**-0.5,
            alpha=0.0, block_size=4
        )
        np.testing.assert_allclose(changed, expected[:3], rtol=1e-6, atol=1e-6)

    def test_omitted_constant_block_uses_mean_value_and_log_count(self):
        q = np.ones((1, 1, 1), np.float32)
        k = np.array([[[-10.0], [-10.0], [1.0], [1.0], [0.0]]], np.float32)
        v = np.array([[[3.0], [3.0], [7.0], [9.0], [11.0]]], np.float32)
        actual, stats = indexless_mean_attention(
            q, k, v, np.array([4]), scale=1.0, alpha=0.5,
            block_size=2, sink_tokens=0, local_tokens=0,
        )
        pseudo_logits = np.array([-10.0 + np.log(2), 1.0, 1.0, 0.0])
        pseudo_values = np.array([3.0, 7.0, 9.0, 11.0])
        weights = np.exp(pseudo_logits - pseudo_logits.max())
        expected = np.sum(weights * pseudo_values) / np.sum(weights)
        np.testing.assert_allclose(actual[0, 0, 0], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(stats["selected_blocks"], 1)

    def test_forced_sink_local_and_boundary_blocks_are_exact(self):
        q = np.zeros((1, 1, 1), np.float32)
        k = np.zeros((1, 10, 1), np.float32)
        v = np.arange(10, dtype=np.float32).reshape(1, 10, 1)
        _, stats = indexless_mean_attention(
            q, k, v, np.array([9]), scale=1.0, alpha=2.0,
            block_size=2, sink_tokens=2, local_tokens=4,
        )
        self.assertEqual(stats["selected_blocks"], 2)
        self.assertEqual(stats["selected_tokens"], 6)
        self.assertEqual(stats["visible_tokens"], 10)

    def test_density_percentiles_expose_nonuniform_dense_rows(self):
        q = np.zeros((2, 1, 1), np.float32)
        k = np.zeros((1, 10, 1), np.float32)
        v = np.zeros_like(k)
        _, stats = indexless_mean_attention(
            q, k, v, np.array([1, 9]), scale=1.0, alpha=2.0,
            block_size=2, sink_tokens=0, local_tokens=0,
        )
        self.assertAlmostEqual(stats["selected_density"], 1 / 3)
        self.assertAlmostEqual(stats["selected_density_p50"], 0.6)
        self.assertAlmostEqual(stats["selected_density_max"], 1.0)

    def test_gqa_maps_each_query_head_group_to_its_distinct_kv_head(self):
        q = np.zeros((1, 4, 1), np.float32)
        k = np.zeros((2, 5, 1), np.float32)
        v = np.stack([
            np.full((5, 1), 1.0, np.float32),
            np.full((5, 1), 9.0, np.float32),
        ])
        positions = np.array([4])
        expected = np.array([[[1.0], [1.0], [9.0], [9.0]]], np.float32)

        dense = dense_attention(q, k, v, positions, scale=1.0)
        sparse, _ = indexless_mean_attention(
            q, k, v, positions, scale=1.0, alpha=2.0,
            block_size=2, sink_tokens=0, local_tokens=0,
        )
        np.testing.assert_array_equal(dense, expected)
        np.testing.assert_array_equal(sparse, expected)

    def test_positive_alpha_causal_boundary_ignores_future_token(self):
        q = np.ones((1, 2, 2), np.float32)
        k = np.zeros((1, 6, 2), np.float32)
        v = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
        before, _ = indexless_mean_attention(
            q, k, v, np.array([4]), scale=0.5, alpha=0.5,
            block_size=2, sink_tokens=0, local_tokens=0,
        )
        k[:, 5] = 10_000
        v[:, 5] = -10_000
        after, _ = indexless_mean_attention(
            q, k, v, np.array([4]), scale=0.5, alpha=0.5,
            block_size=2, sink_tokens=0, local_tokens=0,
        )
        np.testing.assert_array_equal(after, before)

    def test_invalid_inputs_are_rejected(self):
        q = np.zeros((1, 2, 4), np.float32)
        k = np.zeros((1, 2, 4), np.float32)
        v = np.zeros_like(k)
        with self.assertRaisesRegex(ValueError, "query positions"):
            dense_attention(q, k, v, np.array([2]), 0.5)
        with self.assertRaisesRegex(ValueError, "alpha"):
            indexless_mean_attention(q, k, v, np.array([1]), 0.5, np.nan)
        with self.assertRaisesRegex(ValueError, "block"):
            indexless_mean_attention(q, k, v, np.array([1]), 0.5, 0.1, block_size=0)


if __name__ == "__main__":
    unittest.main()
