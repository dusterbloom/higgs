import unittest
from unittest.mock import patch

import numpy as np

from benchmark import memory_preflight, recurrence_reference, render_kernel, run_paired


class BenchmarkTests(unittest.TestCase):
    def test_launch_variant_rejects_unknown_threadgroup(self):
        with self.assertRaises(ValueError):
            render_kernel(3)

    def test_memory_preflight_fails_closed_below_threshold(self):
        result = type("Result", (), {"stdout": "System-wide memory free percentage: 12%\n"})()
        with patch("benchmark.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "only 12% memory free"):
                memory_preflight(30)

    def test_temporal_tile_is_opt_in_source_transform(self):
        plain = render_kernel(4)
        tiled = render_kernel(4, temporal_tile=4)
        self.assertNotIn("unroll_count(4)", plain)
        self.assertIn("#pragma clang loop unroll_count(4)", tiled)
        self.assertEqual(
            plain.replace("// threadgroup_y=4 temporal_tile=0\n", "")
            .replace("for (int t = 0; t < T; ++t) {", ""),
            tiled.replace("// threadgroup_y=4 temporal_tile=4\n", "")
            .replace("#pragma clang loop unroll_count(4)\n", "")
            .replace("for (int t = 0; t < T; ++t) {", ""),
        )

    def test_scalar_reference_returns_every_output_and_final_state(self):
        q = np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[2.0]], [[3.0]]]], dtype=np.float32)
        a = np.zeros((1, 2, 1), dtype=np.float32)
        b = np.zeros((1, 2, 1), dtype=np.float32)
        y, state = recurrence_reference(
            q, k, v, a, b, np.zeros((1,), np.float32), np.zeros((1,), np.float32),
            np.zeros((1, 1, 1, 2), np.float32),
        )
        np.testing.assert_allclose(y.reshape(-1), [1.0, 1.5], rtol=0, atol=1e-7)
        np.testing.assert_allclose(state.reshape(-1), [0.5, 1.5], rtol=0, atol=1e-7)

    def test_paired_launch_uses_each_variants_own_threadgroup(self):
        observed = []

        def kernel(**kwargs):
            observed.append(kwargs["threadgroup"])
            return (None,)

        rows = run_paired(
            kernels={1: kernel, 4: kernel},
            launch_kwargs={1: {"threadgroup": (32, 1, 1)}, 4: {"threadgroup": (32, 4, 1)}},
            candidate=1,
            repeats=2,
            evaluate=lambda _: None,
            clock_ns=iter(range(8)).__next__,
        )
        self.assertEqual(observed, [(32, 4, 1), (32, 1, 1), (32, 1, 1), (32, 4, 1)])
        self.assertEqual([row["actual_threadgroup"] for row in rows], observed)


if __name__ == "__main__":
    unittest.main()
