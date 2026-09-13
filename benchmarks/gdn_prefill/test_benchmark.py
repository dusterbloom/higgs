import unittest

import numpy as np

from benchmark import recurrence_reference, render_kernel, run_paired


class BenchmarkTests(unittest.TestCase):
    def test_launch_variant_rejects_unknown_threadgroup(self):
        with self.assertRaises(ValueError):
            render_kernel(3)

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
