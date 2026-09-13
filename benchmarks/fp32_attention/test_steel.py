"""CPU contract tests: catch offset/mask, GQA, and launch-grid regressions."""
import importlib.util
import pathlib
import unittest
import numpy as np

PATH = pathlib.Path(__file__).with_name('steel.py')
spec = importlib.util.spec_from_file_location('steel', PATH) if PATH.exists() else None
steel = importlib.util.module_from_spec(spec) if spec else None
if spec:
    spec.loader.exec_module(steel)


class Contracts(unittest.TestCase):
    def test_sampled_positions_and_gqa(self):
        self.assertIsNotNone(steel, 'Steel experiment is missing')
        q = np.zeros((1, 4, 2, 256), dtype=np.float32)
        k = np.zeros((1, 2, 5, 256), dtype=np.float32)
        v = np.broadcast_to(np.array([[0, 2, 4, 6, 8], [10, 12, 14, 16, 18]], dtype=np.float32)[None, :, :, None], k.shape)
        mask = steel.position_mask([0, 3], 5)
        np.testing.assert_array_equal(mask, [[True, False, False, False, False], [True, True, True, True, False]])
        out = steel.reference(q, k, v, [0, 3], 0.0625)
        np.testing.assert_array_equal(out[0, :, :, 0], [[0, 3], [0, 3], [10, 13], [10, 13]])
        with self.assertRaises(ValueError):
            steel.position_mask([5], 5)

    def test_pairs_alternate_and_cover_each_candidate(self):
        self.assertTrue(hasattr(steel, 'paired_schedule'), 'paired schedule is missing')
        schedule = steel.paired_schedule(['q8k16', 'full_sdpa'], 2)
        self.assertEqual(len(schedule), 4)
        for pair in schedule:
            self.assertEqual(pair['order'], ['dense128', pair['candidate']] if pair['repeat'] == 0 else [pair['candidate'], 'dense128'])
        self.assertEqual({p['pair_id'] for p in schedule}, {'q8k16:0', 'q8k16:1', 'full_sdpa:0', 'full_sdpa:1'})
        with self.assertRaises(ValueError):
            steel.paired_schedule(['q8k16'], 3)

    def test_register_cache_launch_does_not_allocate_shared_q(self):
        self.assertIn('qreg32k16', steel.VARIANTS, 'register Q variant is missing')
        for name, grid, threads in [('qreg32k16', (256, 4, 1), 128), ('qreg64k16', (256, 4, 1), 256)]:
            launch = steel.launch_config(name, (1, 4, 33, 256))
            self.assertEqual(tuple(launch['grid']), grid)
            self.assertEqual(launch['threadgroup'], [threads, 1, 1])
            self.assertEqual(launch['threadgroup_bytes'], 20480)
            self.assertEqual(launch['q_cache_fp32_scalars_per_thread'], 64)

    def test_launch_covers_partial_tiles_within_shared_memory(self):
        self.assertIsNotNone(steel, 'Steel experiment is missing')
        for variant, wanted in [('q8k16', (96, 4, 1)), ('q16k8', (128, 4, 1))]:
            launch = steel.launch_config(variant, (1, 4, 17, 256))
            self.assertEqual(tuple(launch['grid']), wanted)
            self.assertLessEqual(launch['threadgroup_bytes'], 32768)
        with self.assertRaises(ValueError):
            steel.launch_config('q8k16', (1, 4, 17, 128))

if __name__ == '__main__':
    unittest.main()
