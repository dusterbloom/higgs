import unittest

import numpy as np

from benchmark import causal_mask, selected_block_ids


class AttentionPrefillTests(unittest.TestCase):
    def test_causal_mask_has_visible_boundary_for_tiled_queries(self):
        mask = causal_mask(query_start=0, rows=4, key_len=8, query_offset=4)
        expected = np.array([
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, False],
            [True, True, True, True, True, True, True, True],
        ])
        np.testing.assert_array_equal(mask, expected)

    def test_interleaved_schedule_keeps_visible_tail_block(self):
        blocks = selected_block_ids(visible_end=1024, density=(3, 8), pattern="interleaved")
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[-1], 7)
        self.assertEqual(blocks, sorted(set(blocks)))


if __name__ == "__main__":
    unittest.main()
