import unittest

from analyze_profile import parse_profile, summarize_profile


class ProfileTests(unittest.TestCase):
    def test_decode_is_excluded_and_offsets_follow_real_chunk_lengths(self):
        logs = '\n'.join(
            f'PROFILE: per-layer avg query_tokens={n} gdn_layers=30 fa_layers=10 '
            'gdn_attn_ms="2" gdn_mlp_ms="3" fa_attn_ms="4" fa_mlp_ms="5" est_total_ms="240"'
            for n in (1024, 960, 1)
        )
        rows = parse_profile(logs)
        self.assertEqual([r['query_offset'] for r in rows], [0, 1024])
        self.assertEqual([r['query_tokens'] for r in rows], [1024, 960])

    def test_component_weights_and_ceiling_are_not_layer_averages(self):
        logs = ('PROFILE: per-layer avg query_tokens=1024 gdn_layers=30 fa_layers=10 '
                'gdn_attn_ms="2" gdn_mlp_ms="3" fa_attn_ms="4" fa_mlp_ms="5" est_total_ms="240"')
        summary = summarize_profile(parse_profile(logs))
        self.assertEqual(summary['extrapolated_component_ms']['fa_attn_ms'], 40)
        self.assertAlmostEqual(summary['fa_component_fraction'], 1 / 6)
        self.assertAlmostEqual(summary['optimistic_fa_elimination_speedup'], 1.2)

    def test_missing_component_fails_instead_of_silently_becoming_zero(self):
        with self.assertRaises(ValueError):
            parse_profile('PROFILE: per-layer avg query_tokens=1024 gdn_layers=30 fa_layers=10')


if __name__ == '__main__':
    unittest.main()
