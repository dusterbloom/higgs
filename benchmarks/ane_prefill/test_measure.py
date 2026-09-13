import unittest
from measure import summarize_stream, processed_rate

class MeasureTest(unittest.TestCase):
    def test_rate_uses_processed_tokens(self):
        self.assertAlmostEqual(processed_rate(44992, 467320), 96.27664127364547)
        self.assertIsNone(processed_rate(496, None))
        self.assertIsNone(processed_rate(496, 0))
        with self.assertRaises(ValueError):
            processed_rate(-1, 10)

    def test_error_after_finish_still_fails(self):
        result = summarize_stream([{'event': {'choices': [{'finish_reason': 'stop'}]}},
                                   {'event': {'error': {'message': 'failed'}}}], True)
        self.assertEqual(result['outcome'], 'server_error')

    def test_missing_terminal_timing_stays_null(self):
        result = summarize_stream([
            {'seconds': 1, 'event': {'prompt_progress': {'total': 45003, 'processed': 0, 'time_ms': 12}}},
            {'seconds': 290, 'event': {'choices': [{'delta': {'content': 'ok'}}]}},
            {'seconds': 293, 'event': {'usage': {'prompt_tokens': 45003, 'completion_tokens': 1},
                                     'choices': [{'finish_reason': 'stop'}]}}], True)
        self.assertEqual(result['outcome'], 'finished')
        self.assertIsNone(result['prefill_ms'])
        self.assertEqual(result['ttft_seconds'], 290)
        self.assertEqual(summarize_stream([], False)['outcome'], 'incomplete')

if __name__ == '__main__':
    unittest.main()
