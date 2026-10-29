import csv
import tempfile
import unittest
from pathlib import Path

from observer_ratings import load_observers, render_observers


class ObserverRatingsTests(unittest.TestCase):
    def test_observer_format_and_separate_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'answers.csv'
            with source.open('w', newline='') as stream:
                csv.writer(stream).writerows([
                    ['タイムスタンプ', '回答者ID', 'テーマ', '1. AとB', '2. AとB', '3. AとB'],
                    ['', '9001.0', 'theme', '+0.3', '', '-0.2'],
                    ['', '9002', 'theme', '0', '0.1', '0.2']])
            sources = [dict(experiment='test', label=label, theme='theme', survey='answers.csv')
                       for label in ['動画版', '文字版']]
            ratings = load_observers(sources, root)
            self.assertEqual(len(ratings), 10)
            self.assertEqual(ratings[0]['respondent'], '9001')
            self.assertEqual(ratings[0]['human'], .3)
            systems = [dict(experiment='test', run='run', method='LLM-only', index=index,
                            pair='A-B', prediction=0, target='mean') for index in [1, 2]]
            content = render_observers(root, systems, ratings)
            self.assertIn('文字版 ／ 回答者ID 9001', content)
            self.assertIn('動画版 ／ 回答者ID 9001', content)
            self.assertIn('-0.2', content)
            with (root/'observer_comparisons.csv').open(encoding='utf-8-sig') as stream:
                comparisons = list(csv.DictReader(stream))
            self.assertEqual(len(comparisons), 6)
            self.assertTrue(all(row['target'] == 'individual' for row in comparisons))
            with self.assertRaisesRegex(ValueError, 'duplicate observer'):
                load_observers(sources + sources[:1], root)


if __name__ == '__main__':
    unittest.main()
