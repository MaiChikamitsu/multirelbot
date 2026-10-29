import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fractions import Fraction
from cli import main, input_path


class CLITests(unittest.TestCase):
    def test_dragged_path(self):
        self.assertEqual(str(input_path('/tmp/test\\ video.mp4')), '/tmp/test video.mp4')
        self.assertEqual(str(input_path("'/tmp/test video.mp4'")), '/tmp/test video.mp4')

    def test_terminal_workflow_and_explicit_end_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'input.mp4'; source.touch()
            answers = [str(source),'会話開始：0:00','','1. 0:01','2. 0:00','','完了','','','2','','']
            info = dict(duration=2, size=(640,360), fps=Fraction(30), audio=False)
            with patch('builtins.input', side_effect=answers), patch('cli.binary'), patch('cli.probe',return_value=info), patch('cli.export') as export, patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(main(),0)
                result = export.call_args.args[2]
                self.assertEqual(result.times,(0,2))
                self.assertEqual(result.hold,10)
                self.assertEqual(export.call_args.args[1],source.with_name('input_評価案内.mp4'))
