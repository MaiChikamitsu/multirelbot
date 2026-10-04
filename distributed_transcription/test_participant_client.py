"""Hardware-free regression tests: python -m unittest discover -s distributed_transcription."""
import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import participant_client as client
from host_server import ConversationLog, _extract_message


class ClientTests(unittest.TestCase):
    def args(self, *extra):
        with patch('sys.argv', ['participant_client.py', '--speaker', 'Bさん', *extra]):
            return client.parse_args()

    def test_cli_device_and_vad(self):
        args = self.args('--input-device', '1', '--show-level')
        self.assertEqual(args.input_device, 1)
        self.assertIsInstance(args.input_device, int)
        self.assertTrue(args.show_level)
        self.assertFalse(args.whisper_vad_filter)
        self.assertTrue(self.args('--whisper-vad-filter').whisper_vad_filter)
        self.assertFalse(self.args('--no-vad-filter').whisper_vad_filter)

    def test_recording_continues_during_inference_and_failure(self):
        recorded = threading.Event()
        started = threading.Event()
        sd = Mock()
        sd.default.device = (1, 2)
        sd.query_devices.return_value = dict(name='USB', max_input_channels=1,
                                             default_samplerate=48000)
        sd.PortAudioError = RuntimeError
        recognizer = Mock()

        def infer(audio):
            if audio == 0:
                started.set()
                self.assertTrue(recorded.wait(2), 'recording blocked by inference')
                raise RuntimeError('one failed inference')
            return f'text {audio}'

        recognizer.transcribe.side_effect = infer
        recorder = Mock()

        def segments(stop):
            yield 0
            self.assertTrue(started.wait(2))
            yield 1
            yield 2
            recorded.set()

        recorder.iter_segments.side_effect = segments
        sender = Mock()
        with patch.dict('sys.modules', {'sounddevice': sd}), \
                patch.object(client, 'FasterWhisperRecognizer', return_value=recognizer), \
                patch.object(client, 'SilenceSegmentRecorder', return_value=recorder), \
                contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            client.run_microphone_mode(self.args(), sender)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual([c.args for c in sender.send.call_args_list],
                         [('Bさん', 'text 1'), ('Bさん', 'text 2')])

    def test_ctrl_c_stops_recording_and_drains_final_audio(self):
        sd = Mock()
        sd.default.device = (1, 2)
        sd.query_devices.return_value = dict(name='USB', max_input_channels=1,
                                             default_samplerate=48000)
        sd.PortAudioError = RuntimeError
        recorder = Mock()
        closed = threading.Event()

        def segments(stop):
            stop.wait(2)
            yield 'final audio'
            closed.set()

        recorder.iter_segments.side_effect = segments
        recognizer = Mock()
        recognizer.transcribe.return_value = 'final text'
        sender = Mock()
        original_join = threading.Thread.join
        interrupted = False

        def join(thread, timeout=None):
            nonlocal interrupted
            if thread.name == 'microphone-recorder' and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return original_join(thread, timeout)

        with patch.dict('sys.modules', {'sounddevice': sd}), \
                patch.object(client, 'FasterWhisperRecognizer', return_value=recognizer), \
                patch.object(client, 'SilenceSegmentRecorder', return_value=recorder), \
                patch.object(threading.Thread, 'join', join), \
                contextlib.redirect_stderr(io.StringIO()):
            client.run_microphone_mode(self.args(), sender)
        self.assertTrue(closed.is_set())
        sender.send.assert_called_once_with('Bさん', 'final text')
        self.assertFalse(any(t.name in {'microphone-recorder', 'whisper-recognizer'}
                             for t in threading.enumerate()))

    def test_level_console_and_shutdown_flush(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy is needed for recorder test')
        for show_level in (False, True):
            stop = threading.Event()
            stream = Mock()

            def read(blocksize):
                stop.set()
                return np.full((blocksize, 1), 0.1, dtype=np.float32), True

            stream.read.side_effect = read
            sd = Mock()
            sd.InputStream.return_value.__enter__ = Mock(return_value=stream)
            sd.InputStream.return_value.__exit__ = Mock(return_value=False)
            recorder = client.SilenceSegmentRecorder(
                16000, 1, 0.25, 1.2, 0.6, 20, 0.01, 1, show_level)
            with patch.dict('sys.modules', {'sounddevice': sd}), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                segments = list(recorder.iter_segments(stop))
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].shape, (4000,))
            self.assertEqual(stdout.getvalue(), '')
            self.assertEqual('mean_abs=' in stderr.getvalue(), show_level)
            if show_level:
                self.assertIn('rms=0.1000 dBFS=-20.0 peak=0.1000', stderr.getvalue())
            sd.InputStream.assert_called_once_with(
                samplerate=16000, channels=1, dtype='float32', blocksize=4000, device=1)
            sd.InputStream.return_value.__exit__.assert_called_once()

    def test_output_only_device_rejected(self):
        sd = Mock()
        sd.query_devices.return_value = dict(max_input_channels=0)
        sd.PortAudioError = RuntimeError
        with patch.dict('sys.modules', {'sounddevice': sd}):
            with self.assertRaisesRegex(ValueError, '入力チャンネルが0'):
                client.run_microphone_mode(self.args('--input-device', '1'), Mock())

    def test_background_reconnect_fifo_and_clean_saved_output(self):
        attempted = threading.Event()
        restore = threading.Event()
        delivered = threading.Event()
        wire = []
        sock = Mock()

        def connect(*args, **kwargs):
            attempted.set()
            if not restore.is_set():
                raise OSError('offline')
            return sock

        def send(data):
            wire.append(json.loads(data))
            if len(wire) == 2:
                delivered.set()

        sock.sendall.side_effect = send
        with patch.object(client.socket, 'create_connection', side_effect=connect), \
                contextlib.redirect_stderr(io.StringIO()) as console, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            sender = client.TranscriptSender('host', 123, 0.01, 0.05)
            try:
                sender.send('Bさん', 'first')
                self.assertTrue(attempted.wait(2))
                sender.send('Bさん', 'second')
                restore.set()
                self.assertTrue(delivered.wait(2))
            finally:
                sender.close()
        self.assertFalse(sender._thread.is_alive())
        self.assertEqual([p['text'] for p in wire], ['first', 'second'])
        self.assertTrue(all(set(p) == {'speaker', 'text', 'client_time'} for p in wire))
        self.assertEqual(stdout.getvalue(), '')
        self.assertIn('接続', console.getvalue())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'conversaion.txt'
            log = ConversationLog(path, append=False)
            for payload in wire:
                log.write(*_extract_message(payload))
            self.assertEqual(path.read_text(), '[Bさん] first\n[Bさん] second\n')

    def test_full_send_queue_does_not_block(self):
        attempted = threading.Event()

        def offline(*args, **kwargs):
            attempted.set()
            raise OSError('offline')

        with patch.object(client.socket, 'create_connection', side_effect=offline), \
                contextlib.redirect_stderr(io.StringIO()) as console:
            sender = client.TranscriptSender('host', 123, 10, 0.05, queue_size=1)
            try:
                sender.send('B', 'in flight')
                self.assertTrue(attempted.wait(2))
                sender.send('B', 'queued')
                sender.send('B', 'dropped')
                self.assertIn('満杯', console.getvalue())
            finally:
                sender.close()
        self.assertFalse(sender._thread.is_alive())


if __name__ == '__main__':
    unittest.main()
