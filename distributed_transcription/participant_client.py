from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
import wave
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Any, Iterator, List, Optional


def debug(message: str) -> None:
    # Console only: never put diagnostics into transcript payloads or files.
    print(message, file=sys.stderr, flush=True)


def join_worker(thread: threading.Thread) -> None:
    while thread.is_alive():
        try:
            thread.join(timeout=0.1)
        except KeyboardInterrupt:
            debug("終了処理中です。残りの処理を待っています。")


def save_diagnostic_audio(audio, directory: Path, sample_rate: int) -> Path:
    """Save the exact segmented Whisper input, quantized to portable PCM16."""
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"utterance_{stamp}_{uuid4().hex[:8]}.wav"
    pcm = np.rint(np.clip(np.asarray(audio), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return path


class TranscriptSender:
    def __init__(self, host: str, port: int, retry_seconds: float,
                 socket_timeout: float, queue_size: int = 1000) -> None:
        self.host = host
        self.port = port
        self.retry_seconds = retry_seconds
        self.socket_timeout = socket_timeout
        self._socket: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="transcript-sender")
        self._thread.start()

    def _disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def close(self) -> None:
        # Allow a short flush, but never wait indefinitely for a missing host.
        deadline = time.monotonic() + 2.0
        try:
            while self._queue.unfinished_tasks and time.monotonic() < deadline:
                self._stop.wait(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            join_worker(self._thread)
        if self._queue.unfinished_tasks:
            debug(f"警告: 終了時の未送信結果 {self._queue.unfinished_tasks}件を破棄します。")

    def send(self, speaker: str, text: str) -> None:
        payload = {
            "speaker": speaker,
            "text": text,
            "client_time": datetime.now().isoformat(timespec="seconds"),
        }
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            debug("警告: 送信queueが満杯のため、新しい認識結果を破棄します。")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                while not self._stop.is_set():
                    try:
                        if self._socket is None:
                            self._socket = socket.create_connection(
                                (self.host, self.port), timeout=self.socket_timeout)
                            self._socket.settimeout(self.socket_timeout)
                            debug(f"接続しました: {self.host}:{self.port}")
                        self._socket.sendall(data)
                        self._queue.task_done()
                        break
                    except OSError as exc:
                        debug(f"接続・送信に失敗: {self.host}:{self.port} ({exc})。"
                              f" {self.retry_seconds:.1f}秒後に再試行します。")
                        self._disconnect()
                        self._stop.wait(self.retry_seconds)
        finally:
            self._disconnect()


class FasterWhisperRecognizer:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
    ) -> None:
        from faster_whisper import WhisperModel

        print(f"faster-whisper モデル読み込み中: {model_name}", file=sys.stderr, flush=True)
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe(self, audio) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return "".join(segment.text.strip() for segment in segments).strip()


class SilenceSegmentRecorder:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        chunk_seconds: float,
        silence_seconds: float,
        min_segment_seconds: float,
        max_segment_seconds: float,
        volume_threshold: float,
        input_device: Optional[int],
        show_level: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self.silence_seconds = silence_seconds
        self.min_segment_seconds = min_segment_seconds
        self.max_segment_seconds = max_segment_seconds
        self.volume_threshold = volume_threshold
        self.input_device = input_device
        self.show_level = show_level

    def iter_segments(self, stop: Optional[threading.Event] = None) -> Iterator:
        import numpy as np
        import sounddevice as sd

        stop = stop or threading.Event()
        last_level = 0.0
        blocksize = int(self.sample_rate * self.chunk_seconds)
        print("マイク入力を開始します。終了するには Ctrl+C を押してください。", file=sys.stderr, flush=True)
        print(
            f"設定: {self.sample_rate}Hz / {self.channels}ch / "
            f"無音{self.silence_seconds:.1f}秒で区切り",
            file=sys.stderr, flush=True,
        )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=blocksize,
            device=self.input_device,
        ) as stream:
            frames: List[Any] = []
            has_voice = False
            silence_started_at: Optional[float] = None
            segment_started_at: Optional[float] = None

            while not stop.is_set():
                data, overflowed = stream.read(blocksize)
                if overflowed:
                    print("警告: マイク入力が一部取りこぼされました。", file=sys.stderr)

                samples = np.asarray(data, dtype=np.float32).reshape(-1)
                volume = float(np.abs(samples).mean())
                now = time.monotonic()
                if self.show_level and now - last_level >= 0.25:
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    db = 20 * np.log10(max(rms, 1e-12))
                    peak = float(np.max(np.abs(samples)))
                    debug(f"mean_abs={volume:.4f} rms={rms:.4f} dBFS={db:.1f} peak={peak:.4f}")
                    last_level = now

                if volume >= self.volume_threshold:
                    if not has_voice:
                        print("発話を検出しました。", file=sys.stderr, flush=True)
                        has_voice = True
                        segment_started_at = now
                    silence_started_at = None

                if has_voice:
                    frames.append(samples.copy())

                    if volume < self.volume_threshold:
                        if silence_started_at is None:
                            silence_started_at = now
                    else:
                        silence_started_at = None

                    elapsed = now - (segment_started_at or now)
                    silence_elapsed = (
                        now - silence_started_at if silence_started_at is not None else 0.0
                    )

                    should_close_by_silence = (
                        silence_elapsed >= self.silence_seconds
                        and elapsed >= self.min_segment_seconds
                    )
                    should_close_by_length = elapsed >= self.max_segment_seconds

                    if should_close_by_silence or should_close_by_length:
                        audio = np.concatenate(frames)
                        frames = []
                        has_voice = False
                        silence_started_at = None
                        segment_started_at = None
                        yield audio

            if frames:
                yield np.concatenate(frames)


def run_manual_mode(args: argparse.Namespace, sender: TranscriptSender) -> None:
    print("手動入力モードです。1行入力するたびにホストへ送信します。", file=sys.stderr, flush=True)
    print("終了するには /quit または /exit を入力してください。", file=sys.stderr, flush=True)

    for raw_line in sys.stdin:
        text = raw_line.strip()
        if text in {"/quit", "/exit"}:
            break
        if not text:
            continue
        sender.send(args.speaker, text)
        print(f"送信待ち: [{args.speaker}] {text}", file=sys.stderr, flush=True)


def run_microphone_mode(args: argparse.Namespace, sender: TranscriptSender) -> None:
    import sounddevice as sd

    device_index = args.input_device
    if device_index is None:
        device_index = int(sd.default.device[0])
    try:
        if device_index < 0:
            raise ValueError("既定の入力デバイスがありません")
        device_info = sd.query_devices(device_index)
        if device_info["max_input_channels"] == 0:
            raise ValueError("入力チャンネルが0のため録音できません")
        sd.check_input_settings(device=device_index, channels=1,
                                dtype="float32", samplerate=args.sample_rate)
    except (ValueError, sd.PortAudioError) as exc:
        raise ValueError(f"入力デバイス {device_index} を使用できません: {exc}") from exc
    debug(f"使用入力デバイス:\n  index: {device_index}\n"
          f"  name: {device_info['name']}\n"
          f"  input channels: {device_info['max_input_channels']}\n"
          f"  default samplerate: {device_info['default_samplerate']}")
    debug("=== participant client settings ===\n"
          f"speaker: {args.speaker}\nhost: {args.host}\nport: {args.port}\n"
          f"input device: {device_index} / {device_info['name']}\n"
          f"sample rate: {args.sample_rate}\nchannels: 1\n"
          f"chunk seconds: {args.chunk_seconds}\nsilence seconds: {args.silence_seconds}\n"
          f"volume threshold: {args.volume_threshold}\nwhisper model: {args.model}\n"
          f"whisper device: {args.whisper_device}\n"
          f"whisper VAD: {'ON' if args.whisper_vad_filter else 'OFF'}\n"
          "===============================")
    recognizer = FasterWhisperRecognizer(
        model_name=args.model, device=args.whisper_device,
        compute_type=args.compute_type, beam_size=args.beam_size,
        vad_filter=args.whisper_vad_filter,
    )
    recorder = SilenceSegmentRecorder(
        sample_rate=args.sample_rate, channels=1,
        chunk_seconds=args.chunk_seconds, silence_seconds=args.silence_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
        volume_threshold=args.volume_threshold, input_device=device_index,
        show_level=args.show_level,
    )
    # Unbounded audio queue deliberately preserves speech during slow inference.
    audio_queue = queue.Queue()
    stop = threading.Event()
    errors = queue.Queue()

    def record() -> None:
        try:
            for audio in recorder.iter_segments(stop):
                audio_queue.put_nowait(audio)
                if audio_queue.qsize() == 20:
                    debug("警告: 音声queueが20件に達しました。認識が録音に追いついていません。")
        except Exception as exc:
            errors.put(exc)
            stop.set()
        finally:
            audio_queue.put_nowait(None)

    def transcribe() -> None:
        while True:
            audio = audio_queue.get()
            if audio is None:
                return
            audio_path = None
            if args.save_audio_dir is not None:
                try:
                    audio_path = save_diagnostic_audio(audio, args.save_audio_dir, args.sample_rate)
                    debug(f"診断音声: {audio_path.resolve()}")
                except Exception as exc:
                    debug(f"警告: 診断音声を保存できません（認識は継続）: {exc}")
            try:
                debug("文字起こし中...")
                text = recognizer.transcribe(audio)
                if text:
                    debug(f"認識: {text}" + (f" [音声: {audio_path.name}]" if audio_path else ""))
                    sender.send(args.speaker, text)
                else:
                    debug("文字起こし結果が空だったため送信しません。")
            except Exception as exc:
                debug(f"警告: 文字起こしに失敗しました: {exc}")

    recording_thread = threading.Thread(target=record, name="microphone-recorder")
    recognition_thread = threading.Thread(target=transcribe, name="whisper-recognizer")
    recognition_thread.start()
    recording_thread.start()
    try:
        while recording_thread.is_alive():
            recording_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        debug("録音を停止し、残りの発話を文字起こしします。")
    finally:
        stop.set()
        # Native inference cannot safely be killed; drain recorded speech first.
        join_worker(recording_thread)
        join_worker(recognition_thread)
    if not errors.empty():
        raise errors.get()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="参加者PCでマイク音声または手動入力を文字起こしし、ホストへ送信します。"
    )
    parser.add_argument("--speaker", required=True, help="例: Aさん / Bさん / Cさん")
    parser.add_argument("--host", default="127.0.0.1", help="ホストPCのIPアドレス")
    parser.add_argument("--port", type=int, default=50007, help="ホストPCのポート番号")
    parser.add_argument(
        "--mode",
        choices=("mic", "manual"),
        default="mic",
        help="mic: マイク入力、manual: 手動入力",
    )
    parser.add_argument(
        "--retry-seconds",
        type=float,
        default=2.0,
        help="接続失敗時の再試行間隔",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=5.0,
        help="接続・送信のタイムアウト秒数",
    )

    parser.add_argument("--model", default="small", help="faster-whisper のモデル名")
    parser.add_argument(
        "--whisper-device",
        default="cpu",
        help="faster-whisper の実行デバイス。例: cpu / cuda / auto",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper の compute_type。例: int8 / float16 / auto",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Whisperのbeam size")
    vad = parser.add_mutually_exclusive_group()
    vad.add_argument("--whisper-vad-filter", action="store_true",
                     help="faster-whisper 側のVADを有効化します（既定OFF）。")
    vad.add_argument("--no-vad-filter", dest="whisper_vad_filter",
                     action="store_false", help="互換用: Whisper VADを無効化します。")
    parser.set_defaults(whisper_vad_filter=False)
    parser.add_argument("--show-level", action="store_true", help="入力音量をstderrに表示")
    parser.add_argument("--save-audio-dir", type=Path, default=None,
                        help="診断用: Whisperへ渡す発話音声を指定フォルダにWAV保存（既定OFF）")
    parser.add_argument("--send-queue-size", type=int, default=1000,
                        help="送信待ち最大件数（満杯時は新しい結果を破棄）")

    parser.add_argument("--sample-rate", type=int, default=16000, help="録音サンプルレート")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="マイクから読み取る1チャンクの秒数",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=1.2,
        help="この秒数だけ無音が続いたら1発話として区切ります。",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.6,
        help="送信する最短発話秒数",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=20.0,
        help="無音がなくてもこの秒数で一度文字起こしします。",
    )
    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=0.01,
        help="発話検出の音量しきい値。環境に応じて調整します。",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="sounddevice の入力デバイス番号。未指定なら起動時の既定デバイス。",
    )
    args = parser.parse_args()
    for name in ("sample_rate", "chunk_seconds", "silence_seconds",
                 "min_segment_seconds", "max_segment_seconds", "socket_timeout",
                 "retry_seconds", "send_queue_size", "beam_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} は正数を指定してください")
    if int(args.sample_rate * args.chunk_seconds) < 1:
        parser.error("chunkが1サンプル未満です")
    return args


def main() -> None:
    args = parse_args()
    sender = TranscriptSender(
        host=args.host,
        port=args.port,
        retry_seconds=args.retry_seconds,
        socket_timeout=args.socket_timeout,
        queue_size=args.send_queue_size,
    )

    try:
        if args.mode == "manual":
            run_manual_mode(args, sender)
        else:
            run_microphone_mode(args, sender)
    except KeyboardInterrupt:
        print("\nクライアントを終了します。", file=sys.stderr, flush=True)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
