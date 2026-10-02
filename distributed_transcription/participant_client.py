from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime
from typing import Any, Iterator, List, Optional, Union


class TranscriptSender:
    def __init__(
        self,
        host: str,
        port: int,
        retry_seconds: float,
        socket_timeout: float,
    ) -> None:
        self.host = host
        self.port = port
        self.retry_seconds = retry_seconds
        self.socket_timeout = socket_timeout
        self._socket: Optional[socket.socket] = None

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def send(self, speaker: str, text: str) -> None:
        payload = {
            "speaker": speaker,
            "text": text,
            "client_time": datetime.now().isoformat(timespec="seconds"),
        }
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

        while True:
            if self._socket is None:
                self._connect()

            try:
                assert self._socket is not None
                self._socket.sendall(data)
                return
            except OSError as exc:
                print(f"送信に失敗しました。再接続します: {exc}", file=sys.stderr, flush=True)
                self.close()
                time.sleep(self.retry_seconds)

    def _connect(self) -> None:
        while True:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.socket_timeout
                )
                sock.settimeout(None)
                self._socket = sock
                print(f"接続しました: {self.host}:{self.port}", flush=True)
                return
            except OSError as exc:
                print(
                    f"接続できません: {self.host}:{self.port} ({exc})。"
                    f" {self.retry_seconds:.1f}秒後に再試行します。",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(self.retry_seconds)


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

        print(f"faster-whisper モデル読み込み中: {model_name}", flush=True)
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
        input_device: Optional[Union[int, str]],
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self.silence_seconds = silence_seconds
        self.min_segment_seconds = min_segment_seconds
        self.max_segment_seconds = max_segment_seconds
        self.volume_threshold = volume_threshold
        self.input_device = input_device

    def iter_segments(self) -> Iterator:
        import numpy as np
        import sounddevice as sd

        blocksize = int(self.sample_rate * self.chunk_seconds)
        print("マイク入力を開始します。終了するには Ctrl+C を押してください。", flush=True)
        print(
            f"設定: {self.sample_rate}Hz / {self.channels}ch / "
            f"無音{self.silence_seconds:.1f}秒で区切り",
            flush=True,
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

            while True:
                data, overflowed = stream.read(blocksize)
                if overflowed:
                    print("警告: マイク入力が一部取りこぼされました。", file=sys.stderr)

                samples = np.asarray(data, dtype=np.float32).reshape(-1)
                volume = float(np.abs(samples).mean())
                now = time.monotonic()

                if volume >= self.volume_threshold:
                    if not has_voice:
                        print("発話を検出しました。", flush=True)
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


def run_manual_mode(args: argparse.Namespace, sender: TranscriptSender) -> None:
    print("手動入力モードです。1行入力するたびにホストへ送信します。", flush=True)
    print("終了するには /quit または /exit を入力してください。", flush=True)

    for raw_line in sys.stdin:
        text = raw_line.strip()
        if text in {"/quit", "/exit"}:
            break
        if not text:
            continue
        sender.send(args.speaker, text)
        print(f"送信: [{args.speaker}] {text}", flush=True)


def run_microphone_mode(args: argparse.Namespace, sender: TranscriptSender) -> None:
    recognizer = FasterWhisperRecognizer(
        model_name=args.model,
        device=args.whisper_device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad_filter,
    )
    recorder = SilenceSegmentRecorder(
        sample_rate=args.sample_rate,
        channels=1,
        chunk_seconds=args.chunk_seconds,
        silence_seconds=args.silence_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
        volume_threshold=args.volume_threshold,
        input_device=args.input_device,
    )

    for audio in recorder.iter_segments():
        print("文字起こし中...", flush=True)
        text = recognizer.transcribe(audio)
        if not text:
            print("文字起こし結果が空だったため送信しません。", flush=True)
            continue

        print(f"認識: {text}", flush=True)
        sender.send(args.speaker, text)
        print(f"送信: [{args.speaker}] {text}", flush=True)


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
        help="初回接続のタイムアウト秒数",
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
    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="faster-whisper 側のVADフィルタを無効化します。",
    )

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
        default=None,
        help="sounddevice の入力デバイス番号または名前。未指定なら既定デバイス。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sender = TranscriptSender(
        host=args.host,
        port=args.port,
        retry_seconds=args.retry_seconds,
        socket_timeout=args.socket_timeout,
    )

    try:
        if args.mode == "manual":
            run_manual_mode(args, sender)
        else:
            run_microphone_mode(args, sender)
    except KeyboardInterrupt:
        print("\nクライアントを終了します。", flush=True)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
