from __future__ import annotations

import argparse
import json
import socketserver
import threading
from pathlib import Path
from typing import Any, Tuple, Type


OUTPUT_FILENAME = "conversaion.txt"


class ConversationLog:
    def __init__(self, output_path: Path, append: bool) -> None:
        if output_path.name != OUTPUT_FILENAME:
            raise ValueError(f"出力ファイル名は必ず {OUTPUT_FILENAME} にしてください。")

        self.output_path = output_path
        self._lock = threading.Lock()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not append:
            self.output_path.write_text("", encoding="utf-8")

    def write(self, speaker: str, text: str) -> str:
        speaker = _single_line(speaker)
        text = _single_line(text)
        line = f"[{speaker}] {text}"

        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        return line


class ConversationTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_class: Type[socketserver.BaseRequestHandler],
        conversation_log: ConversationLog,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.conversation_log = conversation_log


class TranscriptHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"接続: {peer}", flush=True)

        try:
            for raw_line in self.rfile:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"無効なJSONを無視しました ({peer}): {exc}", flush=True)
                    continue

                speaker, text = _extract_message(payload)
                if not speaker or not text:
                    print(f"speaker/text が空のメッセージを無視しました ({peer})", flush=True)
                    continue

                saved_line = self.server.conversation_log.write(speaker, text)
                print(saved_line, flush=True)
        finally:
            print(f"切断: {peer}", flush=True)


def _extract_message(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", ""

    speaker = str(payload.get("speaker", "")).strip()
    text = str(payload.get("text", "")).strip()
    return speaker, text


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _output_path(value: str) -> Path:
    path = Path(value)
    if path.name != OUTPUT_FILENAME:
        raise argparse.ArgumentTypeError(
            f"出力ファイル名は必ず {OUTPUT_FILENAME} にしてください。"
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3人対面会話の文字起こし結果をTCPで受け取り、時系列で保存します。"
    )
    parser.add_argument("--host", default="0.0.0.0", help="待ち受けIPアドレス")
    parser.add_argument("--port", type=int, default=50007, help="待ち受けポート番号")
    parser.add_argument(
        "--output",
        type=_output_path,
        default=Path(OUTPUT_FILENAME),
        help=f"保存先。ファイル名は {OUTPUT_FILENAME} 固定です。",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="既存の conversaion.txt に追記します。指定しない場合は起動時に空にします。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversation_log = ConversationLog(args.output, append=args.append)

    with ConversationTCPServer((args.host, args.port), TranscriptHandler, conversation_log) as server:
        host, port = server.server_address
        print(f"ホストサーバー起動: {host}:{port}", flush=True)
        print(f"保存先: {conversation_log.output_path.resolve()}", flush=True)
        print("終了するには Ctrl+C を押してください。", flush=True)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nホストサーバーを終了します。", flush=True)


if __name__ == "__main__":
    main()
