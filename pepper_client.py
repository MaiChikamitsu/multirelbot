from __future__ import annotations

import argparse
import socket
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class PepperClient:
    host: str
    port: int = 2003
    timeout_sec: float = 3.0

    def send_command(self, command: str) -> None:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_sec
        ) as sock:
            sock.sendall(f"{command}\n".encode("utf-8"))

    def say(self, message: str) -> None:
        self.send_command(f"say:{message}")

    def animate(self, name: str, seconds: Optional[int] = None) -> None:
        suffix = f",{seconds}" if seconds is not None else ""
        self.send_command(f"anim:{name}{suffix}")


def from_config() -> PepperClient:
    cfg = config.get_config()
    host = getattr(cfg.pepper, "ip", None)
    port = getattr(cfg.pepper, "port", 2003) or 2003
    if not host:
        raise RuntimeError("config.local.yaml の pepper.ip を設定してください。")
    return PepperClient(host=host, port=port)


def send_to_pepper(message: str) -> None:
    client = from_config()
    client.say(message)


def send_to_pepper_async(message: str) -> None:
    threading.Thread(target=send_to_pepper, args=(message,), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a test utterance to Pepper.")
    parser.add_argument("message", nargs="?", help="Pepperに言わせるテキスト")
    parser.add_argument("--host", help="config.local.yaml の pepper.ip を一時的に上書き")
    parser.add_argument("--port", type=int, help="config.local.yaml の pepper.port を一時的に上書き")
    parser.add_argument(
        "--check",
        action="store_true",
        help="発話せず、PepperのTCP待ち受けに接続できるかだけ確認",
    )
    args = parser.parse_args()
    try:
        client = _client_from_args(args)
        if args.check:
            client.send_command("ping")
            print(f"Pepper TCP server is reachable: {client.host}:{client.port}")
        elif args.message:
            client.say(args.message)
            print(f"Sent to Pepper TCP server: {client.host}:{client.port}")
        else:
            parser.error("message か --check を指定してください。")
    except ConnectionRefusedError:
        host, port = _endpoint_from_args(args)
        print(
            f"Pepperへの接続は拒否されました: {host}:{port}\n"
            "IPは合っていても、Pepper側でTCPサーバアプリが起動していない、"
            "またはポート番号が違う可能性が高いです。",
            file=sys.stderr,
        )
        print(
            "確認: Pepper側Android/Javaアプリを起動し、ServerSocket(port=2003) が"
            " 待ち受けている状態で再実行してください。",
            file=sys.stderr,
        )
        raise
    except TimeoutError:
        print(
            "Pepperへの接続がタイムアウトしました。同じWi-Fiか、IPアドレスが正しいかを確認してください。",
            file=sys.stderr,
        )
        raise
    except OSError as exc:
        print(f"Pepper接続エラー: {exc}", file=sys.stderr)
        raise


def _endpoint_from_args(args: argparse.Namespace) -> tuple[str, int]:
    cfg = config.get_config()
    host = args.host or getattr(cfg.pepper, "ip", None)
    port = args.port or getattr(cfg.pepper, "port", 2003) or 2003
    if not host:
        raise RuntimeError("config.local.yaml の pepper.ip を設定してください。")
    return host, port


def _client_from_args(args: argparse.Namespace) -> PepperClient:
    host, port = _endpoint_from_args(args)
    return PepperClient(host=host, port=port)


if __name__ == "__main__":
    main()
