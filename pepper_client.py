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
    port: int = 2002
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
    port = getattr(cfg.pepper, "port", 2002) or 2002
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
    parser.add_argument("message", help="Pepperに言わせるテキスト")
    parser.add_argument("--host", help="config.local.yaml の pepper.ip を一時的に上書き")
    parser.add_argument("--port", type=int, help="config.local.yaml の pepper.port を一時的に上書き")
    args = parser.parse_args()
    try:
        if args.host:
            cfg = config.get_config()
            client = PepperClient(
                host=args.host,
                port=args.port or getattr(cfg.pepper, "port", 2002) or 2002,
            )
            client.say(args.message)
        elif args.port:
            cfg = config.get_config()
            host = getattr(cfg.pepper, "ip", None)
            if not host:
                raise RuntimeError("config.local.yaml の pepper.ip を設定してください。")
            PepperClient(host=host, port=args.port).say(args.message)
        else:
            send_to_pepper(args.message)
    except ConnectionRefusedError:
        print(
            "Pepperへの接続は拒否されました。IPは合っていても、Pepper側で"
            " TCPサーバアプリが起動していない、またはポート番号が違う可能性が高いです。",
            file=sys.stderr,
        )
        print(
            "確認: Pepper側Android/Javaアプリを起動し、ServerSocket(port=2002) が"
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


if __name__ == "__main__":
    main()
