from __future__ import annotations

import argparse
import socket
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
    args = parser.parse_args()
    send_to_pepper(args.message)


if __name__ == "__main__":
    main()
