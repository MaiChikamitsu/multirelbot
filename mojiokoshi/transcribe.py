#!/usr/bin/env python3
"""録音ファイルをローカルWhisperで認識し、隣のmoji.txtへ追記する。"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "rokuon.m4a")
    parser.add_argument("--model", default="base", help="既定: base。精度優先ならsmall")
    args = parser.parse_args()
    audio = args.audio.expanduser().resolve()
    if not audio.is_file():
        parser.error(f"録音ファイルがありません: {audio}")

    try:
        from faster_whisper import WhisperModel

        print(f"文字起こし中: {audio.name} (モデル: {args.model})", file=sys.stderr)
        model = WhisperModel(args.model, device="cpu", compute_type="int8",
                             download_root=str(Path(__file__).resolve().parent / ".models"))
        segments, _ = model.transcribe(str(audio), language="ja", beam_size=1,
                                       vad_filter=True)
        # 認識完了後にだけ追記し、認識エラーで途中の結果が残るのを防ぐ。
        lines = [segment.text.strip() for segment in segments if segment.text.strip()]
        if not lines:
            print("発話を検出できませんでした。moji.txtへの追記はありません。", file=sys.stderr)
            return 0
        output = audio.parent / "moji.txt"
        separator = ""
        if output.exists() and output.stat().st_size:
            with output.open("rb") as existing:
                existing.seek(-1, 2)
                if existing.read(1) != b"\n":
                    separator = "\n"
        with output.open("a", encoding="utf-8") as destination:
            destination.write(separator + "\n".join(lines) + "\n")
        print(f"追記しました: {output}")
        return 0
    except ImportError:
        print("依存パッケージが必要です。./run.sh で実行してください。", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"文字起こしに失敗しました: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
