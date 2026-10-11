#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
audio="${1:-$script_dir/rokuon.m4a}"
if [[ ! -f "$audio" ]]; then
    echo "録音ファイルがありません: $audio" >&2
    echo "rokuon.m4aをこのフォルダに置くか、./run.sh 音声ファイル を実行してください。" >&2
    exit 1
fi
if [[ ! -x "$script_dir/.venv/bin/python" ]]; then
    python3 -m venv "$script_dir/.venv"
fi
if ! "$script_dir/.venv/bin/python" -c 'import faster_whisper' 2>/dev/null; then
    "$script_dir/.venv/bin/python" -m pip install -r "$script_dir/requirements.txt"
fi
if [[ $# -gt 0 ]]; then
    shift
fi
exec "$script_dir/.venv/bin/python" "$script_dir/transcribe.py" "$audio" "$@"
