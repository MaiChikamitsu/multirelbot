#!/bin/zsh
set -e
cd "${0:A:h}"
trap 'echo "セットアップに失敗しました。README.mdを確認してください。"; read "?Enterで閉じます: "' ZERR
PYTHON_BIN="${EDIT_PYTHON:-$(command -v python3)}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,9), "Python 3.9以上が必要です"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'from editor.engine import binary; print(binary("ffmpeg")); print(binary("ffprobe"))'
echo 'セットアップが完了しました。次回から 起動.command をダブルクリックしてください。'
read '?Enterで閉じます: '
