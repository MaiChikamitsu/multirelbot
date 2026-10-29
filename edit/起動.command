#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo '初回は、このフォルダの セットアップ.command をダブルクリックしてください。'
  read '?Enterで閉じます: '
  exit 1
fi
.venv/bin/python cli.py
read '?Enterで閉じます: '
