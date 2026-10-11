# Macで録音 → Whisper → moji.txtへ追記

Mac / iPhoneの「ボイスメモ」で5分ほど録音し、音声をこのフォルダに
`rokuon.m4a` として保存します。Macでは録音をFinderへドラッグして保存できます。

今は 「ボイスメモ」アプリ内に保存されています。文字起こし用には、Finderへ取り出すのが手っ取り早いです。
1. 画面の 「riokuon」行をデスクトップへドラッグ。
2. 出てきた音声ファイルを rokuon.m4a に名前変更。
3. MultiRelBot/mojiokoshi フォルダへ移動。
拡張子を名前だけ変える必要はありません。MP3やWAVもそのまま指定できます。

リポジトリのルートから:

```bash
cd mojiokoshi
./run.sh
```

初回はPython仮想環境・依存パッケージ・Whisperモデルを自動で用意します。
Python 3.9以上と、初回ダウンロード用のインターネット接続が必要です。
APIキー不要。音声の認識はMac内で実行し、音声ファイルをAPIへ送信しません。
別途FFmpegをインストールする必要はありません。

完了すると録音ファイルと同じディレクトリの `moji.txt` に、UTF-8で本文だけを
追記します。既存の文章は残ります。同じ音声で再実行すると同じ内容も再度追記します。
5分以外の長さでも実行できます。録音そのものはボイスメモで行ってください。

別の名前・場所の音声も指定できます（出力は指定した音声の隣）:

```bash
./run.sh "/Users/yourname/Desktop/録音.m4a"
```

速度を優先して、既定は日本語・`base` モデル・CPU int8・beam_size=1です。
認識精度を上げたい場合:

```bash
./run.sh rokuon.m4a --model small
```

実行時間はMacの性能と音声により変わります。初回はモデルのダウンロード分もかかります。
実装: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)。
