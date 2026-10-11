# 3人対面会話 文字起こし集約ツール

3人の参加者PCでそれぞれクライアントを起動し、有線マイクの音声を文字起こしして、ホストPCのTCPサーバーへ集約します。ホスト側は受信した順に、必ず `conversaion.txt` へ保存します。

出力形式は次の通りです。

```text
[Aさん] そうだよね
[Bさん] うんうん
[Cさん] 私もそう思う
```

## 前のコードで使っていた音声認識のやり方

既存コードでは、`recording.py` や `realtime_communicator.py` で、16kHz・モノラルの音声を無音区間で区切り、1発話ぶんの音声バッファまたはWAVを文字起こしへ渡していました。OpenAI の `gpt-4o-transcribe` や Google Speech-to-Text の分岐もありますが、今回のコードでは同じ「無音で区切って発話単位で認識する」流れを、ローカル実行しやすい `faster-whisper` で実装しています。

## ファイル構成

```text
distributed_transcription/
  host_server.py          # ホストPCで起動するTCPサーバー
  participant_client.py   # 各参加者PCで起動するクライアント
  requirements.txt        # 必要なPythonパッケージ
  README.md               # この説明
```

## セットアップ

各PCで次を実行します。

```bash
cd distributed_transcription
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windowsの場合は仮想環境の有効化だけ次のようにします。

```powershell
.venv\Scripts\activate
```

## ホストPCでサーバーを起動

ホストPCで次を実行します。

```bash
python host_server.py --host 0.0.0.0 --port 50007
```

同じフォルダに `conversaion.txt` が作成されます。起動し直すと既定ではファイルを空にしてから保存します。追記したい場合は `--append` を付けます。

```bash
python host_server.py --host 0.0.0.0 --port 50007 --append
```

## 参加者PCでクライアントを起動

参加者ごとに話者名を変えて起動します。`--host` にはホストPCのIPアドレスを指定してください。

```bash
python participant_client.py --speaker Aさん --host 192.168.1.10 --port 50007 --mode mic
python participant_client.py --speaker Bさん --host 192.168.1.10 --port 50007 --mode mic
python participant_client.py --speaker Cさん --host 192.168.1.10 --port 50007 --mode mic
```

マイクなしで確認する場合は手動入力モードを使います。

```bash
python participant_client.py --speaker Aさん --host 192.168.1.10 --port 50007 --mode manual
```

手動入力モードでは、1行入力するたびに送信します。終了は `/quit` または `/exit` です。

## マイク入力の調整

有線マイクの音量や部屋の環境によって、発話検出のしきい値を調整してください。

```bash
python participant_client.py \
  --speaker Aさん \
  --host 192.168.1.10 \
  --mode mic \
  --volume-threshold 0.008 \
  --silence-seconds 1.0
```

入力デバイスを指定したい場合は、sounddevice の入力デバイス番号（整数）を `--input-device` に指定します。

```bash
python participant_client.py --speaker Aさん --host 192.168.1.10 --mode mic --input-device 1
```

Whisperモデルは既定で `small` です。軽くしたい場合は `base`、精度を上げたい場合は `medium` などに変更できます。

```bash
python participant_client.py --speaker Aさん --host 192.168.1.10 --mode mic --model base
```

## 再接続

録音 → 音声queue → 文字起こし → 送信queue → TCP送信の3スレッド構成です。送信スレッドだけが接続・再接続を担当するため、ホストに接続できなくても録音と認識を継続します。既定で2秒ごとに再試行し、復旧後は保留中の結果をFIFO順で送ります。`client_time` は送信queueへ登録した時刻で、再送時も変わりません。

送信queueは既定1000件（別途、送信中1件）。`--send-queue-size` で変更できます。満杯時は新しい結果を破棄しstderrに警告します。保留結果はメモリ内のみです。TCPには受信確認の仕組みがないため、切断タイミングによる重複や未達は完全には防げません。

音声queueは録音をブロックせず発話を保持するため上限を設けていません。20件に達すると警告します。認識が長時間追いつかない場合はメモリ使用量が増えるため、軽いモデルを選び実験前に連続発話で確認してください。

Ctrl+Cで録音を停止し、発話途中の音声と音声queueを最後まで認識します。その後、最大2秒間送信完了を待ち、再接続待機を解除して終了します。実行中のsocket操作は `--socket-timeout`（既定5秒）まで待つ場合があります。Whisperの処理は安全に強制中断できないため、音声の残量によって終了まで時間がかかります。終了時に残る未送信結果は破棄し件数をstderrに表示します。USBマイク抜去などの録音エラーは表示して終了し、別のマイクへ自動切替はしません。

## Zoomとの併用

Zoomを同時に使っても構いません。このプログラムはZoom音声を解析せず、各参加者PCに接続された有線マイクの入力だけを文字起こしします。

## 手動入力モードのテスト例

ホストPCのターミナルでサーバーを起動します。

```bash
python host_server.py --host 127.0.0.1 --port 50007
```

別のターミナルで、Aさん、Bさん、Cさんとして手動入力します。

```bash
python participant_client.py --speaker Aさん --host 127.0.0.1 --port 50007 --mode manual
python participant_client.py --speaker Bさん --host 127.0.0.1 --port 50007 --mode manual
python participant_client.py --speaker Cさん --host 127.0.0.1 --port 50007 --mode manual
```

保存結果は `conversaion.txt` です。ファイル名は依頼内容に合わせて `conversation.txt` ではなく `conversaion.txt` にしています。

## USBマイクと音量確認

デバイス一覧を確認してUSBマイクの番号を指定してください。

```bash
python -m sounddevice
python participant_client.py --speaker Bさん --host 10.32.1.16 --port 50007 --mode mic --input-device 1
python participant_client.py --speaker Bさん --host 10.32.1.16 --port 50007 --mode mic --input-device 1 --show-level
```

起動時に番号・名前・入力チャンネル数・既定サンプルレートと実行設定をstderrへ表示します。番号未指定の場合も、起動時の既定入力の番号を解決して固定します。入力不可のデバイスや指定録音形式に非対応のデバイスはモデル読込前にエラーにします。デバイス番号は接続順によって変わり得るため、毎回表示された名前も確認してください。

`--show-level` は約0.25秒ごと（読み取りチャンク以上の間隔）にmean absolute amplitude・RMS・dBFS・peakをstderrに表示します。既定OFFです。faster-whisperのVADも既定OFFで、必要な場合は `--whisper-vad-filter` を指定します。以前の `--no-vad-filter` も互換用に使用できます。

## 検証とログ分離

依存ライブラリやマイクを必要としない回帰テスト:

```bash
python -m unittest discover -s distributed_transcription -p 'test_participant_client.py' -v
```

上のコマンドはリポジトリルートで実行してください。CLIの整数/VAD、認識中の録音継続、認識エラー後の継続、Ctrl+C時の残音声処理、入力不可デバイス、通信復旧時の順序、queue満杯、保存内容を模擬テストします。

実機では `--show-level` 付きで連続発話し、認識中も音量表示が続くことを確認してください。ホスト停止中にも認識が続き、再起動すると保留結果が届くこと、Ctrl+Cでマイクが解放されることも確認してください。再起動時に既存データを残す場合はホストに `--append` を指定します。

クライアントの状態表示・警告・認識表示はすべてstderrへ出します。送信するJSONは従来どおり `speaker`・`text`・`client_time` のみです。ホストの専用保存処理は `[話者] 本文` だけを `conversaion.txt` に書き、デバッグ情報は保存しません。stdout/stderrのリダイレクト先を実験結果ファイルに指定しないでください。

回帰テストは接続ログがstderrだけに存在すること、JSONのキーが上記3つだけであること、ホスト保存ファイルが認識結果だけであることを検証します。実機でもホストの `conversaion.txt` を開き、設定・音量・接続・警告が含まれないことを確認してください。第三者ライブラリ自身の表示はこのアプリの保存処理には入りません。

## 認識精度の切り分け用WAV保存

`--save-audio-dir ./mic_debug` を追加すると、Whisperに渡す発話区間を16-bit PCM・モノラルWAVとして保存します。既定では保存しません。ファイル名は一意で、認識結果と対応する名前をstderrに表示します。WAV保存は認識スレッドで行い、失敗しても認識と録音を継続します。会話ファイル・送信JSONには診断情報を追加しません。

```bash
python participant_client.py --speaker Bさん --host 127.0.0.1 --input-device 1 --show-level --save-audio-dir ./mic_debug
```

最初は1人だけがマイクの近くで読み、保存WAVをヘッドホンで聞いてください。小さい声、雑音、音割れ、相手の声の混入は入力側の問題を疑います。語頭・語尾の欠けは発話検出も疑います。明瞭で完全な発話音声なのに認識だけ崩れる場合はモデル・認識設定が主な疑いです。このWAVは区切った後の音声なので、未検出で捨てた部分までは確認できません。

同じWAVでモデルを比較する場合は、仮想環境で次を実行できます（最初にモデルのダウンロードが必要な場合があります）。

```bash
python -c 'import sys; from faster_whisper import WhisperModel; m=WhisperModel(sys.argv[2], device="cpu", compute_type="int8"); segments,_=m.transcribe(sys.argv[1], language="ja", beam_size=5, vad_filter=False); print("".join(s.text for s in segments))' mic_debug/対象のファイル.wav small
```

同じコマンドの末尾だけ `medium` に変えると、同一音声で比較できます。再録音して比較すると入力条件まで変わるため、同じWAVを使ってください。
