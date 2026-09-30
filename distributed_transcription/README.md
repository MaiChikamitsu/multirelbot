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

入力デバイスを指定したい場合は、sounddevice のデバイス番号または名前を `--input-device` に指定します。

```bash
python participant_client.py --speaker Aさん --host 192.168.1.10 --mode mic --input-device 1
```

Whisperモデルは既定で `small` です。軽くしたい場合は `base`、精度を上げたい場合は `medium` などに変更できます。

```bash
python participant_client.py --speaker Aさん --host 192.168.1.10 --mode mic --model base
```

## 再接続

クライアントはホストPCへ接続できない場合、既定で2秒ごとに再接続を試みます。通信が一時的に切れた場合も、次の送信時に再接続して同じ発話を送り直します。

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
