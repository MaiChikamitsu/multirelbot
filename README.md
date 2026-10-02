# Multi-Party Conversation Facilitation System

## 概要
本プロジェクトは、複数人の対話をリアルタイムに録音・文字起こしし、Pyannote と SpeakerRecognition による話者分離・話者識別を行います。さらに、NetworkX と OpenAI GPT API を使って「関係性グラフ」を可視化し、よりバランスの取れた関係性を目指したロボット介入プランを自動生成・実行します。Pepper 連携に対応。

## 主な機能
1. **リアルタイム録音 & VAD（realtime_communicator.py）**  
   - WebRTC-VAD (感度モード 0–3) で沈黙検知  
   - `SILENCE_DURATION` 秒以上の無音で区切ってバッファ化

2. **話者分離・話者識別（realtime_communicator.py）**  
   - Pyannote Audio でダイアライゼーション  
   - SpeechBrain ECAPA-VOXCELEB で埋め込み比較による話者ラベリング

3. **音声文字起こし（realtime_communicator.py）**  
   - Google Cloud Speech-to-Text v1 ストリーミング

4. **関係性グラフ & 介入ロジック**  
   - **community_analyzer.py**  
     - NetworkX でエッジに GPT+EMA 評価スコア (-1.0〜+1.0) を付与  
     - 力学モデルを用いてレイアウトし、連番で画像出力  
     - Pepper への介入指示（音声＆アニメーション）を管理  
   - **intervention_planner.py**  
     - “孤立検出”“不安定三角形検出”“弱リンク促進” など複数戦略を実装  
     - 過去発話回避／モード切替（少発話者フォロー／ランダム呼びかけ）対応

5. **Web サーバ & UI（graph_realtime_web.py）**  
   - Flask + Flask-SocketIO でリアルタイム制御・更新中継  
   - **index.html**（`templates/`）と JS で、録音制御ボタン・グラフ・会話ログを表示  
   - Socket.IO イベント：  
     - `control`（録音開始／停止）  
     - `conversation_update`（発話ログ更新）  
     - `robot_speak` → `conversation_update` と `robot_log` 中継  
     - `graph_updated` → `refresh_graph` 中継（画像キャッシュ回避付き）

6. **Pepper 連携（community_analyzer.py）**  
   - TCP ソケット経由で音声合成指示を送信

7. **ロギング & 可視化**  
   - `LOG_ROOT` 配下にタイムスタンプ付きディレクトリを自動生成  
   - 会話ログ（`conversation.txt`）、ターミナルログ（`terminal.txt`）、グラフ画像（`relation_graph0.png`, `relation_graph1.png`, …）を保存  
   - Web UI でログディレクトリ名を元に画像リフレッシュ対応

## 実行手順
1. **サーバ起動**  
   python graph_realtime_web.py
   
2. **ブラウザで UI を開く**  
   URL: http://localhost:8888

3. **リアルタイム録音ワーカー起動**  
   python realtime_communicator.py

4. **対話開始**

## MAVeRD評価・追加ベースラインの実行
IEEE Access再投稿向けの比較実験は `evaluate_baselines.py` で実行します。既存の LLM-only と MAVeRD の実装本体は変更せず、同じ raw LLM score から以下の6手法を比較します。

- Interaction Frequency
- Sentiment-based
- Stance-based
- LLM-only
- LLM + SMA
- MAVeRD

### どちらの実行ファイルを使うか

目的によって、実行するファイルと入力設定が異なります。

| 目的 | 実行コマンド | 入力設定 | 人間評価CSV |
| --- | --- | --- | --- |
| MAVeRDまたはLLM-onlyだけを単体実行する | `python3 relation_estimator_from_txt.py` | `relation_estimator_from_txt.py` 上部の `INPUT_FILE` | 不要 |
| 6手法をまとめて比較し、MAE / Pearsonを算出する | `python3 evaluate_baselines.py` | `evaluation_utils.py` の `DEFAULT_EPISODES` | 必要 |

`relation_estimator_from_txt.py` は従来からある単体実行用スクリプトです。今回追加した `evaluate_baselines.py` は、既存の LLM-only / MAVeRD の実装を変更せずに追加ベースラインと比較するため、別の実行ファイルとして作成しています。

この2つは入力ファイルの設定を共有していません。たとえば `evaluation_utils.py` の `DEFAULT_EPISODES` に `conversation1.txt` を設定していても、`relation_estimator_from_txt.py` はその設定を参照せず、自身の `INPUT_FILE` を読みます。初期設定のまま単体実行する場合は `estimation_accuracy/conversation.txt` が必要です。`conversation1.txt` を単体実行したい場合は、`relation_estimator_from_txt.py` の設定を次のように変更します。

```python
INPUT_FILE = "estimation_accuracy/conversation1.txt"
```

### 文字で作成した会話ログを使う場合
音声認識を使わず、手入力・生成・文字起こし済みの会話を評価したい場合は、会話を `.txt` に保存して読み込ませます。会話ファイルは1行1発話で、必ず以下の形式にしてください。

```text
[話者名] 発話内容
```

例:

```text
[A] キャッシュレスなんて面倒なだけじゃん。現金の方がわかりやすいし安心なんだけど。
[B] は？現金の方が面倒くさいでしょ。いちいち小銭数えるのとかイライラするわ。
[C] どっちも不便なところあるだろ。キャッシュレスだって、スマホの電池切れたら何もできないし。
[ロボット] AさんもBさんも、結局使いやすくてストレスが少ない方法を求めてるって点では同じですよね。
[A] 同じって言われても納得いかないんだけど。
```

参加者名は `A`, `B`, `C` のように一貫させてください。`[ロボット]` の発話は入れても構いませんが、参加者ペアの評価対象からは除外されます。

関係スコアだけを見たい場合は、`relation_estimator_from_txt.py` 用に以下へ保存します。

```text
estimation_accuracy/conversation.txt
```

論文用の6手法比較をしたい場合は、既定では以下の2ファイルを使います。

```text
estimation_accuracy/conversation1.txt
estimation_accuracy/conversation2.txt
```

別名の会話ファイルを使いたい場合は、`relation_estimator_from_txt.py` の `INPUT_FILE`、または `evaluation_utils.py` の `DEFAULT_EPISODES` を変更してください。

6手法比較で MAE / Pearson まで出すには、会話ログに対応する人間評価CSVも必要です。既定では以下を使います。

```text
estimation_accuracy/human1.csv
estimation_accuracy/human2.csv
```

人間評価CSVは、既存ファイルと同じく最低限以下の列を持つ形式にしてください。評価値は `人の平均` 列に入れます。

```csv
識別ラベル,エピソード,セクション,ペア,人の平均
1-1-A-B,1,1,A-B,-0.746153846
1-1-B-C,1,1,B-C,-0.376923077
1-1-C-A,1,1,C-A,-0.269230769
```

`識別ラベル` は `エピソード-セクション-参加者-参加者` の形です。`セクション` が round 番号として扱われ、`C-A` のような順序でも内部で `A-C` に正規化されます。

### 6手法をまとめて実行
```bash
python3 evaluate_baselines.py \
  --output-dir estimation_accuracy/baseline_comparison \
  --max-history-human 6 \
  --num-trials 10 \
  --sma-window 3 \
  --gamma 0.8 \
  --max-history-sessions 3
```

出力ファイル:

- `estimation_accuracy/baseline_comparison/baseline_predictions.csv`: 6手法の episode / round / pair ごとの予測値。人手評価がなくても出力
- `estimation_accuracy/baseline_comparison/baseline_summary.csv`: 手法ごとの MAE / Pearson summary
- `estimation_accuracy/baseline_comparison/baseline_details.csv`: episode / round / pair ごとの prediction, human ground truth, error
- `estimation_accuracy/baseline_comparison/baseline_metadata.json`: 実行条件、モデル、パラメータ
- `estimation_accuracy/baseline_comparison/raw_scores_cache.json`: LLM-only / SMA / MAVeRD で共有する raw LLM score

会話ファイルと人間評価CSVのセッション・ペアが一致しない場合も、6手法の推定は続行して `baseline_predictions.csv` に保存します。この場合、MAE / Pearsonの計算は行わず、ターミナルと `baseline_summary.csv` に `SKIPPED` および理由を記録します。

通常実行でも、現在の処理段階、episode、round、LLM試行の進捗を簡潔に表示します。各roundの行に表示される `.` は、LLM試行が1回完了したことを表します。LLMの生スコアなども確認する場合は `--debug` を付けます。

### MAVeRDやLLM条件の調節
`evaluate_baselines.py` では、主に以下の引数で調節します。

- `--llm-model`: 関係推定に使う GPT / Azure deployment 名。省略時は `.env` の `RELATION_MODEL`、なければ `AZURE_MODEL` を使用
- `--max-history-human`: 関係推定LLMに入れる人間発話履歴数
- `--num-trials`: LLM推定の試行回数
- `--gamma`: MAVeRD の DIWS/EMA 時間減衰率
- `--max-history-sessions`: MAVeRD の EMA で保持する履歴セッション数
- `--sma-window`: LLM + SMA の直近 K 個の窓幅

例: GPTモデル、履歴長、MAVeRDパラメータを変える場合

```bash
python3 evaluate_baselines.py \
  --output-dir estimation_accuracy/baseline_comparison_gpt5_g075 \
  --llm-model gpt-5-chat \
  --max-history-human 9 \
  --num-trials 10 \
  --sma-window 3 \
  --gamma 0.75 \
  --max-history-sessions 3 \
  --refresh-raw-cache
```

`--llm-model`、`--max-history-human`、`--num-trials` を変えた場合は raw LLM score も変わるため、`--refresh-raw-cache` を付けて再生成してください。`--gamma`、`--max-history-sessions`、`--sma-window` だけを変える場合は、同じ raw score cache を再利用できます。

### 既存MAVeRDのパラメータ探索
既存の MAVeRD grid search は `parameter_grid_search.py` で実行します。

```bash
python3 parameter_grid_search.py
```

探索範囲は `parameter_grid_search.py` 上部の以下を編集します。

```python
LLM_MODELS = ["gpt-4.1", "gpt-5-chat"]
GAMMAS = [...]
MAX_HISTORY_SESSIONS_LIST = [2, 3]
MAX_HISTORY_HUMAN_LIST = [6]
NUM_TRIALS = 5
```

### 既存の単体関係推定スクリプト
従来の単体実行は `relation_estimator_from_txt.py` を使います。

```bash
python3 relation_estimator_from_txt.py
```

これは `estimation_accuracy/conversation.txt` のような文字会話ログから、LLM-onlyまたはMAVeRDの関係スコアCSVを生成する用途です。人間評価CSVがなくても実行できます。

調節箇所は `relation_estimator_from_txt.py` 上部です。

```python
INPUT_FILE = "estimation_accuracy/conversation.txt"
OUTPUT_FILE = "estimation_accuracy/relation_scores.csv"
LLM_MODEL = "gpt-4.1"
USE_EMA = True
GAMMA = 0.8
MAX_HISTORY_SESSIONS = 3
MAX_HISTORY_HUMAN = 9
NUM_TRIALS = 5
```

`USE_EMA = True` なら MAVeRD、`USE_EMA = False` ならEMAなしのLLM-onlyとして出力します。

`LLM_MODEL = None` にした場合は `config.local.yaml` / 環境変数側の relation model 設定を使用します。関係推定時の temperature は `config.local.yaml` の `llm.relation_temperature` で調節します。

### テスト
追加ベースラインと既存 EMA 計算の regression test は以下で実行します。

```bash
pytest -q
```

## パラメータ。括弧内はデフォルト値
### realtime_communicator.py
- **SILENCE_DURATION**: 無音時間（0.5秒）
- **N_BATCH**: セッション分割判定の頻度（5発話）
- **USE_GOOGLE_STT**: 音声認識モデルの設定 (gcp v1)
- **USE_DIRECT_STREAM**: マイクから直接ストリーミング (False)
- **DIARIZATION_THRESHOLD**: この秒数以上なら話者分離する（5秒）
- **SKIP_THRESHOLD_BYTES**: このバイト数以下なら処理スキップ (30000バイト←大体「うん」以上)
- **register_reference_speaker**: 話者を登録
- **process_audio** か **process_audio_batch** か: セッション判定の頻度（5発話に1回）
- **add_utterance** か **add_utterance_count** か: セッションを意味判定か、固定発話数（10）か

### session_manager.py
- **utterances_per_session**: 何会話分を分析するか（10発話）
- **analyze_every**: 何発話ごとに関係性推定するか（5発話）

### community_analyzer.py
- **pepper_ip**: IPアドレス
- **use_robot**: Pepperを使用するか（True）
- **robot_included**: ロボットを関係性学習に組み込むか（機能しないかも）（False）
- **mode**: 条件切り替え
- **rc.set_robot_count(5)**: 介入決定してから何発話、ロボットを話者識別に組み込むか（5）
