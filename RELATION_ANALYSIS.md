# 自動保存とアンケート比較

## まず普段どおり実行

リポジトリのルートで実行します。

```sh
python3 text_relation_viewer.py --input estimation_accuracy/conversation_muzintou_ari.txt
```

ブラウザの開始ボタンを押し、推定するたびに以下に追記されます。

```
estimation_accuracy/viewer_runs/実行日時_マイクロ秒/
  metadata.json
  predictions.csv
```

別の実行は別フォルダになります。途中で終了した場合も、完了した推定のCSVは残ります。起動しただけでは推定APIを呼びません。CSVは最初の推定成功時に作られます。`--output-dir` で保存先の親フォルダを変更できます。

CSVは1行＝1手法・1時点・1ペアです。3人の場合1時点18行、今回の123人間発話では41時点738行です。

| 列 | 定義 |
|---|---|
|Method|6手法の名前|
|Episode|入力ファイル名（拡張子なし）|
|Run|保存フォルダ名。同条件の反復も別run|
|Round|1始まりの推定番号。既定では人間3発話ごと|
|Pair|A–B / A–C / B–C。無向ペア|
|Prediction|[-1,1]の推定値|
|Human_Turn|推定までの累積人間発話数（3,6,…123）|
|Log_Index|パース済み会話の0始まりの位置。原文の物理行番号ではない|
|Window_Start_Log_Index|推定に使った窓の先頭。0始まり|
|Estimated_At|API推定が終わり保存した実行時刻。会話動画の時刻ではない|
|Interaction_Intensity|Interaction Frequencyのみ補助的な相互作用強度。他手法は空欄|

`LLM-only` は生のLLM出力、`LLM + SMA` は同じ生値の直近3推定の単純移動平均、`MAVeRD` は既存EMAScorerによるDIWS/EMAです。window_size既定6は「直近の人間6発話」、step既定3は「推定間隔」。最終端数は推定しません。ロボット発話はstepに含めず、窓内には残ります。

Interaction Frequencyの符号付きPredictionは既存仕様どおり常に0で、強度は正負の関係性ではありません。強度をMAEのPredictionに置き換えません。

metadata.jsonには入力の絶対パス・SHA256・step・窓・モデル名・温度・SMA幅・EMA設定を保存します。APIキーは保存しません。

## 10試行平均をviewerで使う

```sh
python3 text_relation_viewer.py --input estimation_accuracy/conversation_muzintou_ari.txt --num-trials 10
```

開始ボタンを押すと、各時点で同じ会話窓について10回LLMを呼びます。各試行番号に独立したSMA・EMAの状態を持ち、時系列を通じて保持します。LLM-onlyは10個の生値の平均、LLM + SMAとMAVeRDは各試行の平滑化後の値の平均を表示・保存します。生値を先に平均してEMAへ渡す方式ではありません。

追加依存はありません。`--num-trials`省略時は従来どおり1試行です。今回41時点なら10試行で410回のAPI呼び出しになります。3ペアは1回の応答でまとめて取得します。LLMを使わない3手法は時点ごとに1回計算します。

同じrunの保存フォルダ内で、`predictions.csv`には6手法の代表値（LLM系は試行平均）、`trial_predictions.csv`にはLLM系3手法の各試行の値とTrial番号を残します。`Num_Trials`列とmetadata.jsonにも試行数を記録します。分析スクリプトは従来どおりpredictions.csvだけを読み、試行別CSVは読みません。

ある時点でAPI呼び出しが失敗した場合、完了した前の時点の平均は保持し、その時点の一部試行だけを平均して公開しません。成功済みの試行別値はtrial_predictions.csvに残ります。途中再開には対応していません。

保存後は同じ分析コマンドです。

```sh
python3 analyze_relations.py --config analysis_config.example.json
open estimation_accuracy/questionnaire_analysis/report.html
```

実行ごとに別run・別比較ページになるので、過去の1試行の結果と混ざりません。MAEは10試行平均の推定値と人の平均評定との誤差です。各試行のMAEの平均とは区別してください。

## 保存後にアンケートを比較

`analysis_config.example.json` の survey（Excelのパス）、theme（完全一致）、predictionsを指定します。現在の例は今回のテーマに設定済みです。

```sh
python3 analyze_relations.py --config analysis_config.example.json
```

追加依存はありません。標準ライブラリでExcelとCSVを読み、CSV・SVG・HTMLを生成します。結果は `estimation_accuracy/questionnaire_analysis/report.html` をブラウザで開いてください。

- human_ratings.csv: フォームの回答 2だけを縦持ちに整形。回答者・役割・テーマ・番号・ペア・値・元セル。
- predictions.csv: 対象Episodeの保存済み推定値。
- point_comparisons.csv: 各推定と全員平均（mean）および各回答者（individual）の比較、符号付き誤差と絶対誤差。
- metrics.csv: 実験・手法・run・対象別、およびペア別・回答者別のn、MAE、RMSE、bias、Pearson、同順位対応Spearman。定数系列の相関は空欄。
- coverage.csv: 人の評定がない推定点、推定値のない評定点。空欄は0補完せず、末尾42〜54も評定点にしません。
- report.html / ペア別SVG: 6手法の推移と平均評定・回答者の最小最大、指標表、対応漏れ。runが複数ならすべて線を表示。
- provenance.json: 設定と入力ファイルのSHA256。

MAE(mean)は各番号・ペアの回答者平均に対する誤差で、MAE(individual)は各人の評定に対する誤差です。反復runは混ぜず別集計し、異なるrunの推定を平均して誤差を小さくする処理はしません。欠測がある場合はnとcoverageを確認して比較してください。推定がある点だけを集計するため、手法間でnが違う場合の単純順位づけには注意が必要です。

## 次回データ

1. 新しい会話を追加し、同じviewerコマンドを新しい `--input` で実行。
2. Excelを保存し、設定のexperimentsにid・survey・theme・episode・predictionsを追加。episodeは会話ファイル名（拡張子なし）。異なる実験には一意なidを付ける。
3. 同じ分析コマンドを再実行。結果は現在の入力に基づき再生成されます。

同じ実験の再実行は設定追加不要です。globで新しいrunも読み込みます。同一テーマの別セッションを同じExcelへ混ぜる場合、現状のID・役割・テーマだけでは識別できないので、Excelをセッション別に分けてください。重複回答と重複推定キーはエラーにします。

通常はindex_offset=0です。人手で実際の対応が確定した場合だけ変更します。時刻を使った自動照合や相関がよくなるような自動シフトは行いません。

## 確認したことと未確認事項

- 今回はA=ID1、B=ID2、C=ID3。ExcelのCとAは無向のA-Cとして対応。
- 123人間発話→41推定時点。Excelは3人×41番号×3ペア＝369評定。+0.3などは数値化し、42〜54の空欄を除外。フォームの回答 1・自由記述・送信時刻は正解値に使いません。
- 既存baseline_comparison/baseline_predictions.csvはEpisode1の29時点で今回の会話ではありません。raw_scores_cache.jsonには試行別の生値、baseline_predictions.csvには既存評価コードが集約したPredictionが入ります。relation_scores.csvにはペア・ラウンド、各試行、平均があり、生/平滑化や対象会話がファイル単独で確定しないため今回のデータには流用しません。
- viewerは従来メモリ内のhistoryのみでした。今回追加する自動保存で今後の結果をそのまま分析可能にします。
- 番号と3発話の対応はコードと質問票仕様の一致を確認しましたが、動画と書き起こしの分割が実際に各評定タイミングと一致するか、役割の実人物対応、尺度の心理的意味が同じかは実験担当者による確認が必要です。会話ファイルに動画の経過秒はありません。

## 検証

```sh
python3 -m unittest test_relation_analysis.py test_viewer_autosave.py -v
```

API応答を固定したviewerの自動保存で、2時点×6手法×3ペア、発話位置、別実行の保持を確認。Excel実データの369評定、6手法の保存済み生値からの再開・再実行一致、SMAの独立計算、欠測・重複拒否・相関の境界値も検証します。テスト内の合成推定は実データの評価結果として保存しません。

今回、実API推定は実行許可が得られず未実行です。API不要の3手法だけの実データ結果は別途analysis_outputにあり、LLM由来3手法の欠測も明示しています。

補助の `collect_relation_predictions.py` はviewerを操作せず同じ推定を実行する場合に使えますが、普段のviewer実行には不要です。既定でAPIを呼び、`.raw.json`に時点ごとの生値を保存して途中再開します。新しいモデル条件は別outputにしてください。
