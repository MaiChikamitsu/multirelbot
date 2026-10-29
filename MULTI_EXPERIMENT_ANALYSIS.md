# 複数会話・参加者3名ずつの分析

最新版の `analysis_config.example.json` は、同じExcel内の複数の実験を、テーマと回答者IDで抽出します。回答の行順・送信順では区切りません。各実験にA/B/Cが1人ずつ必要です。

現在の対応は以下です。

| 実験ID | 会話（Episode） | テーマ | A / B / Cの回答者ID | 評定時点数 |
|---|---|---|---|---|
| muzintou_ari | conversation_muzintou_ari | 無人島ー縛りあり | 1 / 2 / 3 | 41 |
| ai_nashi | conversation_ai_nashi | AIー縛りなし | 4 / 5 / 6 | 23 |

無人島は20261019_001309_552336を総合評価に採用しています。AIの評定207件は取り込めています。AIは実行中の20261019_154217_874494を採用runに指定しました。23時点すべて揃うまで総合値には含めません。総合ページの採用一覧で確認できます。現在は1実験だけの暫定総合値です。

## コマンド

リポジトリのルートで実行します。

```sh
python3 analyze_relations.py --config analysis_config.example.json
open estimation_accuracy/questionnaire_analysis/report.html
```

入口の「全実験の総合MAE・変化量MAEを見る」から総合ページへ進めます。従来の各実行の分析・各点の表も残ります。

AIの推定を取得する場合は、ユーザー自身で以下を実行します（23時点×10試行＝230回のLLM呼び出し）。

```sh
python3 text_relation_viewer.py --input estimation_accuracy/conversation_ai_nashi.txt --num-trials 10
```

現在指定済みのAIの実行が完了したら、そのまま分析コマンドを再実行します。別のrunを採用する場合は、完了した保存フォルダ名をAIの設定の `aggregate_run` に記入します。未完了runは総合評価から除外します。

## 2組目・次のテーマを追加する

同じテーマでも実験は別々です。2組目の命名例は `conversation_muzintou_ari_2.txt`。設定のexperimentsに別項目を追加します。

```json
{
  "id": "muzintou_ari_2",
  "enabled": true,
  "survey": "/絶対パス/最新アンケート.xlsx",
  "theme": "無人島ー縛りあり",
  "episode": "conversation_muzintou_ari_2",
  "predictions": "estimation_accuracy/viewer_runs/*/predictions.csv",
  "respondent_ids": ["7", "8", "9"],
  "roles": {"A": "7", "B": "8", "C": "9"},
  "aggregate_run": "実際に完了した保存フォルダ名",
  "condition_id": "main_prompt_gpt5_10trials_g08_s3_w6_step3",
  "index_offset": 0
}
```

ID7/8/9は説明用で、実際のIDへ置き換えてください。役割の並びも実際の割当てに合わせます。テーマはフォーム回答の文字列と完全一致させます。AIー縛りあり、AIー縛りなし、無人島ー縛りなし、外国人受け入れー縛りあり／なしも同じ方式です。会話ファイル名のローマ字部分に制限はなく、episodeを拡張子なしの実際のファイル名に合わせます。

- `enabled:false`：将来予定の実験を保留。Excelの未回答は読まない。
- `respondent_ids`：その会話を行った3人のID。同じテーマに6名以上いても3名を明示して区別。
- `roles`：IDとA/B/Cの照合。誤った組合せはエラー。
- `aggregate_run`：総合評価に採用するrunを各実験で1つだけ指定。空欄なら個別分析のみで総合には入れない。
- `condition_id`：同じプロンプト・推定方式の比較条件名。異なるプロンプトのrunは別名にし、同じ総合値へ混ぜない。

Excelを更新したら各実験のsurveyを最新版のパスへ変更してください。古いExcelの回答を新しいExcelへ移しても、行番号に依存しません。同じ回答を複数実験へ二重登録した場合はエラーにします。フォームへ実験IDを追加しなくても、回答者IDを再利用しない方針ならこの対応設定で運用できます。

## 総合指標

新規ファイルは以下です。

- overall_report.html：条件別の総合指標と採用／保留の対応表。
- overall_metrics.csv：条件・手法・mean/individual・全ペア／ペア別に集計。
- experiment_selection.csv：各実験の採用run・ID・状態。

集計方式は2種類です。

1. pooled_points：全会話の絶対誤差合計÷全評価点数。変化量MAEは全会話の区間誤差合計÷全有効区間数。
2. equal_experiments：会話ごとのMAEを算術平均。変化量MAEも会話ごとの値を算術平均（変化区間がある会話のみ、件数を併記）。

RMSEの会話均等版は、会話別の平均二乗誤差を平均してから平方根を取ります。biasも各方式の重みで集計します。

変化量は必ず同じ実験・run・ペア・回答者内の隣接番号で計算します。会話間をつないだ差分や欠測をまたぐ差分は作りません。完了した41時点の会話ならmeanは123点・120区間、23時点なら69点・66区間。2会話を合わせるとmeanは192点・186区間です。

総合値へは、その実験の全有効評定点について6手法が揃ったrunのみ採用します。途中終了runや再実行を重複して総合へ入れません。モデル名・窓・step・試行数・gamma等が同じcondition_id内で異なる場合はエラーにします。古いmetadata.jsonはプロンプト本文・ハッシュを持たないため、プロンプトの一致はcondition_idを付ける人が確認します。temperatureは保存設定値であり、実送信値の検証を意味しません。

## 動作確認

```sh
python3 -m unittest test_multi_experiment_analysis.py test_relation_analysis.py test_relation_trends.py -v
```

最新版6名のID別抽出、同一テーマ2組の分離、役割の誤対応拒否、長さの違う2会話の点数加重／会話均等集計、変化量の会話境界の除外、古いrunの重複除外、条件差の混在拒否を確認します。合成推定を実際の評価結果に混ぜません。
