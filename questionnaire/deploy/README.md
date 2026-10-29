# HTTPSで公開するための準備

公開先はまだ未設定です。Dockerfileはquestionnaireだけをビルド対象にし、private（回答DB・動画・招待コード・秘密鍵）をイメージに含めません。既存の解析システムやAzure設定も含めません。

リポジトリ直下からビルド：

```sh
docker build -t multirelbot-questionnaire ./questionnaire
```

公開先に必要なもの：

- HTTPSを終端するリバースプロキシ1段（アプリはProxyFixでその1段を信頼）
- 再起動後も保持される永続ディスクを `/data` にマウント
- `/data` へのUID10001の書込権限
- `QUESTIONNAIRE_PUBLIC=1`
- `QUESTIONNAIRE_BASE_URL=https://実際のホスト名`（パスなし）
- `QUESTIONNAIRE_SECRET_KEY`（32文字以上の十分にランダムな秘密鍵。管理画面のsecretとして設定）

DBを初期化して実験設定を登録し、1人につき1つ本番招待コードを発行する必要があります。ローカルの確認用回答や秘密鍵をそのまま移行せず、本番は空のDBから始めます。文字版の設定JSONは管理者だけがアクセスできる領域に転送し、`questionnaire.storage.register`で登録します。動画版を公開する場合は、動画も非公開領域 `/data/videos` に転送します。

起動後、別端末からHTTPS URLへの接続、未ログイン時の拒否、回答保存・再開、CSV出力、サービス再起動後の回答保持を検証してから本番用URLとコードを配布します。SQLiteのバックアップはsqlite3のbackup APIを使い、稼働中DBファイルの単純コピーは避けます。

文字版：`https://実際のホスト名/text`
動画版：`https://実際のホスト名/`

Renderを使う場合、SQLite保存には永続ディスク付きの有料Web Serviceが必要です。無料Web Serviceの一時ファイル領域に回答DBを置かないでください。契約・費用と公開対象の確認後にデプロイします。
