# Pepper 連携手順

このリポジトリの Python 側は、Pepper に直接 `say` できるわけではありません。
Python から Pepper に TCP で文字列を送り、Pepper 側の Android/Java アプリが受信して発話します。

## 全体構成

```text
Mac / PC
  live_intervention_loop.py
  pepper_client.py
    |
    | TCP: say:こんにちは\n
    v
Pepper Android app
  ServerSocket(port=2003)
  QiSDK Say
```

## 1. Pepper 側アプリ

Android Studio で Pepper/QiSDK の Android アプリを作り、以下のように TCP サーバを立てます。
ポート番号は `config.local.yaml` の `pepper.port` と合わせます。

`AndroidManifest.xml` にはネットワーク権限を追加します。

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

`MainActivity.java` の最小例:

```java
package com.example.peppersocket;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;

import com.aldebaran.qi.sdk.QiContext;
import com.aldebaran.qi.sdk.QiSDK;
import com.aldebaran.qi.sdk.RobotLifecycleCallbacks;
import com.aldebaran.qi.sdk.builder.SayBuilder;
import com.aldebaran.qi.sdk.design.activity.RobotActivity;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class MainActivity extends RobotActivity implements RobotLifecycleCallbacks {
    private static final int PORT = 2003;
    private QiContext qiContext;
    private volatile boolean running = true;
    private ServerSocket serverSocket;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        QiSDK.register(this, this);
        startSocketServer();
    }

    @Override
    public void onRobotFocusGained(QiContext qiContext) {
        this.qiContext = qiContext;
    }

    @Override
    public void onRobotFocusLost() {
        this.qiContext = null;
    }

    @Override
    public void onRobotFocusRefused(String reason) {
    }

    private void startSocketServer() {
        new Thread(() -> {
            try {
                serverSocket = new ServerSocket(PORT);
                while (running) {
                    Socket socket = serverSocket.accept();
                    BufferedReader reader = new BufferedReader(
                            new InputStreamReader(socket.getInputStream(), "UTF-8")
                    );
                    String line = reader.readLine();
                    socket.close();

                    if (line != null && line.startsWith("say:")) {
                        String text = line.substring(4).trim();
                        say(text);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    private void say(String text) {
        if (qiContext == null || text.isEmpty()) {
            return;
        }
        mainHandler.post(() ->
                SayBuilder.with(qiContext)
                        .withText(text)
                        .buildAsync()
                        .andThenConsume(say -> say.async().run())
        );
    }

    @Override
    protected void onDestroy() {
        running = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (Exception ignored) {
        }
        QiSDK.unregister(this, this);
        super.onDestroy();
    }
}
```

## 2. Pepper のIPを確認

Pepper と Mac/PC を同じ Wi-Fi に接続します。
Pepper 側のIPアドレスを確認して、`config.local.yaml` に設定します。

```yaml
pepper:
  ip: "192.168.11.15"
  port: 2003
  use_robot: true
```

## 3. 接続テスト

Pepper 側アプリを起動したまま、Mac/PC で実行します。

```bash
python pepper_client.py "こんにちは、接続テストです"
```

Pepper が発話すれば接続成功です。

## 4. live_intervention_loop.py から発話させる

テキストモード:

```bash
python live_intervention_loop.py --input-mode text --participants A B --analyze-every 5 --relation-history-count 5 --mode proposal --send-to-pepper
```

音声認識モード:

```bash
python live_intervention_loop.py --input-mode audio --participants A B --analyze-every 5 --relation-history-count 5 --mode proposal --send-to-pepper
```

`--send-to-pepper` を付けると、生成されたロボット発話が `pepper_client.py` 経由で Pepper に送られます。

## うまくいかない時

- Pepper と Mac/PC が同じネットワークにいるか確認する
- `config.local.yaml` の `pepper.ip` が正しいか確認する
- Pepper 側アプリが起動しているか確認する
- Pepper 側アプリのポート番号と `pepper.port` が一致しているか確認する
- Mac から `python pepper_client.py "テスト"` を先に試す
