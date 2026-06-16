import os
import threading
import queue
from collections import defaultdict, deque
from typing import Dict, Tuple
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
import matplotlib
import socketio
import re
import time
import socket
from intervention_planner import InterventionPlanner
from datetime import datetime
import config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from log_filtering import filter_logs_by_human_count
# ここから追加
from ingroup_detector import detect_ingroup_outgroup

# Pepper設定（config.local.yamlから読み込み）
load_dotenv()
_CFG_LOADED = config.get_config()
# Pepperの設定
pepper_ip = getattr(_CFG_LOADED.pepper, "ip", "192.168.11.15")  # PepperのIPアドレス
pepper_port = getattr(_CFG_LOADED.pepper, "port", 2003)  # Android アプリのポート
use_robot = getattr(_CFG_LOADED.pepper, "use_robot", True)  # Pepperを使用するかどうか
robot_included = getattr(
    _CFG_LOADED.pepper, "robot_included", True
)  # ロボットを関係性学習に組み込むかどうか
mode = getattr(
    _CFG_LOADED.intervention, "mode", "proposal"
)  # 介入モード ("proposal", "few_utterances", "random_target")


matplotlib.use("Agg")  # GUI 非対応の描画エンジンを指定
socketio_cli = socketio.Client()

_CFG = config.get_config()
plt.rcParams["font.family"] = "Hiragino Sans"


def try_connect_socketio(url="http://localhost:8888", max_retries=10, interval_sec=2):
    """Flask Socket.IO サーバへの接続をリトライしながら試みる"""
    for attempt in range(1, max_retries + 1):
        try:
            socketio_cli.connect(url)
            print("✅ Socket.IO に接続しました")
            return
        except Exception as e:
            print(f"⚠️ Socket.IO 接続失敗 (試行 {attempt}/{max_retries}): {e}")
            time.sleep(interval_sec)
    print("❌ Socket.IO に接続できませんでした。Web UI 機能は無効になります。")


try_connect_socketio()


def send_to_pepper(message: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((pepper_ip, pepper_port))
            s.sendall(f"say:{message}\n".encode("utf-8"))
            # 応答待ち（オプション）
            # response = s.recv(1024).decode('utf-8')
            # print(f"🤖 Pepper応答: {response}")
    except Exception as e:
        print(f"⚠️ Pepperへの送信失敗: {e}")


def send_to_pepper_async(message: str):
    threading.Thread(target=send_to_pepper, args=(message,), daemon=True).start()


# === Pepperの腕を持ち上げるアニメーションを送信 ===
def perform_arm_lift(self, seconds=2):
    """Pepperの腕をseconds秒間持ち上げるアニメーション指示を送信"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((pepper_ip, pepper_port))
            # アニメーションコマンド: "anim:<動作名>,<秒数>"
            s.sendall(f"anim:liftRightArm,{seconds}\n".encode("utf-8"))
    except Exception as e:
        print(f"⚠️ Pepperアニメーション送信失敗: {e}")


def perform_arm_lift_async(self):
    threading.Thread(target=self.perform_arm_lift, daemon=True).start()


class CommunityAnalyzer:
    def __init__(self, gamma=0.8, max_history_sessions=3):
        self.graph_count = 0
        self.scores = defaultdict(float)
        # config.yamlからuse_ema、gamma、max_history_sessionsを読み込み
        scorer_cfg = getattr(_CFG, "scorer", None)
        if scorer_cfg:
            self.use_ema = getattr(scorer_cfg, "use_ema", True)
            self.gamma = getattr(scorer_cfg, "gamma", gamma)
            self.max_history_sessions = getattr(scorer_cfg, "max_history_sessions", max_history_sessions)
        else:
            self.use_ema = True
            self.gamma = gamma
            self.max_history_sessions = max_history_sessions
        self.history = defaultdict(
            lambda: deque(maxlen=self.max_history_sessions)
        )  # ペアごとの過去発話数
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        # 最後に取得した GPT スコア / EMA スコア（テスト用に外部参照可能）
        self.last_gpt_scores = {}
        self.last_ema_scores = {}
        # ロボットの過去発話（リアルタイム環境用）
        self.past_robot_utterances = []

    def reset_ema(self):
        """EMA履歴とスコアをリセット（新しいエピソード開始時）"""
        self.scores = defaultdict(float)
        self.history = defaultdict(lambda: deque(maxlen=self.max_history_sessions))
        self.past_robot_utterances = []
        print("🔄 EMA履歴とロボット発話履歴をリセットしました")

    # === ワーカースレッドで非同期に処理 ===
    def _worker(self):
        while True:
            session = self.task_queue.get()
            self._analyze(session)
            self.task_queue.task_done()

    # === 5発話ごとの外部トリガに対する統一エントリ ===
    def update_with_robot_if_enabled(self, session_logs):
        """
        robot_included の設定に応じて、以下の動作を行う：
        - True: 先にロボット発話を生成し、ログに付け足してから update() を一度だけ実行
        - False: そのまま update()（解析後に _run_intervention() が発火してロボットが喋る）
        """
        if robot_included:
            robot_log = self._plan_and_speak_robot(session_logs)
            if robot_log:
                combined = list(session_logs) + [robot_log]
                return self.update(combined)
            # 介入不要ならそのまま単回解析
            return self.update(session_logs)
        else:
            return self.update(session_logs)

    def update(self, session):
        self.task_queue.put(session)
        # print(f"関係性推定開始：{datetime.now()}")

    # === 同期的な関係性推定（simulation_environment用） ===
    def estimate_relation(self, session_logs):
        """
        与えられたログから関係性スコアを同期的に推定する。
        simulation_environment.pyから呼び出される。
        """
        if robot_included:
            participants = list({log["speaker"] for log in session_logs})
        else:
            participants = [
                p for p in {log["speaker"] for log in session_logs} if p != "ロボット"
            ]

        if len(participants) < 2:
            return {}

        # GPTのスコアを取得
        gpt_scores = self._get_gpt_friendship_scores(session_logs, participants)

        #ここから追加
        group_structure = detect_ingroup_outgroup(
            session.logs,
            gpt_scores
        )

        print(group_structure)

        #ここまで追加

        # 履歴に基づくEMAを同期的に計算（simulation用）
        self.last_gpt_scores = gpt_scores
        ema_scores = {}

        # セッション内の発話数を記録
        # 注: session_logsには直近max_history_relation発話が含まれるが、
        # EMAのsession_utteranceは「今回のセッション」のみをカウントすべき
        # シミュレーションでは num_participants ごとに評価するため、
        # 最後の num_participants 発話のみをカウント
        utterance_counts = defaultdict(int)
        num_participants = len(participants)
        # 最後の num_participants 発話のみをカウント
        recent_logs = (
            session_logs[-num_participants:]
            if len(session_logs) >= num_participants
            else session_logs
        )
        for log in recent_logs:
            utterance_counts[log["speaker"]] += 1

        # ソート順で処理することで出力順序を統一
        for key in sorted(gpt_scores.keys()):
            x_t = gpt_scores[key]
            session_utterance = min(
                utterance_counts.get(key[0], 0), utterance_counts.get(key[1], 0)
            )
            past_utterances = self.history[key]
            total_past = sum(past_utterances)  # 過去のセッション(t-1, t-2, ...)
            # α = d × min(n_i^(t), n_j^(t)) / Σ(k=t-2 to t) min(n_i^(k), n_j^(k))
            # 分母は過去3セッション(t, t-1, t-2)の合計
            total_with_current = total_past + session_utterance

            # 初回ならそのままセット
            if key not in self.scores:
                self.scores[key] = x_t
                ema_scores[key] = x_t
                if self.use_ema:
                    print(
                        f"🔢 α=1.00 (初回)"
                    )
                    print(f"🔁 EMA初期化: {key} = {x_t:+.1f}")
            else:
                if self.use_ema:
                    # 過去発話数に時間減衰を適用
                    weighted_past = 0
                    for i, count in enumerate(past_utterances):
                        sessions_ago = len(past_utterances) - i
                        weight = self.gamma ** sessions_ago
                        weighted_past += count * weight

                    # αの計算
                    weighted_total = weighted_past + session_utterance
                    alpha = min(1.0, session_utterance / weighted_total) if weighted_total > 0 else 1.0

                    prev = self.scores[key]
                    updated = alpha * x_t + (1 - alpha) * prev
                    self.scores[key] = updated
                    ema_scores[key] = updated
                    print(
                        f"🔢 α計算: {key}, session={session_utterance}, weighted_past={weighted_past:.2f}, total={weighted_total:.2f}, α={alpha:.2f}"
                    )
                    print(
                        f"🔁 EMA更新: {key} = {alpha:.2f}×{x_t:+.1f} + {(1-alpha):.2f}×{prev:+.1f} → {updated:+.1f}"
                    )
                else:
                    # EMA無効時は GPT スコアで直接更新
                    self.scores[key] = x_t
                    ema_scores[key] = x_t

            # 履歴に現在のセッション発話数を追加
            self.history[key].append(session_utterance)

        self.last_ema_scores = ema_scores

        # 出力: GPTスコアとEMA（必要あれば）- ソート済みキーで表示
        sorted_gpt = {
            k: round(v, 1) for k in sorted(gpt_scores.keys()) for v in [gpt_scores[k]]
        }
        print(f"🧠 GPTスコア: {sorted_gpt}")
        if self.use_ema:
            sorted_ema = {
                k: round(v, 1)
                for k in sorted(ema_scores.keys())
                for v in [ema_scores[k]]
            }
            print(f"🔁 EMAスコア: {sorted_ema}")

        return ema_scores if self.use_ema else gpt_scores

    # === 三角形の構造分析（simulation_environment用） ===
    def analyze_triangles(
        self, graph: nx.Graph
    ) -> Dict[Tuple[str, str, str], Tuple[str, float]]:
        """
        与えられたグラフから三角形の構造とスコア平均を計算する。
        simulation_environment.pyから呼び出される。
        """
        from itertools import combinations

        def triangle_structure(a, b, c, g):
            s = {
                (a, b): g[a][b]["score"],
                (b, c): g[b][c]["score"],
                (c, a): g[c][a]["score"],
            }
            signs = ["+" if s[pair] >= 0 else "-" for pair in [(a, b), (b, c), (c, a)]]
            return "".join(signs), sum(s.values()) / 3

        triangle_scores = {}
        for a, b, c in combinations(graph.nodes, 3):
            if graph.has_edge(a, b) and graph.has_edge(b, c) and graph.has_edge(c, a):
                struct, avg = triangle_structure(a, b, c, graph)
                triangle_scores[(a, b, c)] = (struct, avg)
        return triangle_scores

    # === セッションごと関係性を更新 ===
    def _analyze(self, session_logs):
        # robot_includedの設定に応じてロボットを除外
        if robot_included:
            participants = list({log["speaker"] for log in session_logs})
        else:
            participants = [
                p for p in {log["speaker"] for log in session_logs} if p != "ロボット"
            ]
        print(f"👥 参加者: {participants} (robot_included={robot_included})")
        if len(participants) < 2:
            print("⚠️ 参加者が1人以下のためスキップ")
            return

        # セッション内の発話数を記録
        utterance_counts = defaultdict(int)
        for log in session_logs:
            utterance_counts[log["speaker"]] += 1
        print(f"🗣️ 発話数: {utterance_counts}")

        # 関係性LLMへは max_history_relation 個の人間発話のみを渡す（ロボット発話を除外）
        max_history_relation = getattr(_CFG, "env", None)
        if max_history_relation:
            max_history_relation = (
                getattr(max_history_relation, "max_history_relation", 3) or 3
            )
        else:
            max_history_relation = 3

        filtered_logs_for_relation = filter_logs_by_human_count(
            session_logs, max_history_relation, exclude_robot=False
        )

        gpt_scores = self._get_gpt_friendship_scores(
            filtered_logs_for_relation, participants
        )
        print(f"🧠 GPTスコア (1桁): { {k: round(v,1) for k,v in gpt_scores.items()} }")
        print(
            f"⚙️ EMA使用: {'✅ 有効' if self.use_ema else '❌ 無効'} (gamma={self.gamma})"
        )

        for (a, b), score in gpt_scores.items():
            key = tuple(sorted([a, b]))
            session_utterance = min(utterance_counts[a], utterance_counts[b])

            # 過去の発話数を時間減衰を考慮して合計
            past_utterances = list(self.history[key])
            weighted_past = 0
            for i, count in enumerate(past_utterances):
                sessions_ago = len(past_utterances) - i
                weight = self.gamma ** sessions_ago
                weighted_past += count * weight

            x_t = score
            if key not in self.scores:
                self.scores[key] = x_t
                print(f"🆕 初期スコア: {key} = {x_t:.1f}")
            else:
                if self.use_ema:
                    # EMA（指数移動平均）を使用
                    # α = min(n_i^(t), n_j^(t)) / (min(n_i^(t), n_j^(t)) + Σ gamma^(sessions_ago) × min(n_i^(k), n_j^(k)))
                    weighted_total = weighted_past + session_utterance
                    alpha = min(1.0, session_utterance / weighted_total) if weighted_total > 0 else 1.0
                    print(
                        f"🔢 α計算: {key}, session={session_utterance}, weighted_past={weighted_past:.2f}, total={weighted_total:.2f}, α={alpha:.2f}"
                    )
                    prev = self.scores[key]
                    updated = alpha * x_t + (1 - alpha) * prev
                    self.scores[key] = updated
                    print(
                        f"🔁 EMA更新: {key} = {alpha:.2f}×{x_t:.1f} + {(1-alpha):.2f}×{prev:.1f} → {updated:.1f}"
                    )
                else:
                    # EMAを使わない場合は直接上書き
                    self.scores[key] = x_t
                    print(f"🔄 スコア更新（EMA無効）: {key} = {x_t:.1f}")
            # 履歴に追加（発話数のみ）
            self.history[key].append(session_utterance)

        self._draw_graph()

        if socketio_cli.connected:
            socketio_cli.emit(
                "graph_updated",
                {
                    "index": self.graph_count,
                    "log_root": os.path.basename(config.LOG_ROOT),
                },
            )  # === グラフ更新通知をWebに送信 ===

        # print(f"関係性推定終了：{datetime.now()}")

        if "ロボット" not in participants:
            # ロボット介入戦略の実行
            self._run_intervention(session_logs=session_logs)

    # === GPTに仲の良さを尋ねるプロンプト ===
    def _get_gpt_friendship_scores(self, session_logs, participants):
        conversation = "\n".join(
            [f"[{log['speaker']}] {log['utterance']}" for log in session_logs]
        )
        pair_lines = "\n".join(
            [f"- {a} × {b}" for a, b in combinations(participants, 2)]
        )
        output_format = "\n".join(
            [f"{a}-{b}: " for a, b in combinations(participants, 2)]
        )

        prompt = f"""
以下の会話を読み、{', '.join(participants)}の「仲の良さ（親密度）」を -1.0 〜 +1.0 の間の**実数（小数第1位まで）**で評価してください。
0.0 は特に親しさも対立も感じない「中立的な状態」です。
そこから -1.0（強い対立） 〜 +1.0（非常に親しい） に向けて、どれくらい離れているかを評価してください。
出力形式を厳守し、理由・補足説明などは一切加えないでください。

具体例：
-1.0 例：皮肉、批判、無視、相手を無視して話を進める
+1.0 例：共感、褒める、相手に話題を振る、一緒に行動する

評価対象は以下のペアです（ロボットは含みません）：
{pair_lines}

会話：
{conversation}

出力形式：
{output_format}
"""

        # Azure OpenAI (RELATION_MODEL) を使用
        client, deployment = get_azure_chat_completion_client(
            _CFG.llm, model_type="relation"
        )
        if not client or not deployment:
            raise RuntimeError(
                "Failed to obtain Azure OpenAI client for relation scoring."
            )

        messages = [{"role": "user", "content": prompt}]
        # config.yamlから関係性推定用の温度パラメータを読み込み
        relation_temperature = getattr(_CFG.llm, "relation_temperature", 0.3)
        params = build_chat_completion_params(
            deployment, messages, _CFG.llm, temperature=relation_temperature
        )
        res = client.chat.completions.create(**params)

        return self._parse_scores_from_response(res.choices[0].message.content)

    # === GPTの返答を辞書に変換 ===
    def _parse_scores_from_response(self, response_text):
        """GPT 応答 (例: "A-B: +0.50") を安全にパースして辞書を返す。

        出力には説明や複数行が混ざることがあるため、ペアの形式が見つからない
        行は無視する。デバッグモードが有効ならば無視した行を警告する。
        """
        scores = {}
        lines = response_text.strip().split("\n")
        debug = False
        # configからデバッグフラグを取得
        try:
            debug = bool(getattr(_CFG.env, "debug", False))
        except Exception:
            debug = False

        i = 0
        while i < len(lines):
            line = lines[i]
            # ペア (A-B) を本文中から見つける（余分な説明行をスキップ）
            # スキップ: 明らかに説明行（評価理由等）のみ
            if re.match(r"^\s*(評価理由|理由)\W*", line):
                if debug:
                    print(f"⚠️ 説明行をスキップ: {line.strip()}")
                continue

            # 最初に A-B 形式を検索
            pair_match = re.search(r"([^\s:]+)\s*-\s*([^\s:]+)", line)
            if not pair_match:
                # 次に日本語の "XとY" 形式を検索 (例: "CとAは〜(+0.7)")
                pair_match = re.search(
                    r"([A-Za-z0-9_\u4E00-\u9FFF\u3040-\u30FF]+)と([A-Za-z0-9_\u4E00-\u9FFF\u3040-\u30FF]+)",
                    line,
                )
                if not pair_match:
                    if debug:
                        print(
                            f"⚠️ ペア形式エラー（期待: A-B もしくは XとY）: {line.strip()}"
                        )
                    continue

            a, b = pair_match.group(1).strip(), pair_match.group(2).strip()

            # 数値部分だけを抽出（-1.0 ～ 1.0 を想定）: + または - を許可
            match = re.search(r"[+-]?\d+(?:\.\d+)?", line)
            # lookahead: 現在の行にスコアが無ければ、次数行を参照してみる
            lookahead = 0
            while match is None and lookahead < 2 and (i + 1 + lookahead) < len(lines):
                next_line = lines[i + 1 + lookahead]
                # skip reason-only lines
                if re.match(r"^\s*(評価理由|理由)\W*", next_line):
                    lookahead += 1
                    continue
                match = re.search(r"[+-]?\d+(?:\.\d+)?", next_line)
                if match:
                    # advance i to consume the line with score so we don't double-parse
                    i = i + 1 + lookahead
                    break
                lookahead += 1
            if match:
                key = tuple(sorted([a, b]))
                try:
                    scores[key] = float(match.group())
                except ValueError:
                    if debug:
                        print(f"⚠️ 数値変換失敗: {line.strip()}")
            else:
                # ペアが見つかったが値が見つからない場合、デバッグ時にだけ通知
                if debug:
                    print(f"⚠️ 数値抽出失敗: {a}-{b} (lookaheadで見つからず)")
            i += 1
        return scores

    def _run_intervention(self, session_logs):
        """
        関係性グラフ・三角形構造をもとに、介入戦略を自動生成して実行する
        """
        # print(f"ロボット介入開始：{datetime.now()}")

        # 介入判定・ロボット発話LLMへは intervention_max_history 個の人間発話 + その間のロボット発話を渡す
        intervention_max_history = getattr(_CFG, "env", None)
        if intervention_max_history:
            intervention_max_history = (
                getattr(intervention_max_history, "intervention_max_history", 9) or 9
            )
        else:
            intervention_max_history = 9

        filtered_logs_for_intervention = filter_logs_by_human_count(
            session_logs, intervention_max_history, exclude_robot=False
        )

        planner = InterventionPlanner(
            graph=self._build_graph_object(),
            triangle_scores=self._compute_triangle_scores(),
            mode=mode,
            num_participants=getattr(_CFG.participants, "num_participants", 3),
            isolation_threshold=getattr(_CFG.intervention, "isolation_threshold", 0.0),
            past_utterances=self.past_robot_utterances,
        )
        plan = planner.plan_intervention(session_logs=filtered_logs_for_intervention)

        if not plan:
            print("🤖 介入対象なし（安定状態）")
            return
        else:
            if use_robot:
                # Pepperの腕を2秒間持ち上げるアニメーション
                # self.perform_arm_lift_async()
                # フィラーとして「あのー」を挟む
                send_to_pepper_async("あのー")

        utterance = planner.generate_robot_utterance(
            plan, filtered_logs_for_intervention
        )

        # ロボットの発話ログを構成
        robot_log = {
            "time": datetime.now(),
            "speaker": "ロボット",
            "utterance": utterance,
        }

        # ロボット発話したら、次のN発話分はロボットを識別対象にする
        import realtime_communicator as rc

        robot_count = getattr(_CFG.realtime, "robot_count_after_intervention", 5)
        rc.set_robot_count(robot_count)
        print(f"🤖 ロボット発話を識別対象に設定（次の{robot_count}発話分）")

        # 画面表示
        if socketio_cli.connected:
            socketio_cli.emit(
                "robot_speak", {"speaker": "ロボット", "utterance": utterance}
            )
            print(f"💬 ロボット発話送信: {utterance}")

        if use_robot:
            # --- Pepperに喋らせる（非同期） ---
            send_to_pepper_async(utterance)

        # print(f"ロボット介入終了：{datetime.now()}")

        if robot_included:
            # 🔁 ロボットの発話も関係性学習に反映させる
            session_with_robot = session_logs + [robot_log]
            self.update(session_with_robot)

    def _plan_and_speak_robot(self, session_logs):
        """
        『先にロボットを喋らせ、必要なら robot_log を返す』ためのヘルパ。
        - robot_included=True の「先ロボ→単回解析」パスで利用
        - ここでは update() は呼ばない（呼び出し側で単回 update するため）
        """
        # 介入判定・ロボット発話LLMへは intervention_max_history 個の人間発話 + その間のロボット発話を渡す
        intervention_max_history = getattr(_CFG, "env", None)
        if intervention_max_history:
            intervention_max_history = (
                getattr(intervention_max_history, "intervention_max_history", 9) or 9
            )
        else:
            intervention_max_history = 9

        filtered_logs_for_intervention = filter_logs_by_human_count(
            session_logs, intervention_max_history, exclude_robot=False
        )

        planner = InterventionPlanner(
            graph=self._build_graph_object(),
            triangle_scores=self._compute_triangle_scores(),
            mode=mode,
            num_participants=getattr(_CFG.participants, "num_participants", 3),
            isolation_threshold=getattr(_CFG.intervention, "isolation_threshold", 0.0),
            past_utterances=self.past_robot_utterances,
        )
        plan = planner.plan_intervention(session_logs=filtered_logs_for_intervention)
        if not plan:
            print("🤖 介入対象なし（安定状態）")
            return None
        if use_robot:
            send_to_pepper_async("あのー")  # フィラー
        utterance = planner.generate_robot_utterance(
            plan, filtered_logs_for_intervention
        )
        robot_log = {
            "time": datetime.now(),
            "speaker": "ロボット",
            "utterance": utterance,
        }
        # 次のN発話分ロボット識別対象
        import realtime_communicator as rc

        robot_count = getattr(_CFG.realtime, "robot_count_after_intervention", 5)
        rc.set_robot_count(robot_count)
        print(f"🤖 ロボット発話を識別対象に設定（次の{robot_count}発話分）")
        if socketio_cli.connected:
            socketio_cli.emit(
                "robot_speak", {"speaker": "ロボット", "utterance": utterance}
            )
            print(f"💬 ロボット発話送信: {utterance}")
        if use_robot:
            send_to_pepper_async(utterance)
        return robot_log

    def _build_graph_object(self) -> nx.Graph:
        G = nx.Graph()
        for (a, b), score in self.scores.items():
            a, b = sorted([a, b])
            G.add_edge(a, b, score=score)
        return G

    # === NetworkXでグラフ描画 ===
    def _draw_graph(self):
        G = self._build_graph_object()

        # 1. まず計算だけする
        preliminary_weights = {}
        for u, v in G.edges():
            score = G[u][v]["score"]
            scaled = (score + 1.0) / 2.0  # 0.0～1.0に変換
            preliminary_weights[(u, v)] = scaled**6 * 200

        # 2. 最大weightを取得
        max_weight = max(preliminary_weights.values())
        min_weight = max_weight / 7
        print(f"🔎 最大weight: {max_weight:.2f}, 最小weight設定: {min_weight:.2f}")

        # 3. 最小値を考慮してweightをセット
        for (u, v), prelim in preliminary_weights.items():
            adjusted_weight = max(min_weight, prelim)
            G[u][v]["weight"] = adjusted_weight
            print(
                f"スコア調整: {u} - {v}: prelim={prelim:.2f} → weight={adjusted_weight:.2f}"
            )

        # 配置計算（スコアが強いほど引き合う）
        pos = nx.spring_layout(G, weight="weight", seed=42)

        # 線の太さ（親密度の強さに応じて）
        edge_weights = [max(0.5, 5 * abs(G[u][v]["score"])) for u, v in G.edges()]

        # スコアラベル（±1.0）
        edge_labels = {(u, v): f"{G[u][v]['score']:.1f}" for u, v in G.edges()}

        # 敵対関係は赤、親密な関係は青。0以上は青、0未満は赤
        edge_colors = [
            "red" if G[u][v]["score"] < 0 else "skyblue" for u, v in G.edges()
        ]

        plt.figure(figsize=(6, 6))  # 図のキャンバスを正方形サイズで作る
        nx.draw_networkx_nodes(
            G, pos, node_color="skyblue", node_size=1000
        )  # ノード（＝人）を「空色」で大きめ（サイズ1000）に描画する
        nx.draw_networkx_edges(
            G, pos, width=edge_weights, edge_color=edge_colors
        )  # 辺（＝関係性）を重みに応じた太さ（edge_weights）で描画する
        nx.draw_networkx_labels(
            G, pos, font_size=12, font_family="Hiragino Sans"
        )  # 各ノードに人の名前をラベルとして表示
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=10, font_family="Hiragino Sans"
        )  # 各辺に「スコア」を表示（小数点1桁まで）

        plt.title("関係性グラフ")
        plt.axis("off")
        plt.tight_layout()
        # グラフ画像をログディレクトリに連番で保存
        image_path = os.path.join(
            config.LOG_ROOT, f"relation_graph{self.graph_count}.png"
        )
        plt.savefig(image_path)
        plt.close()
        print(f"✅ グラフ画像保存完了: {image_path}")
        self.graph_count += 1

    # === 三角形の構造とスコア平均を計算 ===
    def _compute_triangle_scores(self) -> Dict[Tuple[str, str, str], Tuple[str, float]]:
        from itertools import combinations

        def triangle_structure(a, b, c, g):
            s = {
                (a, b): g[a][b]["score"],
                (b, c): g[b][c]["score"],
                (c, a): g[c][a]["score"],
            }
            signs = [
                "+" if s[pair] >= 0 else "-" for pair in [(a, b), (b, c), (c, a)]
            ]  # スコアの符号を取得
            return "".join(signs), sum(s.values()) / 3  # 例：++-, スコアの平均

        G = self._build_graph_object()
        triangle_scores = {}
        # 三角形の組み合わせを取得
        for a, b, c in combinations(G.nodes, 3):
            if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a):
                struct, avg = triangle_structure(a, b, c, G)
                triangle_scores[(a, b, c)] = (struct, avg)
                print(
                    f"🔺 三角形: {a}, {b}, {c} → 構造: {struct}, スコア平均: {avg:.1f}"
                )
        return triangle_scores
