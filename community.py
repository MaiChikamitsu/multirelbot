from collections import defaultdict, deque
from itertools import combinations
from datetime import datetime
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
import csv
import matplotlib.pyplot as plt
import networkx as nx
import sys

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

plt.rcParams["font.family"] = "Hiragino Sans"


# === ログを構造化 ===
def parse_log(log_text):
    pattern = r"\[(.*?)\] \[(.*?)\] (.+)"
    logs = []
    for line in log_text.strip().split("\n"):
        m = re.match(pattern, line)
        if m:
            timestamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            speaker = m.group(2)
            utterance = m.group(3)
            logs.append({"time": timestamp, "speaker": speaker, "utterance": utterance})
    return logs


# === GPTで話題の一致を判定 ===
def is_same_topic(history_utterances, current_utterance):
    history_text = "\n".join([f"- {u}" for u in history_utterances])
    prompt = f"""以下は、ある会話のログの一部です。
この流れの中で、最後の発話が【同じ話題の続きかどうか】をYes/Noで判定してください。

【これまでの会話】
{history_text}

【最後の発話】
- {current_utterance}

この発話は、上の会話と同じ話題の続きですか？ Yes か No で答えてください。
"""
    res = client.chat.completions.create(
        # model="gpt-4o",
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "あなたは会話分析の専門家です。話題の変化に敏感です。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return "yes" in res.choices[0].message.content.strip().lower()


# === セッションの分割 ===
def split_sessions(logs):
    sessions = []

    if sys.argv[1] == "1":  # 5, 10, 15, 20, 25, 30
        ends = [5, 10, 15, 20, 25, 30]
    elif sys.argv[1] == "2":  # 5, 10, 15, 21, 26, 32
        ends = [5, 10, 15, 21, 26, 32]
    elif sys.argv[1] == "3":  # 5, 10, 16, 22, 29, 37
        ends = [5, 10, 16, 22, 29, 37]

    for i, end in enumerate(ends):
        # 最初の2セッションは、その終端の発話数をそのままウィンドウサイズに
        if i < 2:
            window = end
        else:
            # 2つ前の終端との差分をウィンドウサイズに
            window = end - ends[i - 2]
        start = max(0, end - window)
        sessions.append(logs[start:end])
        # 配列を改行で結合してprint
        # print(f"\n--- セッション {i + 1} ({start}〜{end}) ---")
        # for log in logs[start:end]:
        #     print(f"{log['utterance']}")

    return sessions


# === GPTに仲の良さを尋ねるプロンプト ===
def get_gpt_friendship_scores(session_logs, participants):
    conversation = "\n".join(
        [f"[{log['speaker']}] {log['utterance']}" for log in session_logs]
    )

    pair = [f"- {a} × {b}" for a, b in combinations(participants, 2)]
    pair_lines = "\n".join(pair)
    output_format = "\n".join([f"{a}-{b}:" for a, b in combinations(participants, 2)])
    #     prompt = f"""
    # 以下の会話を読み、参加者それぞれの「仲の良さ（親密度）」を -1.0 〜 1.0 の間の実数（小数第1位まで）で評価してください。
    # -1.0 は明確な対立、0 は中立、+1.0 は非常に親しい関係を表します。
    # 評価対象は以下のペアです：

    # {pair_lines}

    # 会話：
    # {conversation}

    # 出力形式：
    # {output_format}
    # """
    prompt = f"""
以下の会話を読み、参加者それぞれの「仲の良さ（親密度）」を -1.0 〜 +1.0 の間の**実数（小数第1位まで）**で評価してください。
0.0 は特に親しさも対立も感じない「中立的な状態」です。
そこから -1.0（強い対立） 〜 +1.0（非常に親しい） に向けて、どれくらい離れているかを評価してください。

具体例：
-1.0 例：皮肉、批判、無視、相手を無視して話を進める
+1.0 例：共感、褒める、相手に話題を振る、一緒に行動する

評価対象は以下のペアです：
{pair_lines}

会話：
{conversation}

出力形式：
{output_format}
"""
    res = client.chat.completions.create(
        # model="gpt-4o",
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return parse_scores_from_response(res.choices[0].message.content)


# === GPTの返答を辞書に変換 ===
def parse_scores_from_response(response_text):
    scores = {}
    lines = response_text.strip().split("\n")
    for line in lines:
        if ":" in line:
            pair, score = line.split(":")
            a, b = [x.strip() for x in pair.split("-")]
            # 数値部分だけを抽出（-1.0 ～ 1.0 を想定）
            match = re.search(r"-?\d+(\.\d+)?", score)
            if match:
                scores[(a, b)] = float(match.group())
            else:
                print(f"⚠️ 数値抽出失敗: {line.strip()}")
    return scores


# === セッションごとにスコア更新（時間減衰付きEMA） ===


def compute_all_relationship_scores(logs, gamma=0.9, max_history_sessions=3):
    sessions = split_sessions(logs)
    # print(f"\n📎 総セッション数: {len(sessions)}")

    relationship_scores = defaultdict(float)
    interaction_history = defaultdict(
        lambda: deque(maxlen=max_history_sessions)
    )  # 各ペアの過去発話数

    for idx, session in enumerate(sessions, 1):
        print(f"\n--- セッション {idx} ---")
        # for log in session:
        #     print(f"[{log['time'].strftime('%H:%M:%S')}] [{log['speaker']}] {log['utterance']}")

        participants = list(set(log["speaker"] for log in session))
        # print(f"👥 参加者: {participants}")

        if len(participants) < 2:
            print("⚠️ 参加者が1人以下のためスキップ")
            continue

        # 各話者のそのセッションでの発話回数をカウント
        session_utterance_counts = defaultdict(int)
        for log in session:
            session_utterance_counts[log["speaker"]] += 1
        # print(f"🗣️ 発話数: {dict(session_utterance_counts)}")

        # GPTスコアを取得
        gpt_scores = get_gpt_friendship_scores(session, participants)
        # print(f"📐 セッション {idx} の GPTスコア:")
        # for (a, b), s in gpt_scores.items():
        #     print(f"{a} - {b}: {s:.2f}")

        for (a, b), score in gpt_scores.items():
            key = tuple(sorted([a, b]))
            # セッション内での発話回数（ペアのうち少ない方）
            session_utterance = min(
                session_utterance_counts[a], session_utterance_counts[b]
            )

            # 過去の発話数を時間減衰を考慮して合計
            past_utterances = list(interaction_history[key])
            weighted_past = 0
            for i, count in enumerate(past_utterances):
                sessions_ago = len(past_utterances) - i
                weight = gamma ** sessions_ago
                weighted_past += count * weight

            x_t = score  # GPTスコアそのまま使う

            if key not in relationship_scores:
                relationship_scores[key] = x_t  # 初回は代入
                # print(f"🆕 初期スコア: {key} = {x_t:.2f}")
            else:
                # 今回の発話数と時間減衰させた過去の発話数の比率
                weighted_total = weighted_past + session_utterance
                alpha = min(1.0, session_utterance / weighted_total) if weighted_total > 0 else 1.0
                # print(f"🔢 α計算 ({a}-{b}): min発話数={session_utterance}, 減衰過去合計={weighted_past:.2f}, α={alpha:.2f}")
                prev = relationship_scores[key]
                updated = alpha * x_t + (1 - alpha) * prev
                relationship_scores[key] = updated
                # print(f"🔁 EMA更新: {key} = {alpha:.2f}×{x_t:.2f} + {(1-alpha):.2f}×{prev:.2f} → {updated:.2f}")
            # 履歴に追加（発話数のみ）
            interaction_history[key].append(session_utterance)

        # print(f"📊 セッション {idx} 終了時の関係スコア（累積）:")
        for (a, b), score in relationship_scores.items():
            print(f"{a} - {b}: {score:.1f}")

    return relationship_scores


def compute_unified_scores_per_session(logs):
    """
    セッションごとに、そこまでの累積会話履歴をGPTに与えて関係性を推定。
    各時点でのスコア辞書をリストとして返す。
    """
    sessions = split_sessions(logs)
    # print(f"\n📎 総セッション数: {len(sessions)}")

    all_session_scores = []
    cumulative_logs = []
    for idx, session in enumerate(sessions, 1):
        cumulative_logs.extend(session)
        print(f"\n--- セッション {idx}（累積履歴でGPT推定） ---")
        for log in cumulative_logs:
            print(
                f"[{log['time'].strftime('%H:%M:%S')}] [{log['speaker']}] {log['utterance']}"
            )

        participants = set(log["speaker"] for log in cumulative_logs)

        gpt_scores = get_gpt_friendship_scores(cumulative_logs, participants)
        print(f"📐 セッション {idx} の GPTスコア:")

        for (a, b), s in gpt_scores.items():
            print(f"{a} - {b}: {s:.2f}")

        all_session_scores.append(gpt_scores)

    return all_session_scores


# === メイン処理 ===
if __name__ == "__main__":
    USE_SESSION_BASED = True  # ← False にすると一括推定に切り替わる

    with open(f"logs/conversation{sys.argv[1]}.txt", "r", encoding="utf-8") as f:
        log_text = f.read()

    logs = parse_log(log_text)

    if USE_SESSION_BASED:
        scores = compute_all_relationship_scores(logs)
    else:
        session_scores_list = compute_unified_scores_per_session(logs)
        # 最終スコアだけCSV & グラフに使う
        scores = session_scores_list[-1] if session_scores_list else {}

    # === なければディレクトリ作成 ===
    os.makedirs("output", exist_ok=True)

    # # === ① CSVに保存 ===
    # with open("output/relationship_scores.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Person1", "Person2", "Score"])
    #     for (a, b), score in scores.items():
    #         a, b = sorted([a, b])
    #         writer.writerow([a, b, round(score, 2)])
    # # print("✅ CSV保存完了: output/relationship_scores.csv")

    # # === ② NetworkXでグラフ描画 ===
    # G = nx.Graph()
    # for (a, b), score in scores.items():
    #     a, b = sorted([a, b])
    #     G.add_edge(a, b, score=score)

    # # 1. まず計算だけする
    # preliminary_weights = {}
    # for u, v in G.edges():
    #     score = G[u][v]['score']
    #     scaled = (score + 1.0) / 2.0  # 0.0～1.0に変換
    #     preliminary_weights[(u, v)] = scaled ** 6 * 200

    # # 2. 最大weightを取得
    # max_weight = max(preliminary_weights.values())
    # min_weight = max_weight / 7
    # # print(f"🔎 最大weight: {max_weight:.2f}, 最小weight設定: {min_weight:.2f}")

    # # 3. 最小値を考慮してweightをセット
    # for (u, v), prelim in preliminary_weights.items():
    #     adjusted_weight = max(min_weight, prelim)
    #     G[u][v]['weight'] = adjusted_weight
    #     # print(f"スコア調整: {u} - {v}: prelim={prelim:.2f} → weight={adjusted_weight:.2f}")

    # # 配置計算（スコアが強いほど引き合う）
    # pos = nx.spring_layout(G, weight='weight', seed=42, iterations=500)

    # # 線の太さ（親密度の強さに応じて）
    # edge_weights = [max(0.5, 5 * abs(G[u][v]['score'])) for u, v in G.edges()]

    # # スコアラベル（±1.0）
    # edge_labels = {(u, v): f"{G[u][v]['score']:.1f}" for u, v in G.edges()}

    # # 敵対関係は赤、親密な関係は青。0以上は青、0未満は赤
    # edge_colors = ['red' if G[u][v]['score'] < 0 else 'skyblue' for u, v in G.edges()]

    # plt.figure(figsize=(6, 6))  # 図のキャンバスを正方形サイズで作る
    # nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1000)  # ノード（＝人）を「空色」で大きめ（サイズ1000）に描画する
    # nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors)  # 辺（＝関係性）を重みに応じた太さ（edge_weights）で描画する
    # nx.draw_networkx_labels(G, pos, font_size=12, font_family="Hiragino Sans")  # 各ノードに人の名前をラベルとして表示
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_family="Hiragino Sans")  # 各辺に「スコア」を表示（小数点1桁まで）

    # plt.title("関係性グラフ")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig("output/relationship_graph.png")
    # plt.close()
    # print("✅ グラフ画像保存完了: output/relationship_graph.png")
