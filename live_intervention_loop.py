"""
Live orchestration for multi-person human conversation + one robot.

This file intentionally stays small. It reuses:
- relation_estimator_from_txt.estimate_relation_once for -1.0..+1.0 relation scoring
- relation_estimator_from_txt.EMAScorer for score smoothing
- intervention_planner.InterventionPlanner for target selection and robot utterances

The expected real-time input is a stream of recognized human utterances, such as
the logs already produced by realtime_communicator.py:
{"time": datetime, "speaker": "A", "utterance": "..."}
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO, Tuple

import networkx as nx

import config
import relation_estimator_from_txt as relation_estimator
from intervention_planner import InterventionPlanner
from log_filtering import filter_logs_by_human_count


Log = Dict[str, object]
RelationScores = Dict[Tuple[str, str], float]
SpeakCallback = Callable[[str], None]


class LiveInterventionLoop:
    """
    Add human utterances one by one. Every analyze_every human utterances, estimate
    relationships and let the existing InterventionPlanner generate one robot turn.
    """

    def __init__(
        self,
        participants: Optional[List[str]] = None,
        analyze_every: Optional[int] = None,
        window_human_count: Optional[int] = None,
        relation_history_count: Optional[int] = None,
        intervention_history_count: Optional[int] = None,
        mode: str = "proposal",
        llm_model: Optional[str] = None,
        speak_callback: Optional[SpeakCallback] = None,
        output_dir: Optional[str] = None,
        debug: bool = True,
    ) -> None:
        self.cfg = config.get_config()
        self.participants = sorted(participants or [])
        self.analyze_every = (
            analyze_every
            if analyze_every is not None
            else getattr(self.cfg.realtime, "analyze_every", 5) or 5
        )
        self.window_human_count = (
            window_human_count
            if window_human_count is not None
            else getattr(self.cfg.realtime, "utterances_per_session", 10) or 10
        )
        self.relation_history_count = (
            relation_history_count
            if relation_history_count is not None
            else getattr(self.cfg.env, "max_history_relation", 6) or 6
        )
        self.intervention_history_count = (
            intervention_history_count
            if intervention_history_count is not None
            else getattr(self.cfg.env, "intervention_max_history", 9) or 9
        )
        self.mode = mode
        self.llm_model = llm_model
        self.speak_callback = speak_callback
        self.logs: List[Log] = []
        self.human_utterance_count = 0
        self.latest_scores: RelationScores = {}
        self.relation_snapshots: List[Dict[str, object]] = []
        self.past_robot_utterances: List[str] = []
        self.output_dir = Path(output_dir or config.LOG_ROOT)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        scorer_cfg = getattr(self.cfg, "scorer", None)
        self.ema_scorer = relation_estimator.EMAScorer(
            use_ema=getattr(scorer_cfg, "use_ema", True),
            gamma=getattr(scorer_cfg, "gamma", 0.8),
            max_history_sessions=getattr(scorer_cfg, "max_history_sessions", 3),
        )
        relation_estimator.DEBUG = debug

    def add_human_utterance(
        self,
        speaker: str,
        utterance: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Log]:
        """
        Add one recognized human utterance. Returns the robot log when a robot
        intervention is generated, otherwise None.
        """
        speaker = speaker.strip()
        utterance = utterance.strip()
        if not speaker or not utterance or speaker == "ロボット":
            return None

        log = {
            "time": timestamp or datetime.now(),
            "speaker": speaker,
            "utterance": utterance,
        }
        self.logs.append(log)
        self.human_utterance_count += 1
        if speaker not in self.participants:
            self.participants.append(speaker)
            self.participants.sort()

        if self.human_utterance_count % self.analyze_every != 0:
            return None

        return self.intervene_if_needed()

    def add_log(self, log: Log) -> Optional[Log]:
        """Compatibility helper for existing realtime_communicator.py log dicts."""
        return self.add_human_utterance(
            speaker=str(log.get("speaker", "")),
            utterance=str(log.get("utterance", "")),
            timestamp=log.get("time") if isinstance(log.get("time"), datetime) else None,
        )

    def intervene_if_needed(self) -> Optional[Log]:
        """Run relation scoring, planning, and robot utterance generation once."""
        participants = [p for p in self.participants if p != "ロボット"]
        if len(participants) < 2:
            print("Skip intervention: fewer than 2 human participants.")
            return None

        window_logs = filter_logs_by_human_count(
            self.logs, self.window_human_count, exclude_robot=False
        )
        scores = self._estimate_scores(window_logs, participants)
        if not scores:
            print("Skip intervention: relation scores were empty.")
            return None

        graph = self._build_graph(scores)
        triangle_scores = self._compute_triangle_scores(graph)
        intervention_logs = filter_logs_by_human_count(
            self.logs, self.intervention_history_count, exclude_robot=False
        )
        planner = InterventionPlanner(
            graph=graph,
            triangle_scores=triangle_scores,
            isolation_threshold=getattr(
                self.cfg.intervention, "isolation_threshold", 0.0
            ),
            mode=self.mode,
            num_participants=len(participants),
            past_utterances=self.past_robot_utterances,
        )
        plan = planner.plan_intervention(intervention_logs)
        if not plan:
            print("Robot intervention skipped: stable relationship state.")
            return None

        utterance = planner.generate_robot_utterance(plan, intervention_logs)
        if not utterance:
            print("Robot intervention skipped: utterance generation returned empty.")
            return None

        robot_log: Log = {
            "time": datetime.now(),
            "speaker": "ロボット",
            "utterance": utterance,
            "plan": plan,
        }
        self.logs.append(robot_log)
        print(f"[ロボット] {utterance}")
        if self.speak_callback:
            self.speak_callback(utterance)
        return robot_log

    def _estimate_scores(self, logs: List[Log], participants: List[str]) -> RelationScores:
        raw_scores = relation_estimator.estimate_relation_once(
            logs=logs,
            participants=participants,
            max_history=self.relation_history_count,
            llm_model=self.llm_model,
            _CFG=self.cfg,
        )
        utterance_counts = defaultdict(int)
        for log in self.logs:
            speaker = log.get("speaker")
            if speaker in participants:
                utterance_counts[str(speaker)] += 1

        scores: RelationScores = {}
        for pair in combinations(participants, 2):
            key = tuple(sorted(pair))
            if key in raw_scores:
                scores[key] = self.ema_scorer.update(
                    key, raw_scores[key], utterance_counts
                )
            else:
                scores[key] = self.ema_scorer.scores.get(key, 0.0)
        self.ema_scorer.finalize_round(utterance_counts)
        self.latest_scores = scores
        self.relation_snapshots.append(
            {
                "time": datetime.now(),
                "human_utterance_count": self.human_utterance_count,
                "participants": participants,
                "raw_scores": raw_scores,
                "ema_scores": scores,
            }
        )
        return scores

    def save_outputs(self) -> None:
        """Save conversation history and all relation snapshots in one run folder."""
        conversation_path = self.output_dir / "conversation.txt"
        relation_path = self.output_dir / "relation_scores.json"
        record_path = self.output_dir / "session_record.json"

        with conversation_path.open("w", encoding="utf-8") as f:
            for log in self.logs:
                time_text = _format_time(log.get("time"))
                speaker = log.get("speaker", "")
                utterance = log.get("utterance", "")
                f.write(f"[{time_text}] [{speaker}] {utterance}\n")

        relation_payload = {
            "latest_scores": self.latest_scores,
            "snapshots": self.relation_snapshots,
        }
        with relation_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(relation_payload), f, ensure_ascii=False, indent=2)

        session_payload = {
            "saved_at": datetime.now(),
            "output_dir": str(self.output_dir),
            "settings": {
                "participants": self.participants,
                "analyze_every": self.analyze_every,
                "window_human_count": self.window_human_count,
                "relation_history_count": self.relation_history_count,
                "intervention_history_count": self.intervention_history_count,
                "mode": self.mode,
                "llm_model": self.llm_model,
            },
            "conversation": self.logs,
            "relation_scores": self.relation_snapshots,
        }
        with record_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(session_payload), f, ensure_ascii=False, indent=2)

        print(f"Saved run logs to: {self.output_dir}")

    @staticmethod
    def _build_graph(scores: RelationScores) -> nx.Graph:
        graph = nx.Graph()
        for (a, b), score in scores.items():
            graph.add_edge(a, b, score=score)
        return graph

    @staticmethod
    def _compute_triangle_scores(
        graph: nx.Graph,
    ) -> Dict[Tuple[str, str, str], Tuple[str, float]]:
        triangle_scores: Dict[Tuple[str, str, str], Tuple[str, float]] = {}
        for a, b, c in combinations(graph.nodes, 3):
            if not (graph.has_edge(a, b) and graph.has_edge(b, c) and graph.has_edge(c, a)):
                continue
            edge_scores = [
                graph[a][b]["score"],
                graph[b][c]["score"],
                graph[c][a]["score"],
            ]
            struct = "".join("+" if score >= 0 else "-" for score in edge_scores)
            triangle_scores[(a, b, c)] = (struct, sum(edge_scores) / 3)
        return triangle_scores


def _iter_text_logs(lines: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        if ":" in line:
            speaker, utterance = line.split(":", 1)
        elif "]" in line and line.startswith("["):
            speaker, utterance = line[1:].split("]", 1)
        else:
            raise ValueError(
                "Input must be 'speaker: utterance' or '[speaker] utterance'."
            )
        yield speaker.strip(), utterance.strip()


def _format_time(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return "-".join(str(part) for part in value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    return value


def run_text_mode(loop: LiveInterventionLoop, input_stream: TextIO) -> None:
    """
    Run the same intervention loop with typed text instead of speech recognition.
    Input examples:
      A: そうだね
      [B] それは少し違うと思う
    """
    print("Text mode: type human utterances as 'speaker: utterance'.")
    print(f"Robot checks every {loop.analyze_every} human utterances. Type 'exit' to stop.")
    try:
        for speaker, utterance in _iter_text_logs(input_stream):
            print(f"[{speaker}] {utterance}")
            robot_log = loop.add_human_utterance(speaker, utterance)
            if robot_log:
                print()
    finally:
        loop.save_outputs()


class LiveLoopSessionAdapter:
    """SessionManager-compatible adapter used by realtime_communicator audio mode."""

    def __init__(self, loop: LiveInterventionLoop, realtime_module: Any) -> None:
        self.loop = loop
        self.realtime_module = realtime_module

    def add_utterance_count(self, log: Log) -> None:
        self._add_live_log(log)

    def add_utterance(self, log: Log) -> None:
        self._add_live_log(log)

    def finalize(self) -> None:
        self._flush_realtime_buffer()
        self.loop.save_outputs()

    def _add_live_log(self, log: Log) -> None:
        robot_log = self.loop.add_log(log)
        if not robot_log:
            return

        utterance = str(robot_log.get("utterance", ""))
        self.realtime_module.send_conversation("ロボット", utterance)
        robot_time = robot_log.get("time")
        if isinstance(robot_time, datetime):
            time_text = robot_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.realtime_module.conversation_log.append(
            f"[{time_text}] [ロボット] {utterance}"
        )

        robot_count = getattr(
            self.loop.cfg.realtime, "robot_count_after_intervention", 5
        )
        if hasattr(self.realtime_module, "set_robot_count"):
            self.realtime_module.set_robot_count(robot_count)

    def _flush_realtime_buffer(self) -> None:
        speaker = getattr(self.realtime_module, "buffer_speaker", None)
        text = getattr(self.realtime_module, "buffer_text", "")
        timestamp = getattr(self.realtime_module, "buffer_time", None)
        if not speaker or not text:
            return
        self._add_live_log(
            {
                "time": timestamp if isinstance(timestamp, datetime) else datetime.now(),
                "speaker": speaker,
                "utterance": text,
            }
        )
        self.realtime_module.buffer_speaker = None
        self.realtime_module.buffer_text = ""
        self.realtime_module.buffer_time = None


def run_audio_mode(loop: LiveInterventionLoop) -> None:
    """
    Run microphone/STT input using realtime_communicator.py, but route recognized
    utterances through LiveInterventionLoop so text/audio share the same logic.
    """
    import realtime_communicator as rc

    rc.session_manager = LiveLoopSessionAdapter(loop, rc)

    speakers = getattr(rc._CFG.participants, "speakers", None) or {}
    for speaker_name, audio_path in speakers.items():
        rc.register_reference_speaker(speaker_name, audio_path)
        print(f"話者登録: {speaker_name} <- {audio_path}")

    print("Audio mode: microphone recognition is feeding LiveInterventionLoop.")
    print(f"Robot checks every {loop.analyze_every} human utterances.")
    print(f"Run logs will be saved to: {loop.output_dir}")

    threading.Thread(target=rc.record_audio, daemon=True).start()
    if rc.USE_DIRECT_STREAM:
        threading.Thread(target=rc.streaming_stt_worker2, daemon=True).start()
    else:
        threading.Thread(target=rc.process_audio, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Audio mode stopped. Saving logs...")
        rc.session_manager.finalize()


def _pepper_callback(message: str) -> None:
    from community_analyzer import send_to_pepper_async

    send_to_pepper_async(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live relation-based robot interventions from text logs."
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Start interactive text mode instead of microphone input.",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Start microphone/STT mode and route recognized utterances here.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["text", "audio"],
        default=None,
        help="Input mode. Defaults to text unless --audio is used.",
    )
    parser.add_argument("--from-file", help="Read '[speaker] utterance' text logs.")
    parser.add_argument(
        "--participants",
        nargs="+",
        help="Optional initial participant names, e.g. --participants A B C.",
    )
    parser.add_argument("--analyze-every", type=int, help="Human turns per analysis.")
    parser.add_argument("--window-human-count", type=int, help="Recent human turns kept.")
    parser.add_argument("--relation-history-count", type=int)
    parser.add_argument("--intervention-history-count", type=int)
    parser.add_argument("--mode", default="proposal", choices=["proposal", "few_utterances", "random_target"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--send-to-pepper", action="store_true")
    parser.add_argument(
        "--output-dir",
        help="Folder for conversation.txt, relation_scores.json, and session_record.json.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    loop = LiveInterventionLoop(
        participants=args.participants,
        analyze_every=args.analyze_every,
        window_human_count=args.window_human_count,
        relation_history_count=args.relation_history_count,
        intervention_history_count=args.intervention_history_count,
        mode=args.mode,
        llm_model=args.llm_model,
        speak_callback=_pepper_callback if args.send_to_pepper else None,
        output_dir=args.output_dir,
        debug=not args.quiet,
    )

    if args.from_file:
        logs = relation_estimator.parse_conversation_file(args.from_file)
        try:
            for log in logs:
                loop.add_log(log)
        finally:
            loop.save_outputs()
        return

    import sys

    input_mode = args.input_mode or ("audio" if args.audio else "text")
    if args.text:
        input_mode = "text"

    if input_mode == "audio":
        run_audio_mode(loop)
    else:
        run_text_mode(loop, sys.stdin)


if __name__ == "__main__":
    main()
