"""Time values refer exclusively to the original video's presentation timeline."""
from dataclasses import dataclass
import math
import re
import unicodedata


def stamp(t):
    whole = int(t)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    suffix = f'{t-whole:.3f}'[1:].rstrip('0') if t != whole else ''
    return (f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}') + suffix


def parse(text):
    values = []
    for line in text.splitlines():
        line = unicodedata.normalize('NFKC', line).strip()
        if not line:
            continue
        is_start = bool(re.match(r'^会話開始\s*:', line))
        if is_start:
            line = re.sub(r'^会話開始\s*:\s*', '', line)
        else:
            line = re.sub(r'^\d+[.、)]\s*', '', line)
        if not re.fullmatch(r'\d+:\d{2}(?::\d{2})?', line):
            raise ValueError(f'時刻の形式が不正です: {line}（MM:SS または HH:MM:SS）')
        p = list(map(int, line.split(':')))
        if p[-1] >= 60 or (len(p) == 3 and p[-2] >= 60):
            raise ValueError(f'秒・時の後の分は00〜59です: {line}')
        value = p[0]*60+p[1] if len(p)==2 else p[0]*3600+p[1]*60+p[2]
        if is_start:
            continue  # Metadata only: never an evaluation or an implicit cut.
        if value in values:
            raise ValueError(f'時刻が重複しています: {line}')
        values.append(value)
    if not values:
        raise ValueError('評価タイムスタンプを入力してください。')
    return sorted(values)


@dataclass(frozen=True)
class Plan:
    times: tuple
    hold: float
    end: float
    @property
    def duration(self):
        return self.end + len(self.times)*self.hold


def plan(times, duration, hold, policy=None):
    if not math.isfinite(hold) or not 0 < hold <= 3600:
        raise ValueError('表示時間は0より大きく3600秒以下にしてください。')
    times = sorted(times)
    if not times or len(set(times)) != len(times) or times[0] < 0:
        raise ValueError('評価時刻が空、重複、または負数です。')
    if times[-1] > duration:
        raise ValueError('評価時刻が元動画の終了時刻を超えています。')
    if times[-1] != duration:
        if policy == 'cut':
            if times[-1] == 0:
                raise ValueError('0秒で動画全体をカットできません。終了時刻に変更してください。')
            duration = times[-1]
        elif policy == 'end':
            times[-1] = duration
        else:
            raise ValueError('最後の評価時刻と動画終了時刻が違います。処理方法を選択してください。')
    return Plan(tuple(times), hold, duration)
