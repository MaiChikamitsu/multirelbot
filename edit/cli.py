"""Interactive terminal interface. Compatible with Python 3.9; no Tk required."""
from pathlib import Path
import shlex
import threading
from editor.model import parse, plan, stamp
from editor.engine import binary, probe, export, Cancelled


def input_path(text):
    raw = text.strip()
    # Finder drag-and-drop escapes spaces; also accept ordinary pasted paths.
    if Path(raw).expanduser().exists():
        return Path(raw).expanduser()
    parts = shlex.split(raw)
    if len(parts) != 1:
        raise ValueError('動画ファイルを1つ指定してください。')
    return Path(parts[0]).expanduser()


def main():
    print('研究動画・評価案内エディター（ターミナル版）')
    print('動画は外部へ送信されません。中止する場合は Control+C。\n')
    binary('ffmpeg'); binary('ffprobe')
    source = input_path(input('動画をここにドラッグ＆ドロップしてEnter: '))
    if not source.is_file():
        raise ValueError('動画ファイルが見つかりません。')
    info = probe(source)
    print(f'元動画: {stamp(info["duration"])} / {info["size"][0]}×{info["size"][1]}')
    print('評価時刻の一覧をまとめて貼り付けてください。番号付き・空行・0:49形式も使えます。')
    print('「会話開始」の行は評価に含めません。元動画の冒頭は残します。')
    print('貼り付けた後、最後の行でEnterを押し、「完了」と入力してEnterで確定してください。')
    lines = []
    while True:
        line = input('時刻: ').strip()
        if line.lower() in ('完了', 'end'):
            break
        if line:
            lines.append(line)
    times = parse('\n'.join(lines))
    print(f'評価時刻を{len(times)}件読み込みました（{stamp(times[0])}〜{stamp(times[-1])}）。')
    hold = float(input('案内の表示時間（秒）[10]: ').strip() or '10')
    policy = None
    if times[-1] < info['duration']:
        print(f'最後の評価 {stamp(times[-1])} と元動画終了 {stamp(info["duration"])} が違います。')
        print('1: 最後の評価以降の会話をカット / 2: 最後の評価を元動画終了時刻に変更')
        while True:
            choice = input('1 または 2（未入力では進みません）: ').strip()
            if choice in ('1', '2'):
                policy = 'cut' if choice == '1' else 'end'
                break
    result = plan(times, info['duration'], hold, policy)
    default = source.with_name(source.stem+'_評価案内.mp4')
    serial = 2
    while default.exists() or default.with_suffix('.log').exists():
        default = source.with_name(source.stem+f'_評価案内_{serial}.mp4')
        serial += 1
    raw = input(f'出力先 [Enterで {default}]: ').strip()
    target = input_path(raw) if raw else default
    if not target.parent.is_dir() or target.suffix.lower() != '.mp4':
        raise ValueError('既存フォルダ内の.mp4ファイル名を指定してください。')
    print('\n作成内容:')
    for index, t in enumerate(result.times, 1):
        print(f'  インデックス{index}: 元動画 {stamp(t)}（完成動画 {stamp(t+(index-1)*hold)}）')
    print(f'完成予想: {stamp(result.duration)} / 保存先: {target}')
    print('フレーム境界への丸めによる微小な時間差があります。')
    if input('この内容で作成しますか？ [Enterで作成 / nで中止]: ').strip().lower() not in ('', 'y', 'yes'):
        print('中止しました。'); return 0
    stop = threading.Event()
    errors = []
    last = [-1]
    def progress(value, text):
        percent = int(value*100)
        if percent > last[0]:
            print(f'\r{percent:3d}% {text}', end='', flush=True)
            last[0] = percent
    def worker():
        try:
            export(source, target, result, info, progress, stop)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(.2)
    except KeyboardInterrupt:
        print('\n中止処理中です。一時ファイルを片付けています。')
        stop.set()
        # Further Ctrl+C must not interrupt cleanup.
        while thread.is_alive():
            try:
                thread.join(.2)
            except KeyboardInterrupt:
                pass
    if errors:
        raise errors[0]
    print(f'\n完了しました: {target}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError, Cancelled):
        print('\n中止しました。')
        raise SystemExit(130)
    except Exception as exc:
        print(f'\n作成できませんでした: {exc}')
        raise SystemExit(1)
