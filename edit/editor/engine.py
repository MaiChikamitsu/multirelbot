"""Sequential H.264 + PCM intermediate segments; AAC is encoded only once."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from fractions import Fraction


class Cancelled(Exception):
    pass


def binary(name):
    override = os.environ.get('EDIT_'+name.upper())
    local = Path(__file__).resolve().parents[1] / 'bin' / name
    candidates = [override, str(local), shutil.which(name), f'/opt/homebrew/bin/{name}', f'/usr/local/bin/{name}']
    for p in candidates:
        if p and Path(p).is_file():
            return str(p)
    raise ValueError(f'{name}が見つかりません。READMEの初回セットアップでFFmpegを導入してください。')


def probe(path):
    cmd = [binary('ffprobe'), '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise ValueError('動画を読み込めません: '+p.stderr[-2000:])
    data = json.loads(p.stdout)
    video = next((s for s in data['streams'] if s['codec_type']=='video' and not s.get('disposition', {}).get('attached_pic')), None)
    if not video:
        raise ValueError('動画ストリームがありません。')
    duration = float(video.get('duration') or data['format']['duration'])
    fps = Fraction(video.get('avg_frame_rate') or '0/1')
    if fps <= 0:
        fps = Fraction(video.get('r_frame_rate', '30/1'))
    if fps <= 0 or fps > 240 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('動画の長さ・フレームレートを解釈できません。')
    w, h = int(video['width']), int(video['height'])
    sar = video.get('sample_aspect_ratio', '1:1').replace(':', '/')
    try:
        w = round(w*float(Fraction(sar)))
    except (ValueError, ZeroDivisionError):
        pass
    angle = float(video.get('tags', {}).get('rotate', 0))
    for side in video.get('side_data_list', []):
        angle = float(side.get('rotation', angle))
    if round(angle) % 180 == 90:
        w, h = h, w
    w, h = max(2, round(w/2)*2), max(2, round(h/2)*2)
    return dict(duration=duration, fps=fps, size=(w,h), video_index=video['index'],
                audio=any(s['codec_type']=='audio' for s in data['streams']))


def export(source, target, plan, info, progress=lambda value, text: None, cancel=None):
    from .cards import render
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target.exists():
        raise ValueError('元動画や既存の出力ファイルは上書きできません。別の名前にしてください。')
    if target.suffix.lower() != '.mp4' or not target.parent.is_dir():
        raise ValueError('出力先フォルダと.mp4ファイル名を確認してください。')
    ffmpeg = binary('ffmpeg')
    logpath = target.with_suffix('.log')
    # Exclusive log creation also prevents concurrent exports to the same target.
    logfile = logpath.open('x', encoding='utf-8')
    fps = info['fps']
    rate = str(fps)
    w,h = info['size']
    stop = cancel or threading.Event()
    def run(args, duration, base, span):
        cmd = [ffmpeg, '-hide_banner', '-nostdin', '-y', '-progress', 'pipe:1', '-nostats']+args
        logfile.write(json.dumps(cmd, ensure_ascii=False)+'\n'); logfile.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=logfile, text=True)
        def monitor():
            while proc.poll() is None:
                if stop.wait(.2):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        try:
            for line in proc.stdout:
                if line.startswith('out_time_us='):
                    try:
                        fraction = min(1, max(0, int(line.split('=')[1])/1e6/max(duration,.001)))
                        progress(base+span*fraction, '動画を処理しています')
                    except ValueError:
                        pass
            code = proc.wait()
            watcher.join()
            if stop.is_set():
                raise Cancelled('処理を中止しました。')
            if code:
                raise RuntimeError(f'FFmpegの処理に失敗しました。ログ: {logpath}')
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
            proc.stdout.close()
    try:
        with tempfile.TemporaryDirectory(prefix='.research-edit-', dir=target.parent) as temp:
            tmp = Path(temp)
            segments = []
            frame = lambda t: int(math.floor(float(fps)*t+.5))
            hold_frames = max(1, frame(plan.hold))
            previous = 0
            total = len(plan.times)*2+1
            for index, timestamp in enumerate(plan.times, 1):
                boundary = frame(timestamp)
                frames = boundary-previous
                if frames:
                    duration = float(Fraction(frames, 1)/fps)
                    start = float(Fraction(previous, 1)/fps)
                    args = ['-ss', str(start), '-i', str(source)]
                    if not info['audio']:
                        args += ['-f','lavfi','-i','anullsrc=r=48000:cl=stereo']
                    args += ['-map', f'0:{info["video_index"]}', '-map', '0:a:0' if info['audio'] else '1:a:0',
                             '-vf', f'scale={w}:{h},setsar=1,fps={rate}',
                             '-af', f'aresample=48000:first_pts=0,aformat=channel_layouts=stereo,apad,atrim=duration={duration}',
                             '-t', str(duration)]
                    segment = tmp/f'{len(segments):05d}.mkv'
                    run(args+['-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                              '-c:a','pcm_s16le',str(segment)], duration, (index-1)*2/total, 1/total)
                    segments.append(segment)
                duration = float(Fraction(hold_frames, 1)/fps)
                image = tmp/'card.png'
                render((w,h), index, timestamp, index==len(plan.times)).save(image)
                segment = tmp/f'{len(segments):05d}.mkv'
                run(['-loop','1','-framerate',rate,'-i',str(image),'-f','lavfi','-i','anullsrc=r=48000:cl=stereo',
                     '-t',str(duration),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                     '-c:a','pcm_s16le',str(segment)], duration, ((index-1)*2+1)/total,1/total)
                segments.append(segment)
                previous = boundary
            listing = tmp/'segments.txt'
            listing.write_text(''.join(f"file '{s.name}'\n" for s in segments))
            final = tmp/'result.mp4'
            run(['-f','concat','-safe','1','-i',str(listing),'-map','0:v:0','-map','0:a:0',
                 '-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(final)],
                plan.duration, (total-1)/total, 1/total)
            check = probe(final)
            tolerance = (len(plan.times)+2)/float(fps)+.1
            if abs(check['duration']-plan.duration) > tolerance:
                raise RuntimeError('出力時間の検証に失敗しました。ログを確認してください。')
            # Hard link publishes atomically without overwriting a concurrent file.
            os.link(final, target)
            progress(1, '完了しました')
    except Exception as exc:
        logfile.write(f'\nERROR: {exc}\n')
        raise
    finally:
        logfile.close()
    return target
