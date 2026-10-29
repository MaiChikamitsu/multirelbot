"""Change wording and palette here; the preview and export use the same renderer."""
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
from .model import stamp

BACKGROUND = '#101c30'
WHITE = '#ffffff'
TITLE = '動画を一時停止してください'
EVALUATE = 'インデックス{index}の関係性を評価してください。'
NEXT = '評価が終わったら、動画の再生を再開し、次のインデックス{next}の会話に進んでください。'
FINAL = 'これが最後の評価です。評価が終わったら、アンケートを終了してください。'


def font_path():
    custom = os.environ.get('EDIT_JAPANESE_FONT')
    if custom:
        if not Path(custom).is_file():
            raise ValueError('EDIT_JAPANESE_FONTのフォントが見つかりません。')
        return custom
    candidates = list(Path('/System/Library/Fonts').glob('*角*W3*'))
    candidates += [Path('/System/Library/Fonts/Hiragino Sans GB.ttc'),
                   Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise ValueError('日本語フォントがありません。EDIT_JAPANESE_FONTに日本語TTF/TTC/OTFを指定してください。')


def wording(index, last):
    return [TITLE, EVALUATE.format(index=index), FINAL if last else NEXT.format(next=index+1)]


def render(size, index, timestamp, last):
    w, h = size
    image = Image.new('RGB', size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    path = font_path()
    # Wrap by measured glyph width, then shrink the entire group to fit vertically.
    texts = [TITLE, f'インデックス{index}', EVALUATE.format(index=index).removeprefix(f'インデックス{index}'),
             FINAL if last else NEXT.format(next=index+1), f'元動画の評価時刻  {stamp(timestamp)}']
    for factor in range(100, 9, -2):
        scale = factor / 100
        blocks = []
        for text, ratio in zip(texts, [.064, .10, .046, .043, .032]):
            font = ImageFont.truetype(path, max(8, int(min(w, h*16/9)*ratio*scale)))
            lines, line = [], ''
            for ch in text:
                if line and draw.textlength(line+ch, font=font) > w*.88:
                    lines.append(line)
                    line = ''
                line += ch
            lines.append(line)
            height = font.getbbox('国')[3] + max(3, int(h*.012))
            blocks.append((font, lines, height))
        total = sum(len(lines)*height for _, lines, height in blocks)+h*.09
        if total <= h*.88:
            break
    y = (h-total)/2
    for font, lines, height in blocks:
        for line in lines:
            draw.text((w/2, y), line, font=font, fill=WHITE, anchor='mt')
            y += height
        y += h*.018
    return image
