import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from editor.model import parse, plan
from editor.cards import render, wording
from editor.engine import binary, probe, export, Cancelled


class ModelTests(unittest.TestCase):
    def test_times(self):
        self.assertEqual(parse('07:15\n00:00\n03:30\n01:00:00'),[0,210,435,3600])
        for text in ['', '3:3', '00:60', '01:60:00', '03:30\n00:03:30', '-01:00']:
            with self.assertRaises(ValueError): parse(text)

    def test_numbered_bulk_paste(self):
        sample = Path(__file__).resolve().parents[1]/'examples/評価時刻サンプル.txt'
        values = parse(sample.read_text())
        self.assertEqual(len(values),41)
        self.assertEqual(values[:3],[49,73,119])
        self.assertEqual(values[-1],889)
        self.assertNotIn(29,values)
        self.assertEqual(parse('1. 0:49\n2. 1:13'),[49,73])
        self.assertEqual(parse('会話開始：0:29\n１．０：４９'),[49])
        for text in ['1. 0:60', '会話開始：0:99\n1. 0:49', '1. 0:49\n2. 00:49', '1. 0:49 trailing']:
            with self.assertRaises(ValueError): parse(text)

    def test_plan(self):
        p = plan([900,210,435],900,10)
        self.assertEqual(p.duration,930)
        with self.assertRaises(ValueError): plan([3],4,10)
        self.assertEqual(plan([1,3],4,10,'end').times,(1,4))
        self.assertEqual(plan([1,3],4,10,'cut').end,3)
        for times, hold in [([5],10),([0,0],10),([4],float('nan')),([4],0)]:
            with self.assertRaises(ValueError): plan(times,4,hold)

    def test_cards(self):
        self.assertIn('インデックス3',wording(2,False)[2])
        self.assertIn('最後',wording(3,True)[2])
        self.assertNotIn('インデックス4',wording(3,True)[2])
        for size in [(640,360),(360,640),(1920,1080)]:
            image = render(size,123,210,False)
            self.assertEqual(image.size,size)
            self.assertGreater(len(image.getcolors(size[0]*size[1])),20)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: cls.ff = binary('ffmpeg'); binary('ffprobe')
        except ValueError as exc: raise unittest.SkipTest(str(exc))

    def runff(self,*args):
        return subprocess.check_output([self.ff,'-hide_banner','-loglevel','error','-y',*map(str,args)])

    def samples(self,path,start,duration):
        data = self.runff('-ss',start,'-i',path,'-t',duration,'-vn','-ac','1','-ar','48000','-f','f32le','pipe:1')
        return struct.unpack('<'+'f'*(len(data)//4),data)

    def pixel(self,path,t):
        return tuple(self.runff('-ss',t,'-i',path,'-frames:v','1','-vf','scale=1:1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); source=d/'日本語 動画.mp4'; out=d/'完成.mp4'
            self.runff('-f','lavfi','-i','color=c=red:s=640x360:r=30:d=2',
                       '-f','lavfi','-i','color=c=blue:s=640x360:r=30:d=2',
                       '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=4',
                       '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[v]',
                       '-map','[v]','-map','2:a','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',source)
            before=hashlib.sha256(source.read_bytes()).digest()
            p=plan(parse('00:04\n00:02\n00:00'),4,1)
            export(source,out,p,probe(source))
            result=probe(out)
            self.assertAlmostEqual(result['duration'],7,delta=.08)
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).digest())
            streams=json.loads(subprocess.check_output([binary('ffprobe'),'-v','error','-show_streams','-of','json',str(out)]))['streams']
            self.assertEqual([s['codec_name'] for s in streams],['h264','aac'])
            self.assertAlmostEqual(float(streams[0]['duration']),float(streams[1]['duration']),delta=.08)
            # Exact card/conversation transitions including a card at zero and EOF.
            for t in [.1, .95, 3.05, 3.95, 6.05, 6.9]:
                pixel=self.pixel(out,t)
                self.assertLess(max(pixel),85,(t,pixel))
            for t in [1.05,2.95]:
                r,g,b=self.pixel(out,t); self.assertGreater(r,150); self.assertLess(b,60)
            for t in [4.05,5.95]:
                r,g,b=self.pixel(out,t); self.assertGreater(b,150); self.assertLess(r,60)
            for start in [.2,3.2,6.2]:
                samples=self.samples(out,start,.5)
                self.assertLess(max(map(abs,samples)),.002)
            for start in [1.1,2.8,4.1,5.8]:
                samples=self.samples(out,start,.1)
                self.assertGreater(math.sqrt(sum(x*x for x in samples)/len(samples)),.05)
            # A/V alignment: the conversation audio keeps the same phase after insertion.
            original=self.samples(source,.25,.1); edited=self.samples(out,1.25,.1)
            corr=sum(a*b for a,b in zip(original,edited))/math.sqrt(sum(a*a for a in original)*sum(b*b for b in edited))
            self.assertGreater(corr,.95)
            self.runff('-i',out,'-f','null','-')
            with self.assertRaises(ValueError): export(source,source,p,probe(source))
            with self.assertRaises(ValueError): export(source,out,p,probe(source))

    def test_silent_fractional_rotated(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); base=d/'base.mp4'; source=d/'rotated.mov'; out=d/'out.mp4'
            self.runff('-f','lavfi','-i','testsrc2=s=640x360:r=30000/1001:d=2','-c:v','libx264',base)
            self.runff('-display_rotation','90','-i',base,'-c','copy',source)
            info=probe(source); self.assertEqual(info['size'],(360,640))
            p=plan([0,1],info['duration'],.5,'end')
            export(source,out,p,info)
            self.assertEqual(probe(out)['size'],(360,640))
            self.assertAlmostEqual(probe(out)['duration'],p.duration,delta=.1)
            self.assertLess(max(map(abs,self.samples(out,.2,.5))),.002)

    def test_cancel_and_cleanup(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); source=d/'input.mp4'; target=d/'out.mp4'
            self.runff('-f','lavfi','-i','color=s=320x180:d=1','-c:v','libx264',source)
            stop=threading.Event(); stop.set()
            with self.assertRaises(Cancelled): export(source,target,plan([1],1,1),probe(source),cancel=stop)
            self.assertFalse(target.exists())
            self.assertFalse(list(d.glob('.research-edit-*')))


if __name__=='__main__': unittest.main()
