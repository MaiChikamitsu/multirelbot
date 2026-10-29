"""Run this file or double-click 起動.command."""
from pathlib import Path
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk
from editor.cards import render, wording
from editor.engine import probe, export, Cancelled
from editor.model import parse, plan, stamp


class App:
    def __init__(self, root):
        self.root = root
        root.title('研究動画・評価案内エディター')
        root.geometry('920x860')
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.source = tk.StringVar()
        self.folder = tk.StringVar()
        self.name = tk.StringVar(value='評価案内付き.mp4')
        self.hold = tk.StringVar(value='10')
        self.status = tk.StringVar(value='動画を選び、元動画の評価時刻を1行ずつ入力してください。')
        frame = ttk.Frame(root, padding=18); frame.pack(fill='both', expand=True)
        frame.columnconfigure(1, weight=1)
        self.controls = []
        for row, title, variable, callback in [(0,'動画ファイル',self.source,self.choose), (4,'出力フォルダ',self.folder,self.choose_folder)]:
            ttk.Label(frame,text=title).grid(row=row,column=0,sticky='w',pady=5)
            entry = ttk.Entry(frame,textvariable=variable); entry.grid(row=row,column=1,sticky='ew')
            button = ttk.Button(frame,text='選択…',command=callback); button.grid(row=row,column=2)
            self.controls += [entry,button]
        ttk.Label(frame,text='評価時刻\nMM:SS / HH:MM:SS').grid(row=1,column=0,sticky='nw')
        self.times = tk.Text(frame,height=5,width=30); self.times.grid(row=1,column=1,columnspan=2,sticky='ew')
        ttk.Label(frame,text='表示時間（秒）').grid(row=2,column=0,sticky='w',pady=6)
        entry = ttk.Entry(frame,textvariable=self.hold); entry.grid(row=2,column=1,sticky='w'); self.controls.append(entry)
        ttk.Label(frame,text='出力ファイル名').grid(row=5,column=0,sticky='w')
        entry = ttk.Entry(frame,textvariable=self.name); entry.grid(row=5,column=1,columnspan=2,sticky='ew'); self.controls.append(entry)
        actions = ttk.Frame(frame); actions.grid(row=6,column=0,columnspan=3,pady=10)
        for title, command in [('プレビュー', self.preview),('動画を作成',self.start)]:
            button = ttk.Button(actions,text=title,command=command); button.pack(side='left',padx=5); self.controls.append(button)
        self.stop_button = ttk.Button(actions,text='中止',command=self.cancel.set,state='disabled'); self.stop_button.pack(side='left')
        ttk.Button(actions,text='出力フォルダを開く',command=self.open_folder).pack(side='left',padx=5)
        self.progress = ttk.Progressbar(frame,maximum=100); self.progress.grid(row=7,column=0,columnspan=3,sticky='ew')
        ttk.Label(frame,textvariable=self.status,wraplength=850).grid(row=8,column=0,columnspan=3,sticky='w',pady=5)
        self.details = tk.Text(frame,height=5,wrap='word',state='disabled'); self.details.grid(row=9,column=0,columnspan=3,sticky='ew')
        self.card_choice = ttk.Combobox(frame,state='readonly'); self.card_choice.grid(row=10,column=0,columnspan=3,sticky='ew')
        self.card_choice.bind('<<ComboboxSelected>>',lambda event:self.show_card())
        self.image_label = ttk.Label(frame); self.image_label.grid(row=11,column=0,columnspan=3,pady=5)
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(100,self.poll)

    def choose(self):
        path = filedialog.askopenfilename(filetypes=[('動画','*.mp4 *.mov *.m4v *.mkv *.avi *.webm'),('すべて','*')])
        if path:
            self.source.set(path); self.folder.set(str(Path(path).parent)); self.name.set(Path(path).stem+'_評価案内.mp4')

    def choose_folder(self):
        path = filedialog.askdirectory()
        if path: self.folder.set(path)

    def open_folder(self):
        if Path(self.folder.get()).is_dir(): subprocess.Popen(['open',self.folder.get()])

    def prepare(self):
        source = Path(self.source.get()).expanduser()
        if not source.is_file(): raise ValueError('動画ファイルを選択してください。')
        info = probe(source)
        times = parse(self.times.get('1.0','end'))
        hold = float(self.hold.get())
        # Validate all ordinary errors before asking the end-policy question.
        policy = None
        if times[-1] < info['duration']:
            choice = messagebox.askyesnocancel('最後の評価時刻の確認',
                f'最後の評価: {stamp(times[-1])}\n元動画終了: {stamp(info["duration"])}\n\n'
                '「はい」: 最後の評価以降の会話をカット\n'
                '「いいえ」: 最後の評価を元動画の終了時刻に変更\n「キャンセル」: 入力に戻る')
            if choice is None: return None
            policy = 'cut' if choice else 'end'
        result = plan(times,info['duration'],hold,policy)
        return source,info,result

    def display(self, info, result):
        self.current = result
        self.card_choice['values'] = [f'インデックス{i} / {stamp(t)}' for i,t in enumerate(result.times,1)]
        self.card_choice.current(0); self.show_card()
        text = f'元動画: {stamp(info["duration"])} / {info["size"][0]}×{info["size"][1]} / {float(info["fps"]):.3f} fps\n予想完成時間: {stamp(result.duration)}（フレーム境界に丸めます）\n'
        for i,t in enumerate(result.times,1):
            text += f'\nインデックス{i}: 元 {stamp(t)} → 完成 {stamp(t+(i-1)*result.hold)}\n'+'\n'.join(wording(i,i==len(result.times)))+'\n'
        self.details.config(state='normal'); self.details.delete('1.0','end'); self.details.insert('1.0',text); self.details.config(state='disabled')

    def show_card(self):
        i = self.card_choice.current()
        self.photo = ImageTk.PhotoImage(render((560,315),i+1,self.current.times[i],i==len(self.current.times)-1))
        self.image_label.config(image=self.photo)

    def preview(self):
        try:
            prepared = self.prepare()
            if prepared: self.display(prepared[1],prepared[2])
        except Exception as exc: messagebox.showerror('入力を確認してください',str(exc))

    def start(self):
        try:
            prepared = self.prepare()
            if not prepared: return
            source,info,result = prepared
            name = self.name.get().strip()
            if Path(name).name != name or not name.lower().endswith('.mp4'): raise ValueError('ファイル名だけを.mp4形式で入力してください。')
            target = Path(self.folder.get())/name
            if not self.folder.get() or not target.parent.is_dir(): raise ValueError('出力フォルダを指定してください。')
            if target.exists() or target.with_suffix('.log').exists(): raise ValueError('同名の動画またはログがあります。出力名を変更してください。')
            self.display(info,result)
            self.busy = True; self.cancel.clear(); self.progress['value']=0
            for widget in self.controls: widget.config(state='disabled')
            self.times.config(state='disabled'); self.stop_button.config(state='normal')
            def worker():
                try:
                    export(source,target,result,info,lambda v,t:self.events.put(('progress',v,t)),self.cancel)
                    self.events.put(('done',str(target)))
                except Exception as exc: self.events.put(('error',str(exc)))
            threading.Thread(target=worker,daemon=True).start()
        except Exception as exc: messagebox.showerror('作成できません',str(exc))

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0]=='progress':
                    self.progress['value']=event[1]*100; self.status.set(event[2])
                else:
                    self.busy=False
                    for widget in self.controls: widget.config(state='normal')
                    self.times.config(state='normal'); self.stop_button.config(state='disabled')
                    self.status.set(('完了: ' if event[0]=='done' else '中止・エラー: ')+event[1])
                    if event[0]=='done': messagebox.showinfo('完了',event[1])
                    else: messagebox.showerror('処理結果',event[1])
        except queue.Empty: pass
        self.root.after(100,self.poll)

    def close(self):
        if self.busy:
            messagebox.showinfo('処理中','「中止」を押して処理が止まってから閉じてください。')
        else: self.root.destroy()


if __name__=='__main__':
    root = tk.Tk(); App(root); root.mainloop()
