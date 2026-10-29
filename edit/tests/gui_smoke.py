import sys, time, tempfile, os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from gui import App
from editor.engine import binary
import subprocess
with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder).resolve(); source=folder/'input.mp4'
    subprocess.run([binary('ffmpeg'),'-v','error','-f','lavfi','-i','testsrc2=s=640x360:r=30:d=2','-c:v','libx264',str(source)],check=True)
    root=tk.Tk(); app=App(root); root.update()
    print('GUI requested height:',root.winfo_reqheight())
    app.source.set(str(source)); app.folder.set(str(folder)); app.times.insert('1.0','00:01\n00:02'); app.hold.set('1')
    with patch('gui.messagebox.showinfo') as done, patch('gui.messagebox.showerror') as errors:
        app.preview(); app.start()
        deadline=time.monotonic()+30
        while app.busy and time.monotonic()<deadline:
            root.update(); time.sleep(.05)
        assert not app.busy
        assert not errors.called, errors.call_args
        assert (folder/'評価案内付き.mp4').is_file()
        assert done.called
        assert app.progress['value']==100
        print('GUI preview → worker export → progress → completion: OK')
    root.destroy()
