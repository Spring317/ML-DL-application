"""Entry point of the packaged VNOCR.exe.

  VNOCR.exe                     desktop window
  VNOCR.exe --selftest          NPU/CPU check, result shown in a window and written to the log
  VNOCR.exe FILE -o OUT.md ...  command line (see vnocr/cli.py)
"""
import os
import sys

if getattr(sys, "frozen", False):
    os.environ.setdefault("VNOCR_MODELS", os.path.join(sys._MEIPASS, "models"))

if "--selftest" in sys.argv:
    from vnocr.selftest import run
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    try:
        report = run(args[0] if args else None)
    except Exception as e:  # report instead of a silent crash in a windowed app
        import traceback
        report = "Self-test failed:\n" + traceback.format_exc()
    import tkinter as tk
    from tkinter import scrolledtext
    win = tk.Tk()
    win.title("VNOCR self-test")
    box = scrolledtext.ScrolledText(win, width=110, height=24)
    box.insert("1.0", report)
    box.pack(fill="both", expand=True)
    win.mainloop()
else:
    from vnocr.cli import main
    main()
