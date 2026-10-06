"""Desktop window: open a PDF or image, run OCR, read and save the result.

Tkinter ships with the python.org Windows ARM64 installer, so the GUI needs no extra packages.
OCR runs in a worker thread; results appear page by page.
"""

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .engine import OcrEngine, default_model_dir
from .logutil import make_logger


def parse_pages(spec):
    """'1-3,5' -> [1, 2, 3, 5]; empty -> None (all pages)."""
    spec = spec.strip()
    if not spec:
        return None
    pages = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        pages += list(range(int(a), int(b or a) + 1))
    return pages


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vietnamese Document OCR (Snapdragon NPU)")
        self.geometry("1100x760")
        self.events = queue.Queue()
        self.engine = None
        self.results = []
        self.path = None
        self.busy = False

        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Open document…", command=self.open_file).pack(side="left")
        self.file_label = ttk.Label(bar, text="No file selected", width=48)
        self.file_label.pack(side="left", padx=8)
        ttk.Label(bar, text="Pages:").pack(side="left")
        self.pages = ttk.Entry(bar, width=10)
        self.pages.pack(side="left", padx=4)
        self.use_npu = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Use NPU", variable=self.use_npu).pack(side="left", padx=8)
        self.run_btn = ttk.Button(bar, text="Run OCR", command=self.run)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(bar, text="Save Markdown…", command=lambda: self.save("md")).pack(side="right")
        ttk.Button(bar, text="Save text…", command=lambda: self.save("txt")).pack(side="right", padx=4)

        self.view = tk.StringVar(value="markdown")
        tabs = ttk.Frame(self, padding=(6, 0))
        tabs.pack(fill="x")
        ttk.Radiobutton(tabs, text="Markdown (tables)", value="markdown", variable=self.view,
                        command=self.refresh).pack(side="left")
        ttk.Radiobutton(tabs, text="Plain text", value="text", variable=self.view,
                        command=self.refresh).pack(side="left", padx=8)

        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="word", font=("Segoe UI", 11), undo=False)
        scroll = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=6)
        self.status = ttk.Label(self, text="Ready", padding=6, anchor="w")
        self.status.pack(fill="x")
        self.after(100, self.poll)

    # ------------------------------------------------------------------ actions

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Documents", "*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                                                     ("All files", "*.*")])
        if path:
            self.path = Path(path)
            self.file_label.configure(text=self.path.name)

    def run(self):
        if self.busy:
            return
        if not self.path:
            messagebox.showinfo("OCR", "Open a PDF or image first.")
            return
        try:
            pages = parse_pages(self.pages.get())
        except ValueError:
            messagebox.showerror("OCR", "Pages must look like 1-3,5")
            return
        self.busy = True
        self.run_btn.state(["disabled"])
        self.results = []
        self.text.delete("1.0", "end")
        threading.Thread(target=self.worker, args=(self.path, pages, self.use_npu.get()), daemon=True).start()

    def worker(self, path, pages, use_npu):
        try:
            if self.engine is None or self.engine_npu != use_npu:
                self.events.put(("status", "Loading models (first NPU start compiles them; later starts are fast)…"))
                file_log = make_logger(echo=False)

                def log(m):
                    file_log(m)
                    self.events.put(("status", m))

                self.engine = OcrEngine(default_model_dir(), use_npu=use_npu, log=log)
                self.engine_npu = use_npu
                self.events.put(("placement", self.engine.placement))
            numbers = pages
            if numbers is None and path.suffix.lower() == ".pdf":
                import pymupdf
                with pymupdf.open(path) as pdf:
                    numbers = list(range(1, len(pdf) + 1))
            total = len(numbers) if numbers else 1
            self.events.put(("total", total))
            t0 = time.perf_counter()
            for n, img in self.engine.load_document(path, pages=numbers):
                self.events.put(("status", f"Page {n}: detecting and reading text…"))
                self.events.put(("page", self.engine.ocr_page(img, n)))
            self.events.put(("done", time.perf_counter() - t0))
        except Exception as e:  # shown to the user and logged, never swallowed
            import traceback
            make_logger(echo=False)(traceback.format_exc())
            self.events.put(("error", f"{type(e).__name__}: {e}"))

    def save(self, kind):
        if not self.results:
            messagebox.showinfo("OCR", "Nothing to save yet.")
            return
        default = (self.path.stem if self.path else "ocr") + f".{kind}"
        target = filedialog.asksaveasfilename(defaultextension=f".{kind}", initialfile=default)
        if target:
            Path(target).write_text(self.render(kind == "md"), encoding="utf-8")
            self.status.configure(text=f"Saved {target}")

    # ------------------------------------------------------------------ display

    def render(self, markdown):
        parts = []
        for r in self.results:
            parts.append(f"## Page {r.page}\n\n" + (r.markdown if markdown else r.text) + "\n")
        return "\n".join(parts)

    def refresh(self):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.render(self.view.get() == "markdown"))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.configure(text=value)
                elif kind == "placement":
                    self.placement = ", ".join(f"{k.replace('vietocr_', '')}: {v}" for k, v in value.items())
                elif kind == "total":
                    self.progress.configure(maximum=value, value=0)
                elif kind == "page":
                    self.results.append(value)
                    self.progress.step(1)
                    self.refresh()
                    s = value.seconds
                    self.status.configure(text=f"Page {value.page}: {value.lines} lines, {value.tables} tables — "
                                               f"detect {s['detect']:.1f}s, layout {s['layout']:.1f}s, "
                                               f"recognise {s['recognize']:.1f}s")
                elif kind == "done":
                    pages = len(self.results)
                    self.status.configure(text=f"Done: {pages} page(s) in {value:.1f}s "
                                               f"({value / max(pages, 1):.1f}s per page). {getattr(self, 'placement', '')}")
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                elif kind == "error":
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    self.status.configure(text="Error")
                    messagebox.showerror("OCR failed", value)
        except queue.Empty:
            pass
        self.after(100, self.poll)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
