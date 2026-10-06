"""Log to %LOCALAPPDATA%\\vnocr\\vnocr.log (and stderr when there is a console)."""
import datetime
import os
import sys
from pathlib import Path


def log_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    d = Path(base) / "vnocr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_logger(echo=True):
    path = log_dir() / "vnocr.log"

    def log(msg):
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if echo and sys.stderr is not None:
            try:
                print(msg, file=sys.stderr)
            except Exception:
                pass

    log.path = path
    return log
