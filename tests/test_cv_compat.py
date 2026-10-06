"""Compare cv_compat with OpenCV on real page renders (run on a machine that has cv2)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnocr import cv_compat as cc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
img = cv2.imread(str(ROOT / "reports/easyocr_vi/renders/báo_cáo_đo_kiểm/page_002.png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
crop = gray[800:870, 300:1500].copy()


def report(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        print(f"{name:28s} SHAPE {a.shape} vs {b.shape}")
        return
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"{name:28s} equal {np.mean(d == 0):8.4%}  max diff {d.max():.0f}")


report("cvtColor gray", cc.cvtColor(img, cc.COLOR_BGR2GRAY), gray)
report("cvtColor hsv S", cc.cvtColor(img, cc.COLOR_BGR2HSV)[..., 1], cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1])
report("cvtColor hsv V", cc.cvtColor(img, cc.COLOR_BGR2HSV)[..., 2], cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2])
report("resize area 0.45", cc.resize(img, None, fx=0.45, fy=0.45, interpolation=cc.INTER_AREA),
       cv2.resize(img, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))
report("resize area to h32", cc.resize(crop, (549, 32), interpolation=cc.INTER_AREA),
       cv2.resize(crop, (549, 32), interpolation=cv2.INTER_AREA))
small = crop[:, :400]
report("resize area upscale", cc.resize(small[::3, ::3], (200, 32), interpolation=cc.INTER_AREA),
       cv2.resize(small[::3, ::3], (200, 32), interpolation=cv2.INTER_AREA))
report("resize linear 0.25", cc.resize(gray, None, fx=0.25, fy=0.25), cv2.resize(gray, None, fx=0.25, fy=0.25))
q = cv2.resize(gray, None, fx=0.25, fy=0.25)
report("resize linear up", cc.resize(q, (gray.shape[1], gray.shape[0])), cv2.resize(q, (gray.shape[1], gray.shape[0])))
report("medianBlur", cc.medianBlur(q, 31), cv2.medianBlur(q, 31))
bg = np.maximum(cv2.medianBlur(q, 31), 1)
bgf = cv2.resize(bg, (gray.shape[1], gray.shape[0]))
report("divide", cc.divide(gray, np.maximum(bgf, 1), scale=255), cv2.divide(gray, np.maximum(bgf, 1), scale=255))
report("adaptive mean inv", cc.adaptiveThreshold(gray, 255, 0, 1, 33, 15), cv2.adaptiveThreshold(gray, 255, 0, 1, 33, 15))
report("adaptive gauss", cc.adaptiveThreshold(crop, 255, 1, 0, 25, 10), cv2.adaptiveThreshold(crop, 255, 1, 0, 25, 10))
binary = cv2.adaptiveThreshold(gray, 255, 0, 1, 33, 15)
for k in [(103, 1), (1, 60), (3, 25), (2, 2), (4, 4), (9, 9)]:
    ker = np.ones((k[1], k[0]), np.uint8)
    report(f"open {k}", cc.morphologyEx(binary, cc.MORPH_OPEN, ker), cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker))
    report(f"close {k}", cc.morphologyEx(binary, cc.MORPH_CLOSE, ker), cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker))
    report(f"dilate {k}", cc.dilate(binary, ker), cv2.dilate(binary, ker))
report("close gray crop", cc.morphologyEx(crop, cc.MORPH_CLOSE, np.ones((35, 35), np.uint8)),
       cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8)))
n1, l1, s1, _ = cc.connectedComponentsWithStats(binary, 8)
n2, l2, s2, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
print("components", n1, n2, "same set of components", sorted(map(tuple, s1[1:].tolist())) == sorted(map(tuple, s2[1:].tolist())))
n1, l1, s1, _ = cc.connectedComponentsWithStats(binary, 4)
n2, l2, s2, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
print("components4", n1, n2, "labels equal", np.array_equal(l1, l2), "stats equal", np.array_equal(s1[1:], s2[1:]))
src = np.float32([[300.3, 790.1], [1500.7, 801.2], [1500.7, 871.9], [300.3, 860.4]])
dst = np.float32([[0, 0], [1200, 0], [1200, 70], [0, 70]])
m1, m2 = cc.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(src, dst)
print("perspective max diff", np.abs(m1 - m2).max())
report("warpPerspective", cc.warpPerspective(gray, m2, (1200, 70)),
       cv2.warpPerspective(gray, m2, (1200, 70), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
rng = np.random.default_rng(0)
worst = 0
for _ in range(300):
    ang = rng.uniform(-0.3, 0.3)
    pts = rng.uniform(0, 1, (200, 2)) * [rng.uniform(20, 200), rng.uniform(5, 30)]
    pts = (pts @ np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])).astype(np.int32) + 500
    a = cc.boxPoints(cc.minAreaRect(pts))
    b = cv2.boxPoints(cv2.minAreaRect(pts))
    a = np.roll(a, 4 - a.sum(axis=1).argmin(), 0)
    b = np.roll(b, 4 - b.sum(axis=1).argmin(), 0)
    worst = max(worst, np.abs(a - b).max())
print("minAreaRect+boxPoints after roll, max corner diff", worst)
