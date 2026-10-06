"""Page layout helpers (copy of scripts/table_layout.py using cv_compat instead of OpenCV).

Handles photographed scans: ruling lines are fitted as slightly tilted lines,
several tables per page are supported, every detected text box is routed
exactly once, and line crops keep their aspect ratio (long lines are split at
word gaps instead of being squeezed into the recognizer width).
"""

from dataclasses import dataclass, field

from . import cv_compat as cv2
import numpy as np

RECOGNIZER_HEIGHT = 32
RECOGNIZER_MAX_WIDTH = 512  # VietOCR vgg_transformer was trained on widths <= 512


@dataclass
class Rule:
    """A ruling line fitted as ``pos = slope * t + offset``.

    For horizontal rules ``t`` is x and ``pos`` is y; for vertical rules ``t`` is y and ``pos`` is x.
    """

    slope: float
    offset: float
    start: float
    end: float

    def at(self, t):
        return self.slope * t + self.offset


@dataclass
class Table:
    bbox: tuple  # x0, y0, x1, y1
    rows: list  # horizontal Rule objects, top to bottom (outer borders included)
    cols: list  # vertical Rule objects, left to right (outer borders included)
    confidence: float = 0.0

    @property
    def num_rows(self):
        return len(self.rows) - 1

    @property
    def num_cols(self):
        return len(self.cols) - 1

    def contains(self, x, y, margin=0):
        x0, y0, x1, y1 = self.bbox
        return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin

    def row_of(self, x, y):
        """Disjoint row index: count of row rules above the point (None outside the grid)."""
        above = sum(1 for r in self.rows if r.at(x) <= y)
        idx = above - 1
        return idx if 0 <= idx < self.num_rows else None

    def col_of(self, x, y):
        left = sum(1 for c in self.cols if c.at(y) <= x)
        idx = left - 1
        return idx if 0 <= idx < self.num_cols else None


def _normalize_background(gray):
    """Flatten vignetting/uneven lighting of phone photos before thresholding."""
    k = max(31, (min(gray.shape) // 20) | 1)
    background = cv2.medianBlur(cv2.resize(gray, None, fx=0.25, fy=0.25), k // 4 | 1)
    background = cv2.resize(background, (gray.shape[1], gray.shape[0]))
    norm = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return norm


def _line_mask(binary, horizontal, length):
    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, length // 4), 3))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(3, length // 4)))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Reconnect segments broken by tilt, blur, or text touching the rule.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge)


def _fit_rules(mask, horizontal, min_span, mid, tol, min_coverage=0.5):
    """Fit ruling lines, joining collinear pieces before the length check.

    Rules are often broken: full-width section rows cut vertical rules, merged label cells
    cut horizontal ones. Each connected piece is fitted, pieces at the same position are
    joined, and a joined rule is kept if it spans ``min_span`` pixels and its pieces cover
    at least ``min_coverage`` of that extent."""
    pieces = _fit_pieces(mask, horizontal, min_piece=max(15, min_span * 0.08))
    pieces.sort(key=lambda r: r.at(mid))
    groups = []
    for r in pieces:
        if groups and abs(r.at(mid) - groups[-1][-1].at(mid)) <= tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    rules = []
    for g in groups:
        start, end = min(r.start for r in g), max(r.end for r in g)
        covered = sum(r.end - r.start for r in g)
        if end - start < min_span or covered < min_coverage * (end - start):
            continue
        longest = max(g, key=lambda r: r.end - r.start)
        rules.append(Rule(longest.slope, float(np.mean([r.at(mid) for r in g])) - longest.slope * mid,
                          start, end))
    return rules


def _fit_pieces(mask, horizontal, min_piece):
    """Fit one Rule per connected ruling component at least ``min_piece`` pixels long."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rules = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        span = w if horizontal else h
        thickness = h if horizontal else w
        if span < min_piece or thickness > span * 0.25:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        xs = xs + x
        ys = ys + y
        t, pos = (xs, ys) if horizontal else (ys, xs)
        slope, offset = np.polyfit(t.astype(np.float64), pos.astype(np.float64), 1)
        rules.append(Rule(float(slope), float(offset), float(t.min()), float(t.max())))
    return rules


def _merge_rules(rules, mid, tol):
    """Merge rules whose positions at ``mid`` are within ``tol`` (double/fragmented lines)."""
    rules = sorted(rules, key=lambda r: r.at(mid))
    merged = []
    for r in rules:
        if merged and abs(r.at(mid) - merged[-1].at(mid)) <= tol:
            m = merged[-1]
            if (r.end - r.start) > (m.end - m.start):
                merged[-1] = Rule(r.slope, r.offset, min(r.start, m.start), max(r.end, m.end))
            else:
                merged[-1] = Rule(m.slope, m.offset, min(r.start, m.start), max(r.end, m.end))
        else:
            merged.append(r)
    return merged


def detect_tables(gray, min_rows=2, min_cols=2):
    """Detect ruled tables. Returns a list of Table objects sorted top to bottom.

    A grid is accepted only when its rules intersect: each kept horizontal rule must
    cross most kept vertical rules and vice versa. Uncertain geometry yields no table,
    so the caller keeps the region as plain text instead of fabricating cells.
    """
    H, W = gray.shape[:2]
    norm = _normalize_background(gray)
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                   (max(15, W // 80)) | 1, 15)

    h_mask = _line_mask(binary, True, max(40, W // 25))
    v_mask = _line_mask(binary, False, max(30, H // 60))

    # Candidate table regions: clusters of rule pixels.
    grid = cv2.dilate(cv2.bitwise_or(h_mask, v_mask), np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)

    tables = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < W * 0.25 or h < H * 0.02:
            continue
        pad = 6
        rx0, ry0 = max(0, x - pad), max(0, y - pad)
        rx1, ry1 = min(W, x + w + pad), min(H, y + h + pad)
        hm = np.zeros_like(h_mask)
        vm = np.zeros_like(v_mask)
        hm[ry0:ry1, rx0:rx1] = h_mask[ry0:ry1, rx0:rx1]
        vm[ry0:ry1, rx0:rx1] = v_mask[ry0:ry1, rx0:rx1]

        xmid, ymid = x + w / 2, y + h / 2
        line_gap = max(12, H // 150)
        rows = _fit_rules(hm, True, min_span=w * 0.6, mid=xmid, tol=line_gap)
        cols = _fit_rules(vm, False, min_span=h * 0.6, mid=ymid, tol=line_gap)
        rows = _merge_rules(rows, xmid, tol=line_gap)
        cols = _merge_rules(cols, ymid, tol=line_gap)

        # Keep only rules supported by intersections with the perpendicular set.
        def crosses(r, c):
            # Intersection of y = a*x + b and x = c*y + d.
            xc = (c.slope * r.offset + c.offset) / (1 - c.slope * r.slope)
            yc = r.at(xc)
            tol = line_gap * 2
            return (r.start - tol <= xc <= r.end + tol) and (c.start - tol <= yc <= c.end + tol)

        rows = [r for r in rows if sum(crosses(r, c) for c in cols) >= max(2, 0.6 * len(cols))]
        cols = [c for c in cols if sum(crosses(r, c) for r in rows) >= max(2, 0.6 * len(rows))]
        if len(rows) - 1 < min_rows or len(cols) - 1 < min_cols:
            continue

        # Cells shorter than a text line are double rules, not rows.
        min_cell = max(20, H // 120)
        rows = [r for k, r in enumerate(rows) if k == 0 or r.at(xmid) - rows[k - 1].at(xmid) >= min_cell]
        cols = [c for k, c in enumerate(cols) if k == 0 or c.at(ymid) - cols[k - 1].at(ymid) >= min_cell]
        if len(rows) - 1 < min_rows or len(cols) - 1 < min_cols:
            continue

        pairs = len(rows) * len(cols)
        hits = sum(crosses(r, c) for r in rows for c in cols)
        bbox = (min(c.at(ymid) for c in cols), rows[0].at(xmid),
                max(c.at(ymid) for c in cols), rows[-1].at(xmid))
        tables.append(Table(bbox=tuple(int(v) for v in bbox), rows=rows, cols=cols,
                            confidence=hits / pairs))
    tables.sort(key=lambda t: t.bbox[1])
    return tables


def _quad_metrics(q):
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)  # tl, tr, br, bl
    center = q.mean(axis=0)
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return center, max(height, 1.0), q[:, 0].min(), q[:, 0].max()


def group_words_into_lines(quads, gap_factor=1.5):
    """Chain CRAFT word quads into tilted text lines.

    Each returned box carries the line's centre line ``y = slope * x + intercept`` and its
    height, so crops can follow the tilt of photographed pages instead of using an
    axis-aligned rectangle that also contains parts of the neighbouring lines.
    Words further apart than ``gap_factor`` line heights start a new segment, which keeps
    separate columns (e.g. "Số: ..." and "Hà Nội, ngày ...") apart.
    """
    words = []
    for q in quads:
        center, h, x0, x1 = _quad_metrics(q)
        words.append({"cx": center[0], "cy": center[1], "h": h, "x0": x0, "x1": x1})
    words.sort(key=lambda w: w["x0"])

    lines = []
    for w in words:
        best, best_dy = None, None
        for line in lines:
            last = line["words"][-1]
            gap = w["x0"] - last["x1"]
            lh = np.median([x["h"] for x in line["words"]])
            if gap > gap_factor * lh or gap < -0.5 * lh:
                continue
            predicted = line["slope"] * w["cx"] + line["intercept"]
            dy = abs(w["cy"] - predicted)
            if dy < 0.45 * lh and (best is None or dy < best_dy):
                best, best_dy = line, dy
        if best is None:
            lines.append({"words": [w], "slope": 0.0, "intercept": w["cy"]})
            continue
        best["words"].append(w)
        cx = np.array([x["cx"] for x in best["words"]])
        cy = np.array([x["cy"] for x in best["words"]])
        if np.ptp(cx) > 2 * w["h"]:
            slope, intercept = np.polyfit(cx, cy, 1)
            slope = float(np.clip(slope, -0.08, 0.08))  # at most ~4.5 degrees
            best["slope"], best["intercept"] = slope, float(np.mean(cy - slope * cx))
        else:
            best["intercept"] = float(np.mean(cy - best["slope"] * cx))

    boxes = []
    for line in lines:
        ws = line["words"]
        h = float(np.median([x["h"] for x in ws]))
        x0, x1 = min(x["x0"] for x in ws), max(x["x1"] for x in ws)
        s, b = line["slope"], line["intercept"]
        ys = [s * x0 + b, s * x1 + b]
        y0, y1 = min(ys) - h / 2, max(ys) + h / 2
        boxes.append({"x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1),
                      "xc": (x0 + x1) / 2.0, "yc": s * (x0 + x1) / 2.0 + b,
                      "w": int(x1 - x0), "h": h, "slope": s, "intercept": b})
    return boxes


def remove_colored_marks(img_bgr, min_saturation=90):
    """Paint saturated colour (red stamps, blue signatures) with the paper colour.

    Document text is black/grey; stamps produced stray detections such as "00101001"."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > min_saturation) & (hsv[:, :, 2] > 50)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = img_bgr.copy()
    paper = np.median(img_bgr[~mask], axis=0) if (~mask).any() else np.array([255, 255, 255])
    out[mask] = paper.astype(np.uint8)
    return out


def drop_stray_boxes(boxes, min_rel_height=0.45, overlap=0.6):
    """Remove specks and duplicate detections.

    A box much shorter than the page's median line height is noise (rule fragments,
    stamp texture); two boxes sharing most of the smaller one's area are the same text."""
    if not boxes:
        return boxes
    med_h = float(np.median([b["h"] for b in boxes]))
    kept = [b for b in boxes if b["h"] >= min_rel_height * med_h]
    kept.sort(key=lambda b: (b["x1"] - b["x0"]) * b["h"], reverse=True)
    out = []
    for b in kept:
        area = max((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), 1)
        dup = False
        for k in out:
            ix = max(0, min(b["x1"], k["x1"]) - max(b["x0"], k["x0"]))
            iy = max(0, min(b["y1"], k["y1"]) - max(b["y0"], k["y0"]))
            if ix * iy > overlap * area:
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def crop_line(gray, box, pad_x=None, pad_y=None, x_limits=None):
    """Extract a straightened crop of a (possibly tilted) text line box.

    The right side gets more padding than the left: CRAFT boxes often stop before a final
    "." or ":", which were the most common punctuation errors."""
    h = box["h"]
    s = box.get("slope", 0.0)
    b = box.get("intercept", box["yc"])
    px = 0.5 * h if pad_x is None else pad_x
    py = 0.2 * h if pad_y is None else pad_y
    x0, x1 = box["x0"] - px, box["x1"] + (0.9 * h if pad_x is None else pad_x)
    if x_limits is not None:
        x0, x1 = max(x0, x_limits[0]), min(x1, x_limits[1])
    x0, x1 = max(0.0, x0), min(gray.shape[1] - 1.0, x1)
    half = h / 2 + py
    src = np.float32([[x0, s * x0 + b - half], [x1, s * x1 + b - half],
                      [x1, s * x1 + b + half], [x0, s * x0 + b + half]])
    out_w = max(int(round(np.hypot(x1 - x0, s * (x1 - x0)))), 2)
    out_h = max(int(round(2 * half)), 2)
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def split_box_at_columns(box, table, margin=4):
    """Split a text box that crosses vertical rules into per-cell pieces."""
    x0, x1, y0, y1 = box["x0"], box["x1"], box["y0"], box["y1"]
    yc = (y0 + y1) / 2
    cuts = [c.at(yc) for c in table.cols[1:-1] if x0 + margin < c.at(yc) < x1 - margin]
    if not cuts:
        return [box]
    edges = [x0] + cuts + [x1]
    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 6:
            continue
        piece = dict(box)
        xc = (a + b) / 2.0
        yc = box["slope"] * xc + box["intercept"] if "slope" in box else box["yc"]
        piece.update(x0=int(a), x1=int(b), xc=xc, yc=yc, w=int(b - a), split_from_wider_box=True)
        pieces.append(piece)
    return pieces


def route_boxes(boxes, tables):
    """Assign each box to exactly one table cell or to the free-text flow.

    Returns (free_boxes, cells) where cells maps (table_idx, row, col) -> boxes.
    Boxes are split at column rules first, so a CRAFT box that merged two cells
    contributes text to both instead of being dropped into one.
    """
    free, cells = [], {}
    for b in boxes:
        placed = False
        for t_idx, t in enumerate(tables):
            if not t.contains(b["xc"], b["yc"], margin=4):
                continue
            for piece in split_box_at_columns(b, t):
                r = t.row_of(piece["xc"], piece["yc"])
                c = t.col_of(piece["xc"], piece["yc"])
                if r is None or c is None:
                    free.append(piece)
                else:
                    cells.setdefault((t_idx, r, c), []).append(piece)
            placed = True
            break
        if not placed:
            free.append(b)
    return free, cells


def ink_boxes_for_empty_cells(gray, tables, cells, margin=10):
    """Find text in table cells where the detector found nothing.

    CRAFT tends to drop short isolated glyphs (row numbers such as "1" in an STT column).
    The table rules already bound those regions, so look for ink inside each empty cell
    and return one line box per horizontal ink band.
    """
    norm = _normalize_background(gray)
    found = {}
    for t_idx, t in enumerate(tables):
        for r in range(t.num_rows):
            for c in range(t.num_cols):
                if (t_idx, r, c) in cells:
                    continue
                ymid = (t.rows[r].at((t.cols[c].at(0) + t.cols[c + 1].at(0)) / 2) +
                        t.rows[r + 1].at((t.cols[c].at(0) + t.cols[c + 1].at(0)) / 2)) / 2
                x0 = int(t.cols[c].at(ymid)) + margin
                x1 = int(t.cols[c + 1].at(ymid)) - margin
                xmid = (x0 + x1) / 2
                y0 = int(t.rows[r].at(xmid)) + margin
                y1 = int(t.rows[r + 1].at(xmid)) - margin
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                cell = norm[y0:y1, x0:x1]
                ink = (cell < 128).astype(np.uint8)
                ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                rows_ink = ink.sum(axis=1) > 0
                if ink.sum() < 40:
                    continue
                # Split into horizontal bands (one per text line).
                bands, start = [], None
                for i, on in enumerate(np.append(rows_ink, False)):
                    if on and start is None:
                        start = i
                    elif not on and start is not None:
                        if i - start >= 8:
                            bands.append((start, i))
                        start = None
                for a, b in bands:
                    cols_ink = np.nonzero(ink[a:b].sum(axis=0))[0]
                    bx0, bx1 = x0 + int(cols_ink.min()), x0 + int(cols_ink.max()) + 1
                    by0, by1 = y0 + a, y0 + b
                    h = float(by1 - by0)
                    found.setdefault((t_idx, r, c), []).append(
                        {"x0": bx0, "x1": bx1, "y0": by0, "y1": by1, "xc": (bx0 + bx1) / 2.0,
                         "yc": (by0 + by1) / 2.0, "w": bx1 - bx0, "h": h, "slope": 0.0,
                         "intercept": (by0 + by1) / 2.0, "from_cell_ink": True})
    return found


def sort_reading_order(boxes):
    """Group boxes into visual lines by vertical overlap, then order left to right."""
    lines = []
    for b in sorted(boxes, key=lambda b: b["yc"]):
        for line in lines:
            ly = np.mean([x["yc"] for x in line])
            lh = np.mean([x["h"] for x in line])
            if abs(b["yc"] - ly) < 0.55 * max(lh, b["h"]):
                line.append(b)
                break
        else:
            lines.append([b])
    ordered = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda b: b["x0"]))
    return ordered


def two_column_split(boxes, page_w):
    """Return (left, right) when the boxes form two side-by-side columns (e.g. signature
    blocks): both sides non-empty and no box crossing the page centre. Otherwise None."""
    if not boxes:
        return None
    mid = page_w / 2
    if any(b["x0"] < mid - 10 and b["x1"] > mid + 10 for b in boxes):
        return None
    left = [b for b in boxes if b["xc"] < mid]
    right = [b for b in boxes if b["xc"] >= mid]
    if not left or not right:
        return None
    return left, right


def pad_box(x0, x1, y0, y1, W, H):
    """Pad horizontally by ~half a line height so leading bullets and dashes stay in the crop."""
    h = max(y1 - y0, 1)
    px, py = int(0.5 * h), max(3, int(0.12 * h))
    return max(0, x0 - px), min(W, x1 + px), max(0, y0 - py), min(H, y1 + py)


def _split_points(binary_line, max_w):
    """Choose cut columns at the widest ink-free gaps so each piece is <= max_w."""
    ink = (binary_line < 128).sum(axis=0)
    w = binary_line.shape[1]
    cuts, start = [], 0
    while w - start > max_w:
        lo, hi = start + max_w // 2, start + max_w
        window = ink[lo:hi]
        # Prefer the longest run of empty columns; fall back to least ink.
        empty = window == 0
        best, best_len, run_start = None, 0, None
        for i, e in enumerate(np.append(empty, False)):
            if e and run_start is None:
                run_start = i
            elif not e and run_start is not None:
                if i - run_start > best_len:
                    best, best_len = (run_start + i) // 2, i - run_start
                run_start = None
        cut = lo + (best if best is not None else int(np.argmin(window)))
        cuts.append(cut)
        start = cut
    return cuts


def normalize_crop(crop_gray):
    """Divide out the local background (shaded table headers, vignetting) and stretch
    contrast, keeping greyscale anti-aliasing that binarisation would destroy."""
    h = crop_gray.shape[0]
    k = max(3, (h // 2) | 1)
    background = cv2.morphologyEx(crop_gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    norm = cv2.divide(crop_gray, np.maximum(background, 1), scale=255)
    lo = np.percentile(norm, 2)
    return np.clip((norm.astype(np.float32) - lo) * 255.0 / max(255 - lo, 1), 0, 255).astype(np.uint8)


def line_crop_tensors(crop_gray, target_w=RECOGNIZER_MAX_WIDTH, mode="binary", squeeze=False):
    """Convert a line crop to one or more (1,3,32,target_w) tensors without squashing.

    ``mode`` is "binary" (adaptive Gaussian threshold) or "normalized" (background-divided
    greyscale). Returns a list of (tensor, content_width). Lines longer than ``target_w``
    at 32 px height are split at word gaps; pieces are recognised separately and the
    caller joins their text with spaces.
    """
    # Light text on a dark background (console screenshots): the text is the minority of
    # pixels, so it sits on the bright side of the median. Dim photos and shaded headers keep
    # dark text (darkest pixels further from the median) and must not be inverted.
    p5, med, p95 = np.percentile(crop_gray, [5, 50, 95])
    if med < 128 and (p95 - med) > 1.5 * (med - p5):
        crop_gray = 255 - crop_gray
    if mode == "binary":
        crop = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 25, 10)
    elif mode == "normalized":
        crop = normalize_crop(crop_gray)
    else:
        raise ValueError(f"unknown crop mode {mode!r}")
    h, w = crop.shape[:2]
    scaled_w = max(10, int(round(RECOGNIZER_HEIGHT * w / max(h, 1))))
    if squeeze and scaled_w > target_w:   # ablation only: old behaviour, squash long lines to fit
        scaled_w = target_w
    line = cv2.resize(crop, (scaled_w, RECOGNIZER_HEIGHT), interpolation=cv2.INTER_AREA)
    bounds = [0] + _split_points(line, target_w) + [scaled_w]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        piece = line[:, a:b]
        canvas = np.full((RECOGNIZER_HEIGHT, target_w), 255, dtype=np.uint8)
        canvas[:, :piece.shape[1]] = piece[:, :target_w]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        out.append((tensor, min(piece.shape[1], target_w)))
    return out


@dataclass
class PageLayout:
    """Ordered recognition units for one page plus the table structure to rebuild."""

    units: list = field(default_factory=list)  # dicts: kind, box, and table/row/col for cells
    tables: list = field(default_factory=list)


def build_page_layout(boxes, tables, page_w, gray=None):
    """Order free text and table cells. Free text above, between and below tables keeps
    its position; side-by-side blocks (signatures) are read column by column.
    With ``gray``, empty cells are searched for ink the detector missed."""
    free, cells = route_boxes(boxes, tables)
    if gray is not None:
        cells.update(ink_boxes_for_empty_cells(gray, tables, cells))
    units = []
    bands = [t.bbox[1] for t in tables] + [float("inf")]
    top = -float("inf")
    for t_idx, band_end in enumerate(bands):
        band = [b for b in free if top <= b["yc"] < band_end]
        split = two_column_split(band, page_w) if t_idx == len(tables) and tables else None
        if split:
            for side, part in zip(("left", "right"), split):
                units += [{"kind": f"text_{side}", "box": b} for b in sort_reading_order(part)]
        else:
            units += [{"kind": "text", "box": b} for b in sort_reading_order(band)]
        if t_idx < len(tables):
            t = tables[t_idx]
            for r in range(t.num_rows):
                for c in range(t.num_cols):
                    for b in sort_reading_order(cells.get((t_idx, r, c), [])):
                        units.append({"kind": "cell", "table": t_idx, "row": r, "col": c, "box": b})
            top = t.bbox[3]
    return PageLayout(units=units, tables=tables)


def _md_escape(text):
    return text.replace("|", "\\|")


def assemble_page(layout, texts):
    """Rebuild page text from per-unit transcriptions.

    Returns (plain_text, markdown). Table dimensions come from the Table records,
    never from individual text items; empty cells are preserved.
    """
    plain, md = [], []
    grids = {i: [["" for _ in range(t.num_cols)] for _ in range(t.num_rows)]
             for i, t in enumerate(layout.tables)}
    emitted = set()
    for unit, text in zip(layout.units, texts):
        if unit["kind"] == "cell":
            t_idx = unit["table"]
            cell = grids[t_idx][unit["row"]][unit["col"]]
            grids[t_idx][unit["row"]][unit["col"]] = f"{cell} {text}".strip()
            if t_idx not in emitted:
                emitted.add(t_idx)
                md.append(("table", t_idx))
        elif text.strip():
            md.append(("text", text))
    out_md = []
    for kind, value in md:
        if kind == "text":
            out_md.append(value)
            plain.append(value)
        else:
            grid = grids[value]
            out_md.append("")
            out_md.append("| " + " | ".join(_md_escape(c) for c in grid[0]) + " |")
            out_md.append("|" + " --- |" * len(grid[0]))
            for row in grid[1:]:
                out_md.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
            out_md.append("")
            plain.extend(" ".join(c for c in row if c) for row in grid)
    return " ".join(p for p in plain if p), "\n".join(out_md)
