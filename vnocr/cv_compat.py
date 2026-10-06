"""The subset of OpenCV used by the OCR pipeline, implemented with numpy, scipy and Pillow.

OpenCV publishes no Windows ARM64 wheels, and the QNN execution provider needs a native
ARM64 Python, so the app cannot depend on cv2. Each function follows OpenCV's definition
(border handling, anchors, rounding); tests/test_cv_compat.py compares them with cv2.
"""

import math

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.spatial import ConvexHull

MORPH_RECT = 0
MORPH_OPEN = 2
MORPH_CLOSE = 3
THRESH_BINARY = 0
THRESH_BINARY_INV = 1
ADAPTIVE_THRESH_MEAN_C = 0
ADAPTIVE_THRESH_GAUSSIAN_C = 1
INTER_LINEAR = 1
INTER_AREA = 3
BORDER_REPLICATE = 1
COLOR_BGR2GRAY = 6
COLOR_BGR2HSV = 40
COLOR_GRAY2RGB = 8
CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA = range(5)


def _u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- colour

def cvtColor(img, code):
    if code == COLOR_BGR2GRAY:
        b, g, r = (img[..., i].astype(np.int32) for i in range(3))
        return ((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14).astype(np.uint8)  # OpenCV fixed point
    if code == COLOR_GRAY2RGB:
        return np.repeat(img[..., None], 3, axis=2)
    if code == COLOR_BGR2HSV:
        f = img.astype(np.float32)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        v = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        diff = v - mn
        s = np.where(v > 0, diff * 255.0 / np.maximum(v, 1e-6), 0)
        d = np.where(diff > 0, diff, 1)
        h = np.where(v == r, (g - b) * 60.0 / d, np.where(v == g, 120.0 + (b - r) * 60.0 / d, 240.0 + (r - g) * 60.0 / d))
        h = np.where(diff > 0, np.where(h < 0, h + 360.0, h), 0) / 2.0
        return np.stack([_u8(h), _u8(s), _u8(v)], axis=-1)
    raise NotImplementedError(code)


# ---------------------------------------------------------------- geometry

def getStructuringElement(shape, ksize):
    w, h = ksize
    return np.ones((h, w), np.uint8)


def _offsets(k):
    """OpenCV's default anchor is the kernel centre (k // 2)."""
    return -(k // 2), k - 1 - k // 2


def _morph(img, kernel, op):
    kh, kw = kernel.shape
    if op == "dilate":
        f, pad_val = ndimage.maximum_filter, 0
    else:
        f, pad_val = ndimage.minimum_filter, 255 if img.dtype == np.uint8 else np.inf
    # scipy centres even kernels at k // 2 like OpenCV; out-of-image pixels never win.
    return f(img, size=(kh, kw), mode="constant", cval=pad_val)


def dilate(img, kernel, iterations=1):
    out = img
    for _ in range(iterations):
        out = _morph(out, np.asarray(kernel), "dilate")
    return out


def erode(img, kernel, iterations=1):
    out = img
    for _ in range(iterations):
        out = _morph(out, np.asarray(kernel), "erode")
    return out


def morphologyEx(img, op, kernel):
    kernel = np.asarray(kernel)
    if op == MORPH_OPEN:
        return dilate(erode(img, kernel), kernel)
    if op == MORPH_CLOSE:
        return erode(dilate(img, kernel), kernel)
    raise NotImplementedError(op)


def bitwise_or(a, b):
    return np.bitwise_or(a, b)


def threshold(img, thresh, maxval, ttype):
    if ttype == THRESH_BINARY:
        out = np.where(img > thresh, maxval, 0)
    elif ttype == THRESH_BINARY_INV:
        out = np.where(img > thresh, 0, maxval)
    else:
        raise NotImplementedError(ttype)
    return thresh, out.astype(img.dtype)


def medianBlur(img, k):
    """Exact k x k median of a uint8 image (replicated border). For each grey level t the window
    count of pixels <= t comes from an integral image; the median is the number of levels whose
    count is still below half the window - far faster than a sort-based filter for large k."""
    if img.dtype != np.uint8 or k <= 5:
        return ndimage.median_filter(img, size=k, mode="nearest")
    r = k // 2
    pad = np.pad(img, r, mode="edge")
    h, w = img.shape
    half = (k * k) // 2 + 1
    lo, hi = int(img.min()), int(img.max())
    out = np.full((h, w), lo, np.int32)
    for t in range(lo, hi):
        le = (pad <= t).astype(np.int32)
        ii = np.zeros((le.shape[0] + 1, le.shape[1] + 1), np.int32)
        ii[1:, 1:] = le.cumsum(0).cumsum(1)
        count = ii[k:k + h, k:k + w] - ii[:h, k:k + w] - ii[k:k + h, :w] + ii[:h, :w]
        out += count < half
    return out.astype(np.uint8)


def divide(a, b, scale=1.0):
    out = a.astype(np.float64) * scale / np.maximum(b.astype(np.float64), 1e-12)
    out = np.where(b == 0, 0, out)
    return _u8(out)


def _gaussian_kernel(k):
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    x = np.arange(k) - (k - 1) / 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def adaptiveThreshold(img, maxval, method, ttype, block, c):
    f = img.astype(np.float64)
    if method == ADAPTIVE_THRESH_MEAN_C:
        mean = ndimage.uniform_filter(f, size=block, mode="nearest")
    else:
        g = _gaussian_kernel(block)
        mean = ndimage.correlate1d(ndimage.correlate1d(f, g, axis=0, mode="nearest"), g, axis=1, mode="nearest")
    mean = np.rint(mean)                                   # OpenCV filters into uint8 first
    above = (img.astype(np.int32) - mean.astype(np.int32)) > -int(round(c))
    if ttype == THRESH_BINARY:
        return np.where(above, maxval, 0).astype(np.uint8)
    return np.where(above, 0, maxval).astype(np.uint8)


# ---------------------------------------------------------------- resizing

def _area_weights(n_in, n_out, scale):
    """OpenCV INTER_AREA downscaling: source pixels weighted by their overlap with
    [o * scale, (o + 1) * scale), where scale = 1 / fx (not n_in / n_out)."""
    w = np.zeros((n_out, n_in), np.float64)
    for o in range(n_out):
        a, b = o * scale, min((o + 1) * scale, n_in)
        i0, i1 = int(math.floor(a)), min(int(math.ceil(b)), n_in)
        for i in range(i0, i1):
            w[o, i] = max(0.0, min(b, i + 1) - max(a, i))
        w[o] /= max(w[o].sum(), 1e-12)
    return w


def _linear_weights(n_in, n_out, scale, area_upscale=False):
    w = np.zeros((n_out, n_in), np.float64)
    inv = 1.0 / scale
    for o in range(n_out):
        if area_upscale:   # OpenCV INTER_AREA when enlarging
            x0 = int(math.floor(o * scale))
            fx = (o + 1) - (x0 + 1) * inv
            fx = 0.0 if fx <= 0 else fx - math.floor(fx)
        else:
            x = (o + 0.5) * scale - 0.5
            x0 = int(math.floor(x))
            fx = x - x0
        fx = round(fx * 2048) / 2048          # OpenCV's 11-bit interpolation coefficients
        for i, wt in ((x0, 1 - fx), (x0 + 1, fx)):
            w[o, min(max(i, 0), n_in - 1)] += wt
    return w


def resize(img, dsize, fx=None, fy=None, interpolation=INTER_LINEAR):
    h, w = img.shape[:2]
    if dsize is None or tuple(dsize) == (0, 0):
        ow, oh = int(round(w * fx)), int(round(h * fy))
        sx, sy = 1.0 / fx, 1.0 / fy
    else:
        ow, oh = int(dsize[0]), int(dsize[1])
        sx, sy = w / ow, h / oh
    area = interpolation == INTER_AREA

    def weights(n_in, n_out, scale):
        if area and scale > 1:
            return _area_weights(n_in, n_out, scale)
        return _linear_weights(n_in, n_out, scale, area_upscale=area)

    wy, wx = sparse.csr_matrix(weights(h, oh, sy)), sparse.csr_matrix(weights(w, ow, sx))
    f = img.astype(np.float64)

    def apply(ch):
        return np.asarray((wx @ np.asarray(wy @ ch).T).T)

    out = apply(f) if f.ndim == 2 else np.stack([apply(f[..., c]) for c in range(f.shape[2])], axis=-1)
    return _u8(out) if img.dtype == np.uint8 else out.astype(img.dtype)


def getPerspectiveTransform(src, dst):
    a, bvec = [], []
    for (x, y), (u, v) in zip(np.asarray(src, np.float64), np.asarray(dst, np.float64)):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); bvec.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); bvec.append(v)
    h = np.linalg.solve(np.array(a), np.array(bvec))
    return np.append(h, 1.0).reshape(3, 3)


def warpPerspective(img, m, dsize, flags=INTER_LINEAR, borderMode=BORDER_REPLICATE):
    ow, oh = dsize
    minv = np.linalg.inv(m)
    ys, xs = np.mgrid[0:oh, 0:ow].astype(np.float64)
    den = minv[2, 0] * xs + minv[2, 1] * ys + minv[2, 2]
    sx = (minv[0, 0] * xs + minv[0, 1] * ys + minv[0, 2]) / den
    sy = (minv[1, 0] * xs + minv[1, 1] * ys + minv[1, 2]) / den
    sx, sy = np.rint(sx * 32) / 32, np.rint(sy * 32) / 32     # OpenCV samples on a 1/32 pixel grid
    out = ndimage.map_coordinates(img.astype(np.float64), [sy, sx], order=1, mode="nearest")
    return _u8(out)


# ---------------------------------------------------------------- components and shapes

def connectedComponentsWithStats(img, connectivity=8):
    structure = np.ones((3, 3), int) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    labels, n = ndimage.label(img != 0, structure=structure)
    stats = np.zeros((n + 1, 5), np.int32)
    stats[0] = [0, 0, img.shape[1], img.shape[0], int((labels == 0).sum())]
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        stats[i] = [xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start, areas[i]]
    centroids = np.zeros((n + 1, 2))
    return n + 1, labels.astype(np.int32), stats, centroids


def minAreaRect(points):
    """Smallest-area rotated rectangle (rotating calipers over the convex hull)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    uniq = np.unique(pts, axis=0)
    if len(uniq) < 3 or np.linalg.matrix_rank(uniq - uniq[0]) < 2:
        lo, hi = uniq.min(axis=0), uniq.max(axis=0)
        return ((lo + hi) / 2).tolist(), (hi - lo).tolist(), 0.0
    hull = uniq[ConvexHull(uniq).vertices]
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        n = np.linalg.norm(e)
        if n == 0:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = hull @ u, hull @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0] - 1e-9:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    _, u, v, u0, u1, v0, v1 = best
    corners = np.array([u0 * u + v0 * v, u1 * u + v0 * v, u1 * u + v1 * v, u0 * u + v1 * v])
    return corners  # boxPoints() accepts this directly


def boxPoints(rect):
    """Corners of minAreaRect's rectangle, in clockwise order (image coordinates, y down)."""
    if isinstance(rect, np.ndarray):
        c = rect
    else:
        (cx, cy), (w, h), _ = rect
        c = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    center = c.mean(axis=0)
    ang = np.arctan2(c[:, 1] - center[1], c[:, 0] - center[0])
    return c[np.argsort(ang)].astype(np.float32)   # increasing angle = clockwise on screen


# ---------------------------------------------------------------- image IO

def imread_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))
