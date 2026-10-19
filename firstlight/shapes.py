"""Antialiased 2D masks for the picture outside the window (coordinates in base units, px = one pixel)."""
import numpy as np


def seg_mask(X, Y, p0, p1, w, px):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    u = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
    d = np.hypot(X - x0 - u * dx, Y - y0 - u * dy)
    return np.clip((w / 2 - d) / px + 0.5, 0, 1)


def poly_mask(X, Y, pts, px):
    pts = np.asarray(pts, np.float32)
    area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area > 0:                                   # (y points down) walk the edges with the inside on the left
        pts = pts[::-1]
    m = np.ones(np.broadcast(X, Y).shape, np.float32)
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
        nx, ny = (y1 - y0), -(x1 - x0)
        d = ((X - x0) * nx + (Y - y0) * ny) / np.hypot(nx, ny)
        m = np.minimum(m, np.clip(0.5 + d / px, 0, 1))
    return m


def ellipse_mask(X, Y, cx, cy, a, b, px):
    r = np.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2)
    return np.clip((1 - r) * min(a, b) / px + 0.5, 0, 1)


def rect_mask(X, Y, x0, x1, y0, y1, px):
    return (np.clip((X - x0) / px + 0.5, 0, 1) * np.clip((x1 - X) / px + 0.5, 0, 1)
            * np.clip((Y - y0) / px + 0.5, 0, 1) * np.clip((y1 - Y) / px + 0.5, 0, 1))
