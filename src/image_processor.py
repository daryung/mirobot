import math

import cv2
import numpy as np

from .config import (
    CANNY_HIGH,
    CANNY_LOW,
    CANVAS_HEIGHT,
    CANVAS_MARGIN,
    CANVAS_WIDTH,
    MAX_CONTOURS,
    MAX_TOTAL_POINTS,
    MIN_CONTOUR_LENGTH,
    MIN_IMAGE_POINT_DISTANCE,
    MIN_PIXEL_DISTANCE,
)


def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def image_to_strokes(image):
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > 1200:
        scale = 1200.0 / max_side
        image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    candidates = []
    for contour in contours:
        length = cv2.arcLength(contour, False)
        if length < MIN_CONTOUR_LENGTH:
            continue

        pts = []
        last_point = None

        for p in contour:
            x, y = float(p[0][0]), float(p[0][1])

            if last_point is not None:
                if math.hypot(x - last_point[0], y - last_point[1]) < MIN_PIXEL_DISTANCE:
                    continue

            pts.append((x, y))
            last_point = (x, y)

        if len(pts) >= 2:
            candidates.append((length, pts))

    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:MAX_CONTOURS]

    if not candidates:
        return []

    raw_strokes = [pts for _, pts in candidates]
    all_points = [p for stroke in raw_strokes for p in stroke]
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    usable_w = CANVAS_WIDTH - 2 * CANVAS_MARGIN
    usable_h = CANVAS_HEIGHT - 2 * CANVAS_MARGIN
    scale = min(usable_w / width, usable_h / height)

    draw_w = width * scale
    draw_h = height * scale
    offset_x = (CANVAS_WIDTH - draw_w) / 2.0
    offset_y = (CANVAS_HEIGHT - draw_h) / 2.0

    scaled_strokes = []
    total_points = 0

    for stroke in raw_strokes:
        converted = []
        last_point = None

        for x, y in stroke:
            cx = offset_x + (x - min_x) * scale
            cy = offset_y + (y - min_y) * scale
            cx = max(0.0, min(float(CANVAS_WIDTH), cx))
            cy = max(0.0, min(float(CANVAS_HEIGHT), cy))

            if last_point is not None:
                if math.hypot(cx - last_point[0], cy - last_point[1]) < MIN_IMAGE_POINT_DISTANCE:
                    continue

            converted.append((cx, cy))
            last_point = (cx, cy)

        if len(converted) >= 2:
            scaled_strokes.append(converted)
            total_points += len(converted)

        if total_points >= MAX_TOTAL_POINTS:
            break

    return optimize_stroke_order(scaled_strokes)


def optimize_stroke_order(strokes):
    if not strokes:
        return []

    remaining = [stroke[:] for stroke in strokes]
    ordered = [remaining.pop(0)]

    while remaining:
        current = ordered[-1][-1]
        best_index = 0
        best_reverse = False
        best_distance = float("inf")

        for i, stroke in enumerate(remaining):
            d_start = math.hypot(
                stroke[0][0] - current[0],
                stroke[0][1] - current[1],
            )
            d_end = math.hypot(
                stroke[-1][0] - current[0],
                stroke[-1][1] - current[1],
            )

            if d_start < best_distance:
                best_distance = d_start
                best_index = i
                best_reverse = False

            if d_end < best_distance:
                best_distance = d_end
                best_index = i
                best_reverse = True

        next_stroke = remaining.pop(best_index)
        if best_reverse:
            next_stroke.reverse()
        ordered.append(next_stroke)

    return ordered
