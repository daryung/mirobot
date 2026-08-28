import math

import cv2
import numpy as np

from .config import *


def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def image_to_strokes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    all_paths = []

    for contour in contours:
        if cv2.arcLength(contour, True) < MIN_CONTOUR_LENGTH:
            continue

        epsilon = CONTOUR_APPROX_RATIO * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour,epsilon,True)

        path = []
        last_point = None

        for point in contour:
            x = float(point[0][0])
            y = float(point[0][1])

            if last_point is not None:
                distance = math.hypot(x - last_point[0],y - last_point[1])

                if distance < MIN_PIXEL_DISTANCE:
                    continue

            path.append((x, y))
            last_point = (x, y)

        if len(path) >= 2:
            all_paths.append(path)

    return all_paths
