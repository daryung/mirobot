import math

import cv2
import numpy as np
from ultralytics import YOLO

from .config import MIN_PIXEL_DISTANCE

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n-seg.pt")
    return _model


def extract_paths_from_image(image, conf=0.1):
    model = get_model()
    results = model(image, conf=conf, verbose=False)
    all_paths = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            for cnt in contours:
                path = []
                last_point = None

                for point in cnt:
                    x, y = float(point[0][0]), float(point[0][1])

                    if last_point is not None:
                        if math.hypot(x - last_point[0], y - last_point[1]) < MIN_PIXEL_DISTANCE:
                            continue

                    path.append((x, y))
                    last_point = (x, y)

                if len(path) >= 2:
                    all_paths.append(path)

    return all_paths
