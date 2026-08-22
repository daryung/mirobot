import math
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

import serial
import wlkatapython
import numpy as np


SERIAL_PORT = "COM7"
BAUD_RATE = 115200
ROBOT_ADDRESS = -1
ROBOT_SPEED = 2000
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
CANVAS_MARGIN = 20
RULER_LEFT_WIDTH = 48
RULER_BOTTOM_HEIGHT = 38
RULER_CM_COUNT = 10

ROBOT_X_MIN = 170.0
ROBOT_X_MAX = 245.0
ROBOT_Y_MIN = -100.0
ROBOT_Y_MAX = 0.0
PEN_UP_Z = 127.5
PEN_DOWN_Z = 125.0
ROBOT_RX = 0.0
ROBOT_RY = 0.0
ROBOT_RZ = 0.0
MIN_PIXEL_DISTANCE = 6
MEASUREMENT_INTERVAL = 0.08

CANNY_LOW = 60
CANNY_HIGH = 160
MIN_CONTOUR_LENGTH = 80.0
CONTOUR_APPROX_RATIO = 0.020
MIN_IMAGE_POINT_DISTANCE = 6.0
MAX_CONTOURS = 100
MAX_TOTAL_POINTS = 1500
SAVE_FILENAME = "C:/Users/SAMSUNG/OneDrive/바탕 화면/로봇/참고자료/captured_target.jpg"


class DrawingRobotApp:
    def __init__(self, root):
        self.root = root
        self.root.resizable(False, False)

        self.serial1 = None
        self.mirobot1 = None

        self.connected = False
        self.robot_running = False
        self.stop_requested = False

        self.mouse_drawing = False
        self.current_stroke = []
        self.strokes = []
        self.last_mouse_point = None
        self.loaded_image_path = None

        self.live_marker_id = None
        self.live_prev_canvas_point = None

        self.create_widgets()

    def create_widgets(self):
        connection_frame = tk.Frame(self.root)
        connection_frame.pack(padx=10, pady=(10, 3), fill="x")

        tk.Button(
            connection_frame,
            text="로봇 연결",
            width=12,
            command=self.connect_robot,
        ).pack(side="left", padx=3)

        tk.Button(
            connection_frame,
            text="호밍",
            width=10,
            command=self.start_homing,
        ).pack(side="left", padx=3)

        tk.Button(
            connection_frame,
            text="제로 위치",
            width=13,
            command=self.start_zero,
        ).pack(side="left", padx=3)

        image_frame = tk.Frame(self.root)
        image_frame.pack(padx=10, pady=3, fill="x")

        tk.Button(
            image_frame,
            text="사진 불러오기",
            width=14,
            command=self.load_image,
        ).pack(side="left", padx=3)

        tk.Button(
            image_frame,
            text="좌표 TXT 저장",
            width=14,
            command=self.save_robot_coordinates,
        ).pack(side="left", padx=3)

        tk.Button(
            image_frame,
            text="화면 지우기",
            width=12,
            command=self.clear_canvas,
        ).pack(side="left", padx=3)

        robot_frame = tk.Frame(self.root)
        robot_frame.pack(padx=10, pady=3, fill="x")

        tk.Button(
            robot_frame,
            text="중앙 이동",
            width=14,
            command=self.start_move_test,
        ).pack(side="left", padx=3)

        tk.Button(
            robot_frame,
            text="드로잉",
            width=14,
            command=self.start_robot_drawing,
        ).pack(side="left", padx=3)

        tk.Button(
            robot_frame,
            text="좌표 측정 모드",
            width=14,
            command=self.start_measurement_drawing,
        ).pack(side="left", padx=3)

        tk.Button(
            robot_frame,
            text="정지",
            width=10,
            command=self.stop_robot,
        ).pack(side="left", padx=3)

        tk.Button(
            image_frame,
            text="YOLO 세그메테이션",
            width=14,
            command=self.mode_live_capture,
        ).pack(side="left", padx=3)

        self.status_label = tk.Label(
            self.root,
            text="상태: 연결 안 됨",
            anchor="w",
        )
        self.status_label.pack(padx=10, pady=(3, 0), fill="x")

        self.coordinate_label = tk.Label(
            self.root,
            text="좌표: X=170.00, Y=-100.00, Z=--",
            anchor="w",
        )
        self.coordinate_label.pack(padx=10, pady=(2, 5), fill="x")

        robot_x_size = ROBOT_X_MAX - ROBOT_X_MIN
        robot_y_size = ROBOT_Y_MAX - ROBOT_Y_MIN

        self.info_label = tk.Label(
            self.root,
            text=f"사진 또는 마우스 그림을 {robot_y_size:.0f}mm × {robot_x_size:.0f}mm 영역에 맞춰 그립니다.",
            anchor="w",
        )
        self.info_label.pack(padx=10, pady=(0, 5), fill="x")

        canvas_area = tk.Frame(self.root)
        canvas_area.pack(padx=10, pady=10)

        self.left_ruler = tk.Canvas(
            canvas_area,
            width=RULER_LEFT_WIDTH,
            height=CANVAS_HEIGHT,
            bg=self.root.cget("bg"),
            highlightthickness=0,
        )
        self.left_ruler.grid(row=0, column=0, sticky="ns")

        self.canvas = tk.Canvas(
            canvas_area,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="white",
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=1)

        tk.Frame(
            canvas_area,
            width=RULER_LEFT_WIDTH,
            height=RULER_BOTTOM_HEIGHT,
        ).grid(row=1, column=0)

        self.bottom_ruler = tk.Canvas(
            canvas_area,
            width=CANVAS_WIDTH,
            height=RULER_BOTTOM_HEIGHT,
            bg=self.root.cget("bg"),
            highlightthickness=0,
        )
        self.bottom_ruler.grid(row=1, column=1, sticky="ew")

        self.draw_rulers()

        self.canvas.bind("<ButtonPress-1>", self.mouse_press)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.canvas.bind("<Motion>", self.mouse_move)

        self.root.protocol("WM_DELETE_WINDOW", self.close_program)

    def draw_rulers(self):
        self.left_ruler.delete("all")
        self.bottom_ruler.delete("all")

        horizontal_cm = 10.0
        vertical_cm = 7.5

        for cm in range(11):
            x = (cm / horizontal_cm) * CANVAS_WIDTH

            tick_x = min(x, CANVAS_WIDTH - 1)
            self.bottom_ruler.create_line(
                tick_x, 0, tick_x, 9, width=1
            )

            if cm == 0:
                text_x = 1
                anchor = "nw"
            elif cm == 10:
                text_x = CANVAS_WIDTH - 1
                anchor = "ne"
            else:
                text_x = x
                anchor = "n"

            self.bottom_ruler.create_text(
                text_x,
                14,
                text=str(cm),
                anchor=anchor,
                font=("Arial", 9)
            )

        self.bottom_ruler.create_text(
            CANVAS_WIDTH / 2,
            RULER_BOTTOM_HEIGHT - 1,
            text="cm",
            anchor="s",
            font=("Arial", 9)
        )

        step_cm = 1
        step_count = int(vertical_cm / step_cm)

        for i in range(step_count + 1):
            cm = i * step_cm

            y = CANVAS_HEIGHT - (
                cm / vertical_cm
            ) * CANVAS_HEIGHT

            tick_y = max(
                0,
                min(CANVAS_HEIGHT - 1, y)
            )

            self.left_ruler.create_line(
                RULER_LEFT_WIDTH - 9,
                tick_y,
                RULER_LEFT_WIDTH - 1,
                tick_y,
                width=1
            )

            if i == step_count:
                text_y = 1
                anchor = "ne"
            elif i == 0:
                text_y = CANVAS_HEIGHT - 1
                anchor = "se"
            else:
                text_y = y
                anchor = "e"

            if cm.is_integer():
                label = str(int(cm))
            else:
                label = str(cm)

            self.left_ruler.create_text(
                RULER_LEFT_WIDTH - 13,
                text_y,
                text=label,
                anchor=anchor,
                font=("Arial", 9)
            )

        self.left_ruler.create_text(
            8,
            CANVAS_HEIGHT / 2,
            text="cm",
            angle=90,
            font=("Arial", 9)
        )


    def canvas_to_robot(self, canvas_x, canvas_y):
        canvas_x = max(0.0, min(float(CANVAS_WIDTH), float(canvas_x)))
        canvas_y = max(0.0, min(float(CANVAS_HEIGHT), float(canvas_y)))

        robot_x = ROBOT_X_MIN + (canvas_y / CANVAS_HEIGHT) * (
            ROBOT_X_MAX - ROBOT_X_MIN
        )
        robot_y = ROBOT_Y_MIN + (canvas_x / CANVAS_WIDTH) * (
            ROBOT_Y_MAX - ROBOT_Y_MIN
        )

        robot_x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, robot_x))
        robot_y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, robot_y))

        return robot_x, robot_y

    def robot_to_canvas(self, robot_x, robot_y):
        robot_x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, float(robot_x)))
        robot_y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, float(robot_y)))

        canvas_y = (robot_x - ROBOT_X_MIN) / (ROBOT_X_MAX - ROBOT_X_MIN) * CANVAS_HEIGHT
        canvas_x = (robot_y - ROBOT_Y_MIN) / (ROBOT_Y_MAX - ROBOT_Y_MIN) * CANVAS_WIDTH
        return canvas_x, canvas_y


    def load_image(self):
        if self.robot_running:
            self.show_running_warning()
            return

        if cv2 is None or np is None:
            messagebox.showerror(
                "OpenCV 필요",
                "사진 기능에는 opencv-python과 numpy가 필요합니다.\n\n"
                "명령 프롬프트에서:\n"
                "pip install opencv-python numpy",
            )
            return

        path = filedialog.askopenfilename(
            title="그릴 사진 선택",
            filetypes=[
                ("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.set_status("사진에서 윤곽선을 추출하는 중...")
            image = self.read_image_unicode(path)
            if image is None:
                raise ValueError("이미지를 읽을 수 없습니다.")

            strokes = self.image_to_strokes(image)
            if not strokes:
                raise ValueError(
                    "추출된 윤곽선이 없습니다. 선이 더 뚜렷한 사진을 사용해 보세요."
                )

            self.strokes = strokes
            self.current_stroke = []
            self.last_mouse_point = None
            self.mouse_drawing = False
            self.loaded_image_path = path

            self.redraw_strokes()

            total_points = sum(len(stroke) for stroke in self.strokes)
            self.info_label.config(
                text=(
                    f"사진: {os.path.basename(path)} | "
                    f"선 {len(self.strokes)}개 | 좌표 {total_points}개"
                )
            )
            self.set_status(
                f"사진 좌표 추출 완료: {len(self.strokes)}개 선, {total_points}개 좌표"
            )

        except Exception as error:
            messagebox.showerror("사진 처리 오류", str(error))
            self.set_status(f"사진 처리 오류: {error}")

    @staticmethod
    def read_image_unicode(path):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def image_to_strokes(self, image):
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

            if len(pts) < 2:
                continue

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

        return self.optimize_stroke_order(scaled_strokes)

    @staticmethod
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

    def redraw_strokes(self):
        self.canvas.delete("all")

        for stroke in self.strokes:
            if len(stroke) == 1:
                x, y = stroke[0]
                self.canvas.create_oval(
                    x - 2,
                    y - 2,
                    x + 2,
                    y + 2,
                    fill="black",
                    outline="black",
                )
                continue

            flat = []
            for x, y in stroke:
                flat.extend([x, y])

            self.canvas.create_line(
                *flat,
                width=2,
                fill="black",
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )

    def save_robot_coordinates(self):
        if not self.strokes:
            messagebox.showwarning("좌표 없음", "먼저 사진을 불러오거나 그림을 그리세요.")
            return

        path = filedialog.asksaveasfilename(
            title="로봇 좌표 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"# 작업영역 X={ROBOT_X_MIN}~{ROBOT_X_MAX}, "
                    f"Y={ROBOT_Y_MIN}~{ROBOT_Y_MAX}\n"
                )
                f.write(f"# PEN_UP_Z={PEN_UP_Z}, PEN_DOWN_Z={PEN_DOWN_Z}\n\n")

                for stroke_index, stroke in enumerate(self.strokes, start=1):
                    f.write(f"[STROKE {stroke_index}]\n")
                    for cx, cy in stroke:
                        rx, ry = self.canvas_to_robot(cx, cy)
                        f.write(f"X={rx:.2f}, Y={ry:.2f}\n")
                    f.write("\n")

            self.set_status(f"좌표 저장 완료: {path}")
            messagebox.showinfo("저장 완료", "로봇 좌표 TXT를 저장했습니다.")

        except Exception as error:
            messagebox.showerror("저장 오류", str(error))

    def mouse_press(self, event):
        if self.robot_running:
            return

        self.loaded_image_path = None
        self.mouse_drawing = True
        self.current_stroke = [(event.x, event.y)]
        self.last_mouse_point = (event.x, event.y)

        self.canvas.create_oval(
            event.x - 2,
            event.y - 2,
            event.x + 2,
            event.y + 2,
            fill="black",
            outline="black",
        )

    def mouse_drag(self, event):
        if not self.mouse_drawing:
            return

        if self.last_mouse_point is None:
            self.last_mouse_point = (event.x, event.y)
            return

        x = max(0, min(CANVAS_WIDTH, event.x))
        y = max(0, min(CANVAS_HEIGHT, event.y))
        old_x, old_y = self.last_mouse_point

        if math.hypot(x - old_x, y - old_y) < MIN_PIXEL_DISTANCE:
            return

        self.canvas.create_line(
            old_x,
            old_y,
            x,
            y,
            width=3,
            fill="black",
            capstyle=tk.ROUND,
        )

        self.current_stroke.append((x, y))
        self.last_mouse_point = (x, y)

    def mouse_release(self, event):
        if not self.mouse_drawing:
            return

        self.mouse_drawing = False

        if self.current_stroke:
            x = max(0, min(CANVAS_WIDTH, event.x))
            y = max(0, min(CANVAS_HEIGHT, event.y))
            last_x, last_y = self.current_stroke[-1]

            if math.hypot(x - last_x, y - last_y) >= 1:
                self.current_stroke.append((x, y))

            self.strokes.append(self.current_stroke.copy())

        self.current_stroke = []
        self.last_mouse_point = None

        total_points = sum(len(stroke) for stroke in self.strokes)
        self.info_label.config(
            text=f"마우스 그림 | 선 {len(self.strokes)}개 | 좌표 {total_points}개"
        )
        self.set_status(
            f"{len(self.strokes)}개 선, {total_points}개 좌표 저장"
        )

    def mouse_move(self, event):
        robot_x, robot_y = self.canvas_to_robot(event.x, event.y)
        self.coordinate_label.config(
            text=f"마우스 좌표: X={robot_x:.2f}, Y={robot_y:.2f}"
        )

    def update_live_position(self, x, y, z, drawing=False):
        def _update():
            self.coordinate_label.config(
                text=(f"명령 좌표: X={x:.2f}, Y={y:.2f}, Z={z:.2f}"),
                fg="red",
            )

            cx, cy = self.robot_to_canvas(x, y)

            if drawing and self.live_prev_canvas_point is not None:
                px, py = self.live_prev_canvas_point
                self.canvas.create_line(
                    px, py, cx, cy,
                    width=4,
                    fill="red",
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    tags=("live_progress",),
                )

            self.live_prev_canvas_point = (cx, cy)

            if self.live_marker_id is not None:
                try:
                    self.canvas.delete(self.live_marker_id)
                except Exception:
                    pass

            r = 6
            self.live_marker_id = self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r,
                fill="red",
                outline="white",
                width=2,
                tags=("live_marker",),
            )
            self.canvas.tag_raise(self.live_marker_id)

        self.root.after(0, _update)

    def reset_live_position(self):
        def _reset():
            self.canvas.delete("live_progress")
            self.canvas.delete("live_marker")
            self.live_marker_id = None
            self.live_prev_canvas_point = None
            self.coordinate_label.config(fg="black")
        self.root.after(0, _reset)

    def clear_canvas(self):
        if self.robot_running:
            messagebox.showwarning(
                "실행 중",
                "로봇이 움직이는 동안에는 화면을 지울 수 없습니다.",
            )
            return

        self.canvas.delete("all")
        self.strokes.clear()
        self.current_stroke.clear()
        self.last_mouse_point = None
        self.mouse_drawing = False
        self.loaded_image_path = None

        self.info_label.config(
            text="사진 또는 마우스 그림을 100mm × 75mm 영역에 맞춰 그립니다."
        )
        self.set_status("화면과 저장된 그림을 지웠습니다.")

    def connect_robot(self):
        if self.connected:
            messagebox.showinfo("연결 상태", "이미 로봇이 연결되어 있습니다.")
            return

        try:
            self.set_status(f"{SERIAL_PORT} 연결 중...")

            self.serial1 = serial.Serial(SERIAL_PORT, BAUD_RATE)
            time.sleep(2)

            self.mirobot1 = wlkatapython.Wlkata_UART()
            self.mirobot1.init(self.serial1, ROBOT_ADDRESS)
            self.mirobot1.speed(ROBOT_SPEED)

            self.connected = True
            self.set_status(
                f"연결 완료: {SERIAL_PORT}, 속도 {ROBOT_SPEED}"
            )

        except Exception as error:
            self.connected = False
            self.mirobot1 = None

            if self.serial1 is not None:
                try:
                    self.serial1.close()
                except Exception:
                    pass
                self.serial1 = None

            messagebox.showerror("연결 실패", str(error))
            self.set_status("로봇 연결 실패")

    def start_homing(self):
        if not self.check_robot():
            return
        if self.robot_running:
            self.show_running_warning()
            return

        threading.Thread(target=self.homing_worker, daemon=True).start()

    def homing_worker(self):
        try:
            self.robot_running = True
            self.stop_requested = False
            self.set_status("Homing 실행 중...")
            self.mirobot1.homing()
            self.set_status("Homing 완료")
        except Exception as error:
            self.show_error("Homing 오류", error)
        finally:
            self.robot_running = False

    def start_zero(self):
        if not self.check_robot():
            return
        if self.robot_running:
            self.show_running_warning()
            return

        threading.Thread(target=self.zero_worker, daemon=True).start()

    def zero_worker(self):
        try:
            self.robot_running = True
            self.stop_requested = False
            self.set_status("Zero Position으로 이동 중...")
            self.mirobot1.zero()
            self.set_status("Zero Position 이동 완료")
        except Exception as error:
            self.show_error("Zero Position 오류", error)
        finally:
            self.robot_running = False


    def move_robot(self, x, y, z, linear=True):
        if self.stop_requested:
            return

        x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, float(x)))
        y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, float(y)))

        motion = 1 if linear else 0
        position = 0

        self.mirobot1.writecoordinate(
            motion,
            position,
            round(x, 2),
            round(y, 2),
            round(z, 2),
            ROBOT_RX,
            ROBOT_RY,
            ROBOT_RZ,
        )

        drawing = linear and (float(z) <= PEN_DOWN_Z + 0.01)
        self.update_live_position(x, y, z, drawing=drawing)

    def pen_up(self, x, y):
        self.live_prev_canvas_point = None
        self.move_robot(x=x, y=y, z=PEN_UP_Z, linear=False)

    def pen_down(self, x, y):
        self.move_robot(x=x, y=y, z=PEN_DOWN_Z, linear=True)


    def start_move_test(self):
        if not self.check_robot():
            return
        if self.robot_running:
            self.show_running_warning()
            return

        threading.Thread(target=self.move_test_worker, daemon=True).start()

    def move_test_worker(self):
        try:
            self.robot_running = True
            self.stop_requested = False

            test_x = (ROBOT_X_MIN + ROBOT_X_MAX) / 2.0
            test_y = (ROBOT_Y_MIN + ROBOT_Y_MAX) / 2.0
            test_z = PEN_UP_Z

            self.set_status("작업 영역 중앙으로 이동 중...")
            self.move_robot(test_x, test_y, test_z, linear=False)

            if not self.stop_requested:
                self.set_status(
                    f"이동 완료: X={test_x:.2f}, Y={test_y:.2f}, Z={test_z:.2f}"
                )
        except Exception as error:
            self.show_error("좌표 이동 테스트 오류", error)
        finally:
            self.robot_running = False


    def start_robot_drawing(self):
        if not self.check_robot():
            return
        if self.robot_running:
            self.show_running_warning()
            return
        if not self.strokes:
            messagebox.showwarning(
                "그림 없음",
                "먼저 사진을 불러오거나 흰색 영역에 마우스로 그림을 그리세요.",
            )
            return

        threading.Thread(target=self.drawing_worker, daemon=True).start()

    def drawing_worker(self):
        try:
            self.robot_running = True
            self.stop_requested = False
            self.reset_live_position()

            strokes_copy = [stroke.copy() for stroke in self.strokes]
            total_strokes = len(strokes_copy)
            total_points = sum(len(stroke) for stroke in strokes_copy)
            completed_points = 0

            for stroke_index, stroke in enumerate(strokes_copy, start=1):
                if self.stop_requested:
                    break
                if not stroke:
                    continue

                first_canvas_x, first_canvas_y = stroke[0]
                first_robot_x, first_robot_y = self.canvas_to_robot(
                    first_canvas_x,
                    first_canvas_y,
                )

                self.set_status(
                    f"{stroke_index}/{total_strokes}번째 선 시작점으로 이동 중..."
                )

                self.pen_up(first_robot_x, first_robot_y)
                if self.stop_requested:
                    break

                self.pen_down(first_robot_x, first_robot_y)
                if self.stop_requested:
                    break

                for canvas_x, canvas_y in stroke[1:]:
                    if self.stop_requested:
                        break

                    robot_x, robot_y = self.canvas_to_robot(canvas_x, canvas_y)
                    self.move_robot(
                        x=robot_x,
                        y=robot_y,
                        z=PEN_DOWN_Z,
                        linear=True,
                    )

                    completed_points += 1
                    self.set_status(
                        f"그리는 중: {completed_points}/{total_points} | "
                        f"선 {stroke_index}/{total_strokes}"
                    )

                last_canvas_x, last_canvas_y = stroke[-1]
                last_robot_x, last_robot_y = self.canvas_to_robot(
                    last_canvas_x,
                    last_canvas_y,
                )
                self.pen_up(last_robot_x, last_robot_y)

            if self.stop_requested:
                self.set_status("그리기 중지됨")
            else:
                self.set_status("그림 그리기 완료")

        except Exception as error:
            self.show_error("그림 그리기 오류", error)
        finally:
            self.robot_running = False
            self.reset_live_position()


    def move_robot_measurement(self, x, y, z, log_file, start_time, linear=True):
        if self.stop_requested:
            return

        x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, float(x)))
        y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, float(y)))

        motion = 1 if linear else 0
        position = 0

        self.mirobot1.writecoordinate(
            motion,
            position,
            round(x, 2),
            round(y, 2),
            round(z, 2),
            ROBOT_RX,
            ROBOT_RY,
            ROBOT_RZ,
        )

        time.sleep(MEASUREMENT_INTERVAL)

        current_x = float(self.mirobot1.getcoordinate(1))
        current_y = float(self.mirobot1.getcoordinate(2))
        current_z = float(self.mirobot1.getcoordinate(3))

        error_x = x - current_x
        error_y = y - current_y
        error_z = z - current_z
        elapsed = time.monotonic() - start_time

        log_file.write(
            f"{elapsed:.2f}\t{x:.2f}\t{y:.2f}\t{z:.2f}\t"
            f"{current_x:.2f}\t{current_y:.2f}\t{current_z:.2f}\t"
            f"{error_x:.2f}\t{error_y:.2f}\t{error_z:.2f}\n"
        )
        log_file.flush()

        drawing = linear and (float(z) <= PEN_DOWN_Z + 0.01)
        self.update_measurement_position(current_x, current_y, current_z, drawing=drawing)

    def update_measurement_position(self, x, y, z, drawing=False):
        def _update():
            self.coordinate_label.config(
                text=f"현재 좌표: X={x:.2f}, Y={y:.2f}, Z={z:.2f}",
                fg="blue",
            )

            cx, cy = self.robot_to_canvas(x, y)

            if drawing and self.live_prev_canvas_point is not None:
                px, py = self.live_prev_canvas_point
                self.canvas.create_line(
                    px, py, cx, cy,
                    width=4,
                    fill="blue",
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    tags=("live_progress",),
                )

            self.live_prev_canvas_point = (cx, cy)

            if self.live_marker_id is not None:
                try:
                    self.canvas.delete(self.live_marker_id)
                except Exception:
                    pass

            r = 6
            self.live_marker_id = self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r,
                fill="blue",
                outline="white",
                width=2,
                tags=("live_marker",),
            )
            self.canvas.tag_raise(self.live_marker_id)

        self.root.after(0, _update)

    def start_measurement_drawing(self):
        if not self.check_robot():
            return
        if self.robot_running:
            self.show_running_warning()
            return
        if not self.strokes:
            messagebox.showwarning(
                "그림 없음",
                "먼저 사진을 불러오거나 흰색 영역에 마우스로 그림을 그리세요.",
            )
            return

        threading.Thread(target=self.measurement_drawing_worker, daemon=True).start()

    def measurement_drawing_worker(self):
        log_path = None
        try:
            self.robot_running = True
            self.stop_requested = False
            self.reset_live_position()

            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = time.strftime("C:/Users/SAMSUNG/OneDrive/바탕 화면/로봇/참고자료/measure.txt")
            log_path = os.path.join(base_dir, filename)

            strokes_copy = [stroke.copy() for stroke in self.strokes]
            total_strokes = len(strokes_copy)
            total_points = sum(len(stroke) for stroke in strokes_copy)
            completed_points = 0
            start_time = time.monotonic()

            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(
                    "time_s\ttarget_X\ttarget_Y\ttarget_Z\t"
                    "current_X\tcurrent_Y\tcurrent_Z\t"
                    "error_X\terror_Y\terror_Z\n"
                )

                for stroke_index, stroke in enumerate(strokes_copy, start=1):
                    if self.stop_requested:
                        break
                    if not stroke:
                        continue

                    first_canvas_x, first_canvas_y = stroke[0]
                    first_robot_x, first_robot_y = self.canvas_to_robot(
                        first_canvas_x,
                        first_canvas_y,
                    )

                    self.set_status(
                        f"좌표 측정 중: {stroke_index}/{total_strokes}번째 선 시작점 이동"
                    )

                    self.live_prev_canvas_point = None
                    self.move_robot_measurement(
                        first_robot_x, first_robot_y, PEN_UP_Z,
                        log_file, start_time, linear=False
                    )
                    if self.stop_requested:
                        break

                    self.move_robot_measurement(
                        first_robot_x, first_robot_y, PEN_DOWN_Z,
                        log_file, start_time, linear=True
                    )
                    if self.stop_requested:
                        break

                    for canvas_x, canvas_y in stroke[1:]:
                        if self.stop_requested:
                            break

                        robot_x, robot_y = self.canvas_to_robot(canvas_x, canvas_y)
                        self.move_robot_measurement(
                            robot_x, robot_y, PEN_DOWN_Z,
                            log_file, start_time, linear=True
                        )

                        completed_points += 1
                        self.set_status(
                            f"좌표 측정 중: {completed_points}/{total_points} | "
                            f"선 {stroke_index}/{total_strokes}"
                        )

                    last_canvas_x, last_canvas_y = stroke[-1]
                    last_robot_x, last_robot_y = self.canvas_to_robot(
                        last_canvas_x,
                        last_canvas_y,
                    )
                    self.move_robot_measurement(
                        last_robot_x, last_robot_y, PEN_UP_Z,
                        log_file, start_time, linear=False
                    )

            if self.stop_requested:
                self.set_status(f"좌표 측정 중지됨")
            else:
                self.set_status(f"좌표 측정 완료")

        except Exception as error:
            self.show_error("좌표 측정 오류", error)
        finally:
            self.robot_running = False
            self.reset_live_position()


    def stop_robot(self):
        self.stop_requested = True

        if self.mirobot1 is not None:
            try:
                self.mirobot1.cancellation()
            except Exception as error:
                print("정지 명령 오류:", error)

        self.reset_live_position()
        self.set_status("정지 명령을 전송했습니다.")

    def check_robot(self):
        if not self.connected or self.mirobot1 is None:
            messagebox.showwarning(
                "연결 필요",
                "먼저 로봇 연결 버튼을 누르세요.",
            )
            return False
        return True

    def show_running_warning(self):
        messagebox.showwarning(
            "실행 중",
            "로봇이 이미 움직이고 있습니다.",
        )

    def set_status(self, text):
        self.root.after(
            0,
            lambda: self.status_label.config(text=f"상태: {text}"),
        )

    def show_error(self, title, error):
        error_text = str(error)
        self.root.after(
            0,
            lambda: messagebox.showerror(title, error_text),
        )
        self.set_status(f"오류: {error_text}")

    def close_program(self):
        self.stop_requested = True

        if self.robot_running and self.mirobot1 is not None:
            try:
                self.mirobot1.cancellation()
            except Exception:
                pass

        if self.serial1 is not None:
            try:
                self.serial1.close()
            except Exception:
                pass

        self.root.destroy()

    

    def mode_live_capture(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
        except RuntimeError:
                    messagebox.showwarning(
                        "카메라 연결 오류",
                        "RealSense 카메라가 연결되어 있지 않습니다.\n"
                        "카메라 연결 상태를 확인해주세요."
                    )
                    self.set_status("RealSense 카메라 연결 실패")
                    return
        time.sleep(1)

        for _ in range(5): frames = pipeline.wait_for_frames()
    
        color_frame = frames.get_color_frame()
        if color_frame:
            test_image = np.asanyarray(color_frame.get_data())
            h, w = test_image.shape[:2]
            if w != 640 or h != 480: print("설정된 해상도가 640x480과 다릅니다")


        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue
            
                color_image = np.asanyarray(color_frame.get_data())
                paths = extract_paths_from_image(color_image)
                display_img = color_image.copy()
            
                for path in paths:
                    pts = np.array(path, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                cv2.putText(display_img, "Press ENTER to Save, ESC to Exit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Live Capture Mode", display_img)

                key = cv2.waitKey(1)
                if key == 13:
                    if not paths:
                        print("추출된 윤곽선이 없습니다.")
                        continue

                    strokes = []

                    for path in paths:
                        stroke = []

                        for x, y in path:
                            stroke.append((x, y))

                        if len(stroke) >= 2:
                            strokes.append(stroke)

                    self.strokes = strokes


                    self.redraw_strokes()

                    total_points = sum(
                        len(stroke) for stroke in self.strokes
                    )

                    self.info_label.config(
                        text=(
                            f"YOLO 세그멘테이션 | "
                            f"선 {len(self.strokes)}개 | "
                            f"좌표 {total_points}개"
                        )
                    )

                    self.set_status(
                        f"YOLO 윤곽선 저장 완료: "
                        f"{len(self.strokes)}개 선, "
                        f"{total_points}개 좌표"
                    )

                    cv2.imwrite(SAVE_FILENAME, color_image)
                    break

                elif key == 27: break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

def extract_paths_from_image(image, conf=0.1):
    model = YOLO("yolov8n-seg.pt")
    results = model(image, conf=conf, verbose=False)

    all_paths = []

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            mask = (mask * 255).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
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
    

def main():
    root = tk.Tk()
    DrawingRobotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
