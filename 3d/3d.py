import tkinter as tk
from tkinter import messagebox
import serial
import wlkatapython

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

#py 3d\3d.py

#가로 : 60mm 
#세로 : 40mm
#높이 : 135mm

START_X = 258.6
START_Y = -65.0

PEN_UP_Z = 20.0
PEN_DOWN_Z = 15.0

LINE_GAP = 30.0


class LineDrawGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Line Drawing")
        self.root.geometry("950x600")

        try:
            self.serial = serial.Serial("COM7", 115200)

            self.robot = wlkatapython.Wlkata_UART()
            self.robot.init(self.serial, -1)
            self.robot.speed(2000)

            print("로봇 연결 성공")

        except Exception as e:
            messagebox.showerror("연결 오류", str(e))
            self.serial = None
            self.robot = None

        left_frame = tk.Frame(root)
        left_frame.pack(
            side=tk.LEFT,
            padx=20,
            pady=20
        )

        right_frame = tk.Frame(root)
        right_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=20
        )

        tk.Label(
            left_frame,
            text="상자 크기 입력 (cm)",
            font=("Arial", 14)
        ).pack(pady=15)

        labels = [
            "X 길이:",
            "Y 길이:",
            "Z 길이:"
        ]

        self.entries = []

        for label_text in labels:
            frame = tk.Frame(left_frame)
            frame.pack(pady=5)

            tk.Label(
                frame,
                text=label_text,
                width=10
            ).pack(side=tk.LEFT)

            entry = tk.Entry(
                frame,
                width=10
            )
            entry.pack(side=tk.LEFT)

            self.entries.append(entry)

        tk.Button(
            left_frame,
            text="그리기 시작",
            command=self.draw_lines,
            width=15
        ).pack(pady=20)

        tk.Button(
            left_frame,
            text="3축 방향 테스트",
            command=self.draw_xyz_lines,
            width=15
        ).pack(pady=5)

        tk.Button(
            left_frame,
            text="지정 좌표 이동",
            command=self.move_fixed_coordinates,
            width=15
        ).pack(pady=5)

        self.figure = plt.Figure(
            figsize=(6, 5),
            dpi=100
        )

        self.ax = self.figure.add_subplot(
            111,
            projection="3d"
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=right_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        self.setup_3d()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    def setup_3d(self):
        self.ax.clear()

        self.ax.set_title("3D Box Model")

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")

        self.ax.set_xlim(0, 200)
        self.ax.set_ylim(0, 200)
        self.ax.set_zlim(0, 200)

        self.canvas.draw()

    def update_3d(self, lengths):
        x = lengths[0]
        y = lengths[1]
        z = lengths[2]

        self.ax.clear()

        vertices = [
            [0, 0, 0],
            [x, 0, 0],
            [x, y, 0],
            [0, y, 0],
            [0, 0, z],
            [x, 0, z],
            [x, y, z],
            [0, y, z]
        ]

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]

        box = Poly3DCollection(
            faces,
            alpha=0.25,
            edgecolor="black"
        )

        self.ax.add_collection3d(box)

        max_length = max(x, y, z)

        self.ax.set_xlim(0, max_length)
        self.ax.set_ylim(0, max_length)
        self.ax.set_zlim(0, max_length)

        self.ax.set_xlabel(
            f"X = {x / 10:.1f} cm"
        )

        self.ax.set_ylabel(
            f"Y = {y / 10:.1f} cm"
        )

        self.ax.set_zlabel(
            f"Z = {z / 10:.1f} cm"
        )

        self.ax.set_title(
            f"3D Box  "
            f"{x / 10:.1f} × "
            f"{y / 10:.1f} × "
            f"{z / 10:.1f} cm"
        )

        self.ax.set_box_aspect(
            (x, y, z)
        )

        self.canvas.draw()

    def move_linear_4parts(self, x, start_y, end_y, z):
        for i in range(1, 11):
            t = i / 10.0
            y = start_y + (end_y - start_y) * t

            self.robot.writecoordinate(
                0,
                0,
                x,
                y,
                z,
                0,
                0,
                0
            )
            time.sleep(0.05)

    def move_linear_xyz(self, start_x, start_y, start_z, end_x, end_y, end_z):
        for i in range(1, 11):
            t = i / 10.0
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            z = start_z + (end_z - start_z) * t

            self.robot.writecoordinate(
                1, 0, x, y, z, 0, 0, 0
            )
            time.sleep(0.05)

    def draw_xyz_lines(self):
        try:
            if self.robot is None:
                messagebox.showerror("로봇 오류", "로봇이 연결되지 않았습니다.")
                return

            lengths = []
            for entry in self.entries:
                length_cm = float(entry.get())
                if length_cm <= 0 or length_cm > 20:
                    messagebox.showerror(
                        "입력 오류",
                        "모든 길이는 0보다 크고 20cm 이하로 입력해주세요."
                    )
                    return
                lengths.append(length_cm * 10)


            self.update_3d(lengths)

            length_y = lengths[0]
            length_x = lengths[1]
            length_z = lengths[2]

            sx = START_X
            sy = START_Y
            sz = PEN_DOWN_Z


            self.robot.writecoordinate(1, 0, sx, sy, PEN_UP_Z, 0, 0, 0)
            self.robot.writecoordinate(1, 0, sx, sy, sz, 0, 0, 0)
            self.move_linear_xyz(sx, sy, sz, sx, sy + length_y, sz)
            self.robot.writecoordinate(1, 0, sx, sy + length_y, PEN_UP_Z, 0, 0, 0)
            time.sleep(0.5)

            self.robot.writecoordinate(1, 0, sx, sy, PEN_UP_Z, 0, 0, 0)
            self.robot.writecoordinate(1, 0, sx, sy, sz, 0, 0, 0)
            self.move_linear_xyz(sx, sy, sz, sx + length_x, sy, sz)
            self.robot.writecoordinate(1, 0, sx + length_x, sy, PEN_UP_Z, 0, 0, 0)
            time.sleep(0.5)

    
            self.robot.writecoordinate(1, 0, sx, sy, sz, 0, 0, 0)
            self.move_linear_xyz(sx, sy, sz, sx, sy, sz + length_z)


        except ValueError:
            messagebox.showerror(
                "입력 오류",
                "3개의 길이를 모두 숫자로 입력해주세요."
            )
        except Exception as e:
            messagebox.showerror("로봇 오류", str(e))

    def move_fixed_coordinates(self):
        try:
            if self.robot is None:
                messagebox.showerror("로봇 오류", "로봇이 연결되지 않았습니다.")
                return

            points = [
                (223.0, 55.0, 15.0),
                (223.0, 55.0, 75.0),   # 기준점
                (223.0, -80.0, 75.0),
                (223.0, 55.0, 15.0),
                (263.0, 55.0, 15.0),
            ]

            x, y, z = points[0]
            print(f"[지정 좌표 이동] 1 -> X={x}, Y={y}, Z={z}")
            self.robot.writecoordinate(1, 0, x, y, z, 0, 0, 0)
            time.sleep(1.0)

            for i in range(1, len(points)):
                sx, sy, sz = points[i - 1]
                ex, ey, ez = points[i]
                print(f"[지정 좌표 이동] {i + 1} -> X={ex}, Y={ey}, Z={ez}")
                self.move_linear_xyz(sx, sy, sz, ex, ey, ez)
                time.sleep(0.5)

            messagebox.showinfo("완료", "지정 좌표 이동을 완료했습니다.")

        except Exception as e:
            messagebox.showerror("로봇 오류", str(e))

    def draw_lines(self):
        try:
            if self.robot is None:
                messagebox.showerror(
                    "로봇 오류",
                    "로봇이 연결되지 않았습니다."
                )
                return

            lengths = []

            for entry in self.entries:
                length_cm = float(
                    entry.get()
                )

                if length_cm <= 0 or length_cm > 20:
                    messagebox.showerror(
                        "입력 오류",
                        "모든 길이는 0보다 크고 20cm 이하로 입력해주세요."
                    )
                    return

                lengths.append(
                    length_cm * 10
                )

            self.update_3d(lengths)

            for i, length_mm in enumerate(lengths):
                line_x = START_X + (i * LINE_GAP)
                line_y = START_Y
                end_y = line_y + length_mm

                self.robot.writecoordinate(
                    0,
                    0,
                    line_x,
                    line_y,
                    PEN_UP_Z,
                    0,
                    0,
                    0
                )

                self.robot.writecoordinate(
                    0,
                    0,
                    line_x,
                    line_y,
                    PEN_DOWN_Z,
                    0,
                    0,
                    0
                )

                self.move_linear_4parts(
                    line_x,
                    line_y,
                    end_y,
                    PEN_DOWN_Z
                )

                self.robot.writecoordinate(
                    0,
                    0,
                    line_x,
                    end_y,
                    PEN_UP_Z,
                    0,
                    0,
                    0
                )

            print(f"입력 길이: {length_cm} cm")
            print(f"변환 길이: {length_mm} mm")
            print(f"시작 Y: {line_y}")
            print(f"종료 Y: {end_y}")


            self.robot.zero()

        except ValueError:
            messagebox.showerror(
                "입력 오류",
                "3개의 길이를 모두 숫자로 입력해주세요."
            )

        except Exception as e:
            messagebox.showerror(
                "로봇 오류",
                str(e)
            )

    def close(self):
        try:
            if self.robot is not None:
                self.robot.cancellation()

            if self.serial is not None:
                self.serial.close()

        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LineDrawGUI(root)
    root.mainloop()