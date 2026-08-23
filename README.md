YOLO 결과
│
├── Mask 1
│    ├── Contour 1 → Path 1
│    └── Contour 2 → Path 2
│
├── Mask 2
│    └── Contour 1 → Path 3
│
└── Mask 3
     └── Contour 1 → Path 4


RealSense Mirobot Vision & Drawing Control System

Intel RealSense 카메라와 WLKATA Mirobot 로봇 암을 이용하여 2D/3D 경로 생성, 로봇 좌표 변환 및 로봇팔 제어를 수행하는 프로젝트입니다.

GUI에서 직접 입력한 드로잉 경로 또는 이미지에서 추출한 윤곽선을 로봇 좌표계로 변환하여 Mirobot이 해당 경로를 따라 움직이도록 구현하였으며, Intel RealSense Depth 정보를 이용한 3D 공간 좌표 획득 및 3D 경로 추종 기능을 확장하고 있습니다.

"Python" (https://img.shields.io/badge/Python-3.x-blue.svg)
"OpenCV" (https://img.shields.io/badge/OpenCV-4.x-green.svg)
"RealSense" (https://img.shields.io/badge/Intel-RealSense-blue.svg)
"Robot" (https://img.shields.io/badge/Robot-WLKATA%20Mirobot-orange.svg)

---

Features

- Mirobot 로봇 암 제어
  
  - Serial 통신을 이용한 Mirobot 연결
  - Homing 및 Zero Position 이동
  - XYZ 직교좌표 기반 이동
  - 선형 경로 추종
  - 로봇 이동 및 현재 좌표 확인

- GUI 기반 2D Drawing
  
  - Tkinter 기반 드로잉 인터페이스
  - 마우스 드래그를 이용한 경로 입력
  - Canvas Pixel → Robot XYZ 좌표 변환
  - Pen Up / Pen Down을 이용한 드로잉
  - 작업영역 제한 및 좌표 표시

- Image → Robot Path
  
  - OpenCV 기반 이미지 전처리
  - Canny Edge Detection
  - Contour 추출
  - Contour 단순화
  - 이미지 픽셀 좌표를 로봇 좌표로 변환
  - 추출된 경로를 이용한 자동 드로잉

- RealSense Depth Camera
  
  - Intel RealSense D435/D435i 지원
  - RGB 및 Depth 영상 획득
  - Depth 값을 이용한 실제 3D 좌표 계산
  - Pixel "(u, v)" → Camera Coordinate "(X, Y, Z)" 변환

- TCP / Tool Offset
  
  - Mirobot Tool Coordinate System 사용
  - End-effector에 장착된 도구 길이를 고려한 TCP 보정
  - X/Y/Z Tool Offset 설정
  - 실제 도구 끝점을 기준으로 한 로봇 좌표 제어

- 3D Robot Control
  
  - XYZ 3차원 목표 좌표 입력
  - Mirobot Base Coordinate System 기준 이동
  - RealSense 3D 좌표와 Robot 좌표계 연결을 위한 좌표변환 구조

---

System Overview

전체 시스템의 기본 처리 과정은 다음과 같습니다.

[2D Drawing / Image / RealSense Camera]
                │
                ▼
        Pixel Coordinate
             (u, v)
                │
                ▼
       Image / Depth Processing
                │
                ▼
      2D or 3D Path Generation
                │
                ▼
      Coordinate Transformation
                │
                ▼
       Robot Base Coordinate
           (X, Y, Z)
                │
                ▼
         TCP / Tool Offset
                │
                ▼
         WLKATA Mirobot
                │
                ▼
          Path Tracking

---

Coordinate Systems

본 프로젝트에서는 크게 세 가지 좌표계를 사용합니다.

1. Canvas / Image Coordinate

GUI 또는 카메라 이미지의 픽셀 좌표입니다.

(u, v)

---

2. Camera Coordinate

RealSense Depth 정보를 이용하여 픽셀 좌표와 Depth 값을 실제 카메라 기준 3차원 좌표로 변환합니다.

Pixel Coordinate
(u, v)

      ↓

Depth
Z

      ↓

Deprojection

      ↓

Camera Coordinate

RealSense의 카메라 내부 파라미터를 이용하여 픽셀 위치를 실제 3차원 위치로 변환합니다.

---

3. Robot Base Coordinate

Mirobot의 이동 명령은 로봇의 Base Coordinate System을 기준으로 처리합니다.

GUI에서 생성된 좌표 또는 카메라에서 획득한 좌표를 최종적으로 Robot Base Coordinate로 변환한 뒤 로봇에 전달합니다.

---

2D Drawing Coordinate Mapping

GUI에서 입력된 픽셀 좌표를 실제 로봇 작업영역으로 변환합니다.

예를 들어 GUI의 Canvas 좌표를 다음과 같이 로봇 좌표로 변환할 수 있습니다.

robot_x = ROBOT_X_MIN + (canvas_y / CANVAS_HEIGHT) * (
    ROBOT_X_MAX - ROBOT_X_MIN
)

robot_y = ROBOT_Y_MIN + (canvas_x / CANVAS_WIDTH) * (
    ROBOT_Y_MAX - ROBOT_Y_MIN
)

따라서 처리 과정은 다음과 같습니다.

Mouse Drawing
     │
     ▼
Canvas Pixel
(x, y)
     │
     ▼
Coordinate Scaling
     │
     ▼
Robot Coordinate
(X, Y)
     │
     ▼
Mirobot Drawing

---

Image Path Extraction

이미지에서 로봇이 따라갈 수 있는 경로를 생성하기 위해 OpenCV를 이용합니다.

현재 기본 처리 과정은 다음과 같습니다.

Input Image
     │
     ▼
Grayscale
     │
     ▼
Gaussian Blur
     │
     ▼
Canny Edge Detection
     │
     ▼
Morphological Closing
     │
     ▼
Contour Detection
     │
     ▼
Contour Simplification
     │
     ▼
Robot Path

Contour에서 추출된 픽셀 좌표는 GUI 드로잉과 동일한 좌표 변환 과정을 통해 로봇 좌표로 변환됩니다.

---

RealSense 3D Coordinate

RealSense 카메라에서는 RGB 이미지의 픽셀 좌표와 Depth 정보를 이용하여 실제 3D 위치를 계산합니다.

RGB Pixel
(u, v)

Depth Image
   │
   ▼

Depth(u, v)

   │
   ▼

RealSense Intrinsics

   │
   ▼

rs2_deproject_pixel_to_point()

   │
   ▼

Camera 3D Point
(Xc, Yc, Zc)

이를 통해 이미지상의 한 점을 카메라 좌표계 기준 실제 3차원 위치로 표현할 수 있습니다.

---

Camera → Robot Coordinate Transformation

RealSense에서 계산된 좌표는 Camera Coordinate System 기준이므로 Mirobot에서 바로 사용할 수 없습니다.

따라서 다음 좌표 변환 과정이 필요합니다.

Camera Coordinate
Pc = [Xc Yc Zc]

        │
        ▼

Camera → Robot Transformation

        │
        ▼

Robot Base Coordinate
Pr = [Xr Yr Zr]

일반적인 변환식은 다음과 같습니다.

Pr = R · Pc + t

여기서

R = Rotation Matrix
t = Translation Vector

이며, 카메라와 로봇 Base 사이의 위치 및 방향 관계를 이용하여 계산합니다.

---

TCP / Tool Offset

Mirobot의 기본 Tool Coordinate Origin은 6축 끝단의 Flange 기준으로 설정됩니다.

펜이나 다른 End-effector를 장착하면 실제 작업점은 Flange 위치와 달라지므로 Tool Offset을 설정해야 합니다.

사용하는 Tool Offset 명령은 다음과 같습니다.

$46 = Tool X Offset
$47 = Tool Y Offset
$48 = Tool Z Offset

Python에서는 Transparent Command를 이용하여 설정할 수 있습니다.

def custom_tool_offset(mirobot, x, y, z):
    msg = (
        '$46=' + str(x) + '\n'
        '$47=' + str(y) + '\n'
        '$48=' + str(z)
    )

    mirobot.sendMsg(msg)

예를 들어 다음과 같이 설정할 수 있습니다.

custom_tool_offset(mirobot, 0, 0, 120)

Tool Offset을 설정하면 로봇의 좌표 계산에서 장착된 도구의 위치를 고려할 수 있으므로 실제 Tool Center Point(TCP) 를 기준으로 경로를 제어할 수 있습니다.

---

Robot Control

Mirobot은 Python API의 "writecoordinate()"를 이용하여 XYZ 좌표를 전달합니다.

mirobot.writecoordinate(
    motion,
    position,
    x,
    y,
    z,
    rx,
    ry,
    rz
)

본 프로젝트에서는 주로 Base Coordinate System 기준의 절대 XYZ 좌표를 사용합니다.

Target Position

X
Y
Z
RX
RY
RZ

TCP Offset이 설정되어 있는 경우 Tool의 실제 작업점을 고려하여 로봇의 자세가 계산됩니다.

---

Robot Workspace

현재 실험에서는 로봇의 안전한 이동 범위 안에서 제한된 작업영역을 설정하여 사용합니다.

예시:

ROBOT_X_MIN = 170.0
ROBOT_X_MAX = 245.0

ROBOT_Y_MIN = -100.0
ROBOT_Y_MAX = 0.0

따라서 현재 2D Drawing 작업영역은 약

75 mm × 100 mm

범위로 제한하여 사용합니다.

---

Requirements

Hardware

- WLKATA Mirobot
- Intel RealSense D435 / D435i
- USB 3.0 지원 PC
- Drawing Tool / Pen Holder

Software

- Python 3.x
- Intel RealSense SDK 2.0
- OpenCV
- pyrealsense2
- NumPy
- pySerial
- WLKATA Python API
- Tkinter

---

Installation

1. 저장소 클론

git clone <repository-url>
cd <repository-name>

2. 가상환경 생성

python -m venv .venv

Windows:

.venv\Scripts\activate

Linux:

source .venv/bin/activate

3. Python 패키지 설치

pip install -r requirements.txt

4. Intel RealSense SDK 설치

Intel RealSense SDK 2.0 및 "pyrealsense2"를 설치합니다.

---

Usage

1. Mirobot 연결

Mirobot을 PC와 연결하고 COM Port를 설정합니다.

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

2. Robot Homing

프로그램 실행 후 Homing을 수행합니다.

mirobot.homing()

3. Tool Offset 설정

장착한 End-effector의 크기에 맞게 TCP Offset을 설정합니다.

custom_tool_offset(
    mirobot,
    0,
    0,
    120
)

4. Drawing / Image Path 실행

GUI에서 직접 경로를 입력하거나 이미지에서 경로를 추출한 후 로봇을 실행합니다.

5. RealSense 3D Coordinate 획득

RealSense Depth 정보를 이용하여 대상의 실제 3D 좌표를 계산합니다.

---

Project Structure

project/
│
├── main.py
│
├── config.py
│
├── requirements.txt
│
├── README.md
│
├── robot/
│   └── robot_controller.py
│
├── vision/
│   ├── image_processing.py
│   ├── segmentation.py
│   └── realsense.py
│
├── coordinate/
│   ├── pixel_to_robot.py
│   ├── camera_to_robot.py
│   └── transformation.py
│
├── gui/
│   └── drawing_gui.py
│
└── logs/
    └── robot_measurement.csv

---

Current Progress

현재까지 구현 및 확인한 주요 기능은 다음과 같습니다.

- [x] Mirobot Serial 연결
- [x] Homing / Zero Position 제어
- [x] XYZ 좌표 기반 로봇 이동
- [x] GUI 기반 마우스 드로잉
- [x] Canvas Pixel → Robot Coordinate 변환
- [x] Pen Up / Pen Down 드로잉
- [x] OpenCV 기반 이미지 윤곽선 추출
- [x] Contour → Robot Path 변환
- [x] Mirobot Tool Offset 설정
- [x] TCP Offset 동작 확인
- [x] RealSense RGB / Depth 획득
- [x] RealSense Pixel → 3D Camera Coordinate 계산
- [ ] Camera Coordinate → Robot Base Coordinate 변환
- [ ] Camera-Robot Calibration
- [ ] RealSense 기반 실제 물체 3D 경로 생성
- [ ] 3D Path Tracking


---

Future Work

향후 프로젝트에서는 다음 기능을 구현할 예정입니다.

1. Camera-Robot Calibration

RealSense 카메라 좌표계와 Mirobot Base 좌표계 사이의 변환 관계를 계산합니다.

Camera XYZ
     ↓
Rotation + Translation
     ↓
Robot XYZ

2. RealSense 3D Path Generation

RGB/Depth 영상에서 대상의 위치 및 형상을 검출하고 실제 3D 경로를 생성합니다.

3. 3D Path Tracking

생성된 XYZ 좌표를 Mirobot에 순차적으로 전달하여 공간상의 3차원 경로를 따라 이동하도록 구현합니다.

4. Vision-Robot Integration

최종적으로 다음과 같은 전체 파이프라인 구축을 목표로 합니다.

RealSense RGB-D
      │
      ▼
Object / Shape Detection
      │
      ▼
Pixel Coordinate
      │
      ▼
3D Camera Coordinate
      │
      ▼
Camera → Robot Transformation
      │
      ▼
3D Robot Path
      │
      ▼
Mirobot Motion

---

Troubleshooting

Mirobot 연결 실패

- COM Port 확인
- Baud Rate 확인
- 다른 프로그램에서 Serial Port를 사용하고 있는지 확인
- Mirobot 전원 및 USB 연결 상태 확인

RealSense가 감지되지 않음

- USB 3.0 포트 사용 확인
- Intel RealSense SDK 설치 확인
- RealSense Viewer에서 카메라 동작 확인

로봇 좌표와 실제 Tool 위치가 다름

- Tool Offset 설정 확인
- End-effector 길이 측정
- "$46", "$47", "$48" 설정값 확인
- RX / RY / RZ 자세 확인

이미지와 실제 Drawing 결과가 다름

- Canvas → Robot 좌표 변환 범위 확인
- 작업영역 X/Y 최소·최대값 확인
- Contour 좌표 간격 확인
- Pen 압력 및 작업면과의 마찰 확인

---

Development Goal

본 프로젝트의 최종 목표는 카메라를 통해 작업 대상의 공간 정보를 인식하고, 해당 정보를 로봇 좌표계로 변환하여 별도의 수동 티칭 없이 로봇이 자동으로 경로를 생성하고 추종할 수 있도록 하는 것입니다.

Perception
    ↓
3D Coordinate
    ↓
Path Generation
    ↓
Coordinate Transformation
    ↓
Robot Control

이를 통해 Vision → 3D Path Generation → Robot Control로 이어지는 통합 로봇 제어 시스템을 구축하는 것을 목표로 합니다.