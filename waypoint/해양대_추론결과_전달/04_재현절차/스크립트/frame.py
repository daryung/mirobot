# -*- coding: utf-8 -*-
"""
전달용 좌표계 확인 — 우리 PLY 프레임과 RealSense 디프로젝션 프레임이 같은가

연구원 쪽 파이프라인
  rs2_deproject_pixel_to_point(내부파라미터, (u,v), depth) -> 카메라 좌표 (X, Y, Z), 단위 m
  RealSense 카메라 좌표계는 X 오른쪽, Y 아래, Z 앞(카메라에서 멀어지는 방향)이다.
  따라서 Z 는 항상 양수여야 한다. 깊이니까.

우리가 받은 PLY
  헤더에 "pointcloud saved from Realsense Viewer" 라고 적혀 있으나
  좌표가 그대로인지 확인이 필요하다. z 가 음수면 다른 규약이다.

이 확인을 건너뛰면 로봇이 거울상 위치로 움직인다.
"""
import os, sys, glob, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DIRS = [('해양대 66조건', '/kmou/06_용접선_3mm_결과_66조건/PLY/ascii'),
        ('수동 라벨', '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable')]


def read_head(path, nrow=4000):
    with open(path, 'r', encoding='ascii', errors='ignore') as f:
        n = 0
        while True:
            line = f.readline()
            if not line: raise ValueError('헤더 끝 없음')
            s = line.strip()
            if s.startswith('element vertex'): n = int(s.split()[2])
            if s == 'end_header': break
        rows = []
        for _ in range(min(nrow, n)):
            l = f.readline()
            if not l: break
            rows.append([float(x) for x in l.split()[:3]])
    return np.array(rows)


for nm, d in DIRS:
    fs = sorted(glob.glob(os.path.join(d, '**', '*.ply'), recursive=True))
    fs = [f for f in fs if 'ascii' in f or 'ascii' in os.path.basename(f)]
    if not fs: continue
    print(f"=== {nm} ({len(fs)}개 중 3개 표본) ===")
    for f in fs[:3]:
        P = read_head(f)
        print(f"  {os.path.basename(f)[:44]:<44s}")
        for i, ax in enumerate('xyz'):
            print(f"    {ax}  {P[:,i].min():>9.4f} ~ {P[:,i].max():>9.4f} m   "
                  f"부호 {'전부 음수' if P[:,i].max()<0 else ('전부 양수' if P[:,i].min()>0 else '섞임')}")
    print()

print("=== 판정 ===")
f = sorted(glob.glob(os.path.join(DIRS[0][1], '*.ply')))[0]
P = read_head(f, 200000)
z = P[:, 2]
print(f"  z 범위 {z.min():.4f} ~ {z.max():.4f} m")
if z.max() < 0:
    print("  z 가 전부 음수다. RealSense 디프로젝션 결과라면 z 는 깊이이므로 양수여야 한다.")
    print("  즉 이 PLY 는 y 가 위, z 가 뒤를 향하는 규약으로 저장되어 있다.")
    print("  RealSense 카메라 좌표계(x 오른쪽, y 아래, z 앞)로 맞추려면 (x, -y, -z) 로 바꿔야 한다.")
    print("\n  앞서 생기원 시편에서도 같은 문제가 있었고, 내부 파라미터로 투영했을 때")
    print("  (x, -y, -z) 로 바꿔야 실제 사진과 일치함을 확인한 바 있다. 같은 저장 경로로 보인다.")
else:
    print("  z 가 양수다. RealSense 디프로젝션 프레임과 같다. 변환 불필요.")

print(f"\n=== 변환 전후 비교 (첫 점) ===")
p = P[0]
print(f"  PLY 그대로        ({p[0]*1000:>9.2f}, {p[1]*1000:>9.2f}, {p[2]*1000:>9.2f}) mm")
print(f"  (x, -y, -z) 적용  ({p[0]*1000:>9.2f}, {-p[1]*1000:>9.2f}, {-p[2]*1000:>9.2f}) mm")
print("  단위는 m -> mm 로 1000 을 곱한 값이다. Mirobot 이 mm 를 쓰므로 mm 로 전달한다.")
