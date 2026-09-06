# -*- coding: utf-8 -*-
"""조건별 지표에서 두께별 통계를 낸다. 문서에 넣을 수치를 만들기 위한 것."""
import csv, io, sys
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

R = list(csv.DictReader(open('/res/시각화/조건별_지표.csv', encoding='utf-8-sig')))
for r in R:
    for k in ('iou', 'plen', 'dev', 'kgap', 'ratio', 'prob', 'plane_angle'):
        r[k] = float(r[k]) if r[k] not in ('', 'nan') else np.nan
    for k in ('thick', 'ang', 'nwp', 'n_c5', 'n_b3', 'n_pred'):
        r[k] = int(r[k])

print(f"조건 {len(R)}건   판정 " + ', '.join(f"{s} {sum(1 for r in R if r['status']==s)}건"
                                          for s in ('정상', '검토필요', '산출불가')))
print()
hdr = ['두께', '건수', 'IoU 중앙값', '교선-표면 거리', '라벨 편차', '경로 길이', '웨이포인트', '3mm 점수', '5mm 점수']
print(''.join(f"{h:>14s}" for h in hdr))
print('-' * 126)
for th in (4, 8, 12):
    d = [r for r in R if r['thick'] == th]
    f = lambda k: np.nanmedian([r[k] for r in d])
    print(f"{th:>12d}{len(d):>14d}{f('iou'):>14.2f}{f('kgap'):>14.2f}{f('dev'):>14.2f}"
          f"{f('plen'):>14.1f}{f('nwp'):>14.0f}{f('n_b3'):>14.0f}{f('n_c5'):>14.0f}")
print('-' * 126)
f = lambda k: np.nanmedian([r[k] for r in R])
print(f"{'전체':>11s}{len(R):>14d}{f('iou'):>14.2f}{f('kgap'):>14.2f}{f('dev'):>14.2f}"
      f"{f('plen'):>14.1f}{f('nwp'):>14.0f}{f('n_b3'):>14.0f}{f('n_c5'):>14.0f}")

print("\n=== 회전각 두 무리로 나눠 본 IoU ===")
lo = [r['iou'] for r in R if r['ang'] <= 50]
hi = [r['iou'] for r in R if r['ang'] >= 130]
print(f"  0~50도   {len(lo)}건  중앙값 {np.median(lo):.2f} %")
print(f"  130~170도 {len(hi)}건  중앙값 {np.median(hi):.2f} %")

print("\n=== 판 사이 각도 ===")
pa = np.array([r['plane_angle'] for r in R])
print(f"  중앙값 {np.nanmedian(pa):.2f}도   범위 {np.nanmin(pa):.2f} ~ {np.nanmax(pa):.2f}도")

print("\n=== 경로 길이와 웨이포인트 (전체) ===")
pl = np.array([r['plen'] for r in R]); wp = np.array([r['nwp'] for r in R])
print(f"  길이   중앙값 {np.nanmedian(pl):.1f} mm   범위 {np.nanmin(pl):.1f} ~ {np.nanmax(pl):.1f} mm")
print(f"  웨이포인트 중앙값 {int(np.median(wp))}개   범위 {wp.min()} ~ {wp.max()}개")

print("\n=== 가장 낮은 5건 ===")
for r in sorted(R, key=lambda r: r['iou'])[:5]:
    print(f"  {r['condition']:<20s} IoU {r['iou']:5.2f} %  편차 {r['dev']:5.2f} mm  "
          f"교선거리 {r['kgap']:5.2f} mm  판정 {r['status']}")
print("\n=== 수동 라벨 없이 복원한 2건 ===")
for r in R:
    if r['manual'] == 'False':
        print(f"  {r['condition']:<20s} IoU {r['iou']:5.2f} %  판정 {r['status']}")
