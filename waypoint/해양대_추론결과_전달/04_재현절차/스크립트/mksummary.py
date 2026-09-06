# -*- coding: utf-8 -*-
"""조건별_지표.csv 를 사람이 바로 읽을 수 있는 요약 CSV 로 바꾼다.

받는 쪽에서 열 이름만 보고 뜻을 알 수 있어야 해서 한글 머리글을 쓴다.
"""
import csv, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SRC = '/res/시각화/조건별_지표.csv'
DST = '/res/시각화/추론_용접선_요약.csv'

COLS = [
    ('condition', '조건'),
    ('thick', '판재두께_mm'),
    ('line', '용접선'),
    ('ang', '시편회전_도'),
    ('plane_angle', '판사이각도_도'),
    ('status', '판정'),
    ('nwp', '웨이포인트_개수'),
    ('plen', '경로길이_mm'),
    ('iou', '용접선IoU_퍼센트'),
    ('dev', '라벨대비편차_mm'),
    ('kgap', '해양대교선과의거리_mm'),
    ('prob', '평균확률'),
    ('ratio', '유효구간비율'),
    ('n_pred', '추론용접점_개수'),
    ('n_c5', '접촉5mm라벨_개수'),
    ('n_b3', '해양대3mm밴드_개수'),
    ('manual', '수동라벨_있음'),
]
RND = {'plane_angle': 2, 'plen': 1, 'iou': 2, 'dev': 2, 'kgap': 2, 'prob': 3, 'ratio': 3}

R = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
with open(DST, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow([k for _, k in COLS])
    for r in R:
        out = []
        for src, _ in COLS:
            v = r[src]
            if src in RND and v not in ('', 'nan'):
                v = f"{float(v):.{RND[src]}f}"
            if src == 'manual':
                v = '있음' if v == 'True' else '없음'
            out.append(v)
        w.writerow(out)
print(f"저장 {os.path.basename(DST)}  {len(R)}행 {len(COLS)}열")
print('  ' + ', '.join(k for _, k in COLS))
