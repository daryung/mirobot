# -*- coding: utf-8 -*-
"""합동 학습 시드 3개(42, 1, 2)를 모아 평균과 표준편차를 낸다.

표준편차는 표본표준편차(ddof=1)다. 시드 3개는 모집단이 아니라 표본이므로
n-1 로 나누는 쪽이 재현성 폭을 보수적으로(더 넓게) 잡는다.
"""
import os, io, sys, csv
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RES = '/res'
COLS = ['해양대IoU', '해양대mIoU', 'D04', 'D08', 'D12', 'NIA비드IoU', 'NIAmIoU']
SRC = [('42', '해양대66_조합_합동.csv', '합동(NIA+해양대)'),
       ('1', '해양대66_합동_시드1.csv', '합동_시드1'),
       ('2', '해양대66_합동_시드2.csv', '합동_시드2')]

rows = []
for seed, fn, key in SRC:
    p = os.path.join(RES, fn)
    hit = None
    with open(p, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['조건'] == key: hit = r
    if hit is None:
        print(f"  시드 {seed}: {fn} 에서 '{key}' 행을 못 찾음"); continue
    rows.append((seed, {c: float(hit[c]) for c in COLS}))

print(f"시드 {len(rows)}개 수집\n")
hdr = f"{'시드':>6s}" + ''.join(f"{c:>12s}" for c in COLS)
print(hdr); print('-' * len(hdr.encode('utf-8')) if False else '-' * 90)
for s, d in rows:
    print(f"{s:>6s}" + ''.join(f"{d[c]:>12.2f}" for c in COLS))
print('-' * 90)

stat = {}
for c in COLS:
    v = np.array([d[c] for _, d in rows])
    stat[c] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max())
print(f"{'평균':>6s}" + ''.join(f"{stat[c][0]:>12.2f}" for c in COLS))
print(f"{'표준편차':>5s}" + ''.join(f"{stat[c][1]:>12.2f}" for c in COLS))
print(f"{'폭':>7s}" + ''.join(f"{stat[c][3]-stat[c][2]:>12.2f}" for c in COLS))

out = os.path.join(RES, '합동학습_시드3_요약.csv')
with open(out, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['지표', '시드42', '시드1', '시드2', '평균', '표준편차', '최소', '최대', '표기'])
    for c in COLS:
        m, sd, mn, mx = stat[c]
        w.writerow([c] + [f"{d[c]:.2f}" for _, d in rows] +
                   [f"{m:.2f}", f"{sd:.2f}", f"{mn:.2f}", f"{mx:.2f}", f"{m:.2f} ± {sd:.2f}"])
print(f"\n저장: {os.path.basename(out)}")

print("\n=== 보고용 문장 ===")
a, b = stat['해양대IoU'], stat['NIA비드IoU']
print(f"  해양대 용접선 IoU  {a[0]:.2f} ± {a[1]:.2f} %  (시드 3개, 폭 {a[3]-a[2]:.2f} %p)")
print(f"  NIA 용접비드 IoU  {b[0]:.2f} ± {b[1]:.2f} %  (합동 학습 전 81.70 %)")
for c in ('D04', 'D08', 'D12'):
    m, sd, mn, mx = stat[c]
    print(f"  {c}  {m:.2f} ± {sd:.2f} %")
