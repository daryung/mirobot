# -*- coding: utf-8 -*-
"""
접촉 밴드 라벨로 파인튜닝 데이터를 다시 만든다.

현행 라벨과의 차이
  현행 : 두 모재 평면을 피팅해 구한 교선에서 3 mm 이내
         교선이 재료 안쪽에 있어 판이 두꺼울수록 표면 점이 멀어진다.
         D04 506점 / D08 29점 / D12 6점 으로 두꺼운 판에 양성 표본이 없다.
  대안 : 수동 라벨된 수평 모재와 수직 모재가 서로 4 mm 이내로 맞닿는 표면 점
         표면끼리 재므로 두께와 무관하고 평면 피팅 실패에도 영향받지 않는다.
         D04 1,250점 / D08 269점 / D12 101점

분할은 앞과 동일하게 둔다. 라벨 정의만 바뀐 비교가 되도록 하기 위함이다.
  시험 A040·A150, 검증 A020, 학습 나머지
"""
import os, sys, glob, io, csv
import numpy as np
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MAN = '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable'
DST = '/data/KMOU66C'
SEED = 42
CONTACT_MM = 4.0
MAX_BASE, BASE_PER_WELD, MIN_BASE, BG_RATIO = 6000, 12, 1500, 0.30
TEST_A, VAL_A = {'A040', 'A150'}, {'A020'}


def read_ply_ascii(path):
    with open(path, 'r', encoding='ascii', errors='ignore') as f:
        n = 0
        while True:
            line = f.readline()
            if not line: raise ValueError('헤더 끝 없음')
            s = line.strip()
            if s.startswith('element vertex'): n = int(s.split()[2])
            if s == 'end_header': break
        pos = f.tell()
        try:
            import pandas as pd
            arr = pd.read_csv(f, sep=r'\s+', header=None, nrows=n).values
        except Exception:
            f.seek(pos)
            arr = np.loadtxt(f, max_rows=n)
    return arr[:, :3].astype(np.float64), arr[:, 6].astype(int)


for sp in ('train', 'val', 'test'):
    os.makedirs(os.path.join(DST, sp), exist_ok=True)

files = sorted(glob.glob(os.path.join(MAN, '**', 'ascii', '*.ply'), recursive=True))
print(f"수동 라벨 {len(files)}건, 접촉 문턱 {CONTACT_MM:g} mm\n")
rows = []
for f in files:
    cond = os.path.basename(f).split('_label')[0]
    A = cond.split('_')[3]
    split = 'test' if A in TEST_A else ('val' if A in VAL_A else 'train')

    xyz_m, lk = read_ply_ascii(f)
    mm = xyz_m * 1000.0
    ih = np.where(lk == 1)[0]
    iv = np.where(lk == 2)[0]
    ig = np.where(lk == 0)[0]
    if len(ih) < 100 or len(iv) < 100:
        print(f"  [건너뜀] {cond} 모재 점 부족"); continue

    # 접촉 밴드 = 상대 모재까지 CONTACT_MM 이내인 표면 점
    dv, _ = cKDTree(mm[ih]).query(mm[iv], k=1)
    dh, _ = cKDTree(mm[iv]).query(mm[ih], k=1)
    iw = np.concatenate([iv[dv < CONTACT_MM], ih[dh < CONTACT_MM]])
    wset = np.zeros(len(lk), bool); wset[iw] = True
    ib = np.concatenate([ih, iv]); ib = ib[~wset[ib]]      # 밴드에 안 든 모재

    rng = np.random.default_rng(SEED)
    nb = int(np.clip(len(iw) * BASE_PER_WELD, MIN_BASE, MAX_BASE)); nb = min(nb, len(ib))
    sb = rng.choice(ib, nb, replace=False) if nb > 0 else np.empty(0, int)
    ng = min(len(ig), int((len(iw) + nb) * BG_RATIO))
    sg = rng.choice(ig, ng, replace=False) if ng > 0 else np.empty(0, int)

    keep = np.concatenate([iw, sb, sg]); rng.shuffle(keep)
    out = np.zeros((len(keep), 7), np.float32)
    out[:, :3] = xyz_m[keep]
    lab = np.full(len(lk), 1, np.int64); lab[wset] = 0; lab[lk == 0] = 2
    out[:, 6] = lab[keep]
    np.save(os.path.join(DST, split, cond + '.npy'), out)

    rows.append(dict(조건=cond, D=cond.split('_')[1], 분할=split, 원본점=len(lk),
                     용접선=int(len(iw)), 모재=int(nb), 배경=int(ng), 저장점=len(keep),
                     용접선비율=round(len(iw) / max(len(keep), 1) * 100, 2)))

print(f"{'분할':>6s} {'건수':>5s} {'저장점 중앙':>12s} {'용접선 중앙':>12s} {'용접선비율 중앙':>15s}")
for sp in ('train', 'val', 'test'):
    g = [r for r in rows if r['분할'] == sp]
    if not g: continue
    t = np.array([r['저장점'] for r in g]); w = np.array([r['용접선'] for r in g])
    rt = np.array([r['용접선비율'] for r in g])
    print(f"{sp:>6s} {len(g):>5d} {int(np.median(t)):>12,} {int(np.median(w)):>12,} {np.median(rt):>14.2f} %")

print(f"\n=== 판 두께별 용접선 점 수 ===")
for d in ('D04', 'D08', 'D12'):
    g = [r for r in rows if r['D'] == d]
    if not g: continue
    w = np.array([r['용접선'] for r in g])
    print(f"  {d}  {len(g):>2d}건  중앙값 {int(np.median(w)):>5,}점  범위 {w.min():,} ~ {w.max():,}점  "
          f"100점 미만 {int((w<100).sum())}건")

print(f"\n참고 — 현행 교선 3 mm 라벨: D04 506 / D08 29 / D12 6점, 0점 8건")
with open('/res/해양대66_접촉라벨_전처리.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('저장:', DST)
