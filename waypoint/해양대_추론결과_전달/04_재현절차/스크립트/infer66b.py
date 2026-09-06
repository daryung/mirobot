# -*- coding: utf-8 -*-
"""
해양대 66조건 추론 기준선 — 두 조건으로 잰다.

  A. 원본 그대로       : 배경까지 전부 넣는다. 실제 촬영본을 그대로 넣는 상황
  B. 배경 제거         : 라벨 0 을 뺀 시편 점만 넣는다. 2D 마스크 크롭이 완벽했을 때의 상한

앞서 생기원 시편에서 배경 포함 여부가 결과를 갈랐으므로(예측 점 1.1 -> 48.8점),
같은 갈래를 여기서도 확인한다. 어느 쪽이 원인인지 알아야 파인튜닝 설계가 선다.

라벨 대응 : 해양대 3 -> 우리 0(용접선), 해양대 1·2 -> 우리 1(모재), 해양대 0 -> 우리 2(배경)
"""
import os, sys, glob, csv, io
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PNX = "/work/PointNeXt"; sys.path.insert(0, PNX); sys.path.insert(0, "/opt/Pointcept")

PLYDIR = '/kmou/06_용접선_3mm_결과_66조건/PLY/ascii'
OUT, RESULT = '/out', '/res'
CATS = ["Butt", "Corner", "Edge", "Lap", "Tee"]
NCLS, NPTS, EVAL_SEED = 3, 8192, 42
BIN_MM, MIN_PER_BIN, CS, MIN_PTS, KFLOOR = 5.0, 3, 600, 20, 200
os.makedirs(RESULT, exist_ok=True)

st = torch.load(os.path.join(OUT, 'ptv3_3cls_norgb_8192_lovasz.pt'), map_location='cuda', weights_only=False)
FEAT = st['feat']
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
bb = PointTransformerV3(
    in_channels=FEAT, order=("z", "z-trans"), stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(48,) * 5,
    dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16), dec_patch_size=(48,) * 4,
    mlp_ratio=4, qkv_bias=True, enable_flash=False,
    upcast_attention=True, upcast_softmax=True, enc_mode=False).cuda()
hd = nn.Linear(64, NCLS).cuda()
net = nn.ModuleList([bb, hd]); net.load_state_dict(st['model']); net.eval()


def fwd(q, f):
    B, N, _ = q.shape
    p = bb(dict(coord=q.reshape(-1, 3).contiguous(), feat=f.reshape(-1, FEAT).contiguous(),
                grid_size=0.01, offset=torch.arange(1, B + 1, device=q.device, dtype=torch.long) * N))
    return hd(p.feat).view(B, N, NCLS)


def read_ply_ascii(path):
    with open(path, 'r', encoding='ascii', errors='ignore') as f:
        n = 0
        while True:
            line = f.readline()
            if not line: raise ValueError('헤더 끝을 찾지 못함')
            s = line.strip()
            if s.startswith('element vertex'): n = int(s.split()[2])
            if s == 'end_header': break
        arr = np.loadtxt(f, max_rows=n)
    return arr[:, :3].astype(np.float64), arr[:, 6].astype(int)


def wmed(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(np.median(x)) if c[-1] <= 0 else float(x[np.searchsorted(c, c[-1] / 2.0)])


def ma5(y):
    if len(y) < 5: return y
    yp = np.concatenate([np.repeat(y[0], 2), y, np.repeat(y[-1], 2)])
    return np.convolve(yp, np.ones(5) / 5, mode='valid')


def path_bins(P, W=None):
    if len(P) < MIN_PTS: return None
    c = P.mean(0); _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    u, v, w3 = Vt[0], Vt[1], Vt[2]
    q = P - c; t = q @ u; a = q @ v; b = q @ w3
    nb = max(2, int(np.ceil((t.max() - t.min()) / BIN_MM)))
    e = np.linspace(t.min(), t.max(), nb + 1)
    ti, ai, bi = [], [], []
    for k in range(nb):
        m = (t >= e[k]) & (t <= e[k + 1])
        if m.sum() >= MIN_PER_BIN:
            ti.append(np.median(t[m]))
            ai.append(wmed(a[m], W[m]) if W is not None else np.median(a[m]))
            bi.append(wmed(b[m], W[m]) if W is not None else np.median(b[m]))
    if len(ti) < 2: return None
    ti, ai, bi = np.array(ti), np.array(ai), np.array(bi)
    o = np.argsort(ti); ti, ai, bi = ti[o], ai[o], bi[o]
    ai, bi = ma5(ai), ma5(bi)
    ts = np.linspace(ti.min(), ti.max(), CS)
    return c + np.outer(ts, u) + np.outer(np.interp(ts, ti, ai), v) + np.outer(np.interp(ts, ti, bi), w3)


def select(prob):
    p0 = prob[:, 0]; am = prob.argmax(1) == 0
    sel = am if am.sum() >= KFLOOR else np.zeros(len(p0), bool)
    if am.sum() < KFLOOR: sel[np.argsort(-p0)[:KFLOOR]] = True
    return sel, p0


def run(xyz_m, lab):
    n = len(xyz_m)
    rng = np.random.default_rng(EVAL_SEED)
    idx = rng.choice(n, NPTS, replace=False) if n >= NPTS else \
        np.concatenate([np.arange(n), rng.choice(n, NPTS - n, replace=True)])
    xyz = xyz_m[idx].astype(np.float32); y = lab[idx]; mms = xyz_m[idx] * 1000.0
    c0 = xyz.mean(0, keepdims=True); q = xyz - c0
    q = q / (np.max(np.linalg.norm(q, axis=1)) + 1e-8)
    h = (xyz[:, 1:2] - xyz[:, 1].min()).astype(np.float32)
    oh = np.zeros((NPTS, 5), np.float32); oh[:, CATS.index('Tee')] = 1
    feat = np.concatenate([q, oh, h], 1).astype(np.float32)
    with torch.no_grad():
        o = fwd(torch.from_numpy(q).unsqueeze(0).cuda().float(),
                torch.from_numpy(feat).unsqueeze(0).cuda())
    return torch.softmax(o[0], dim=-1).cpu().numpy(), y, mms


files = sorted(glob.glob(os.path.join(PLYDIR, '*.ply')))
print(f"조건 {len(files)}건\n")

MODES = [('A', '원본 그대로'), ('B', '배경 제거')]
accs = {m: np.zeros((NCLS, 3), np.int64) for m, _ in MODES}
rows = []
ext = []
for f in files:
    cond = os.path.basename(f).split('_label')[0]
    xyz_m, lab_k = read_ply_ascii(f)
    lab = np.full(len(lab_k), 1, np.int64)
    lab[lab_k == 3] = 0
    lab[lab_k == 0] = 2
    span = (xyz_m.max(0) - xyz_m.min(0)) * 1000.0
    spec = lab != 2
    sspan = (xyz_m[spec].max(0) - xyz_m[spec].min(0)) * 1000.0 if spec.sum() > 10 else np.zeros(3)
    ext.append((span, sspan, spec.mean()))

    r = dict(조건=cond, 전체점=len(xyz_m), 시편점=int(spec.sum()),
             시편비율=round(float(spec.mean()) * 100, 2),
             전체범위mm=f"{span[0]:.0f}x{span[1]:.0f}x{span[2]:.0f}",
             시편범위mm=f"{sspan[0]:.0f}x{sspan[1]:.0f}x{sspan[2]:.0f}")
    for m, _nm in MODES:
        if m == 'A':
            X, Y = xyz_m, lab
        else:
            X, Y = xyz_m[spec], lab[spec]
            if len(X) < 100:
                r[f'{m}_용접선IoU'] = ''; r[f'{m}_argmax'] = ''; r[f'{m}_편차'] = ''
                continue
        prob, y, mms = run(X, Y)
        pred = prob.argmax(1)
        for c in range(NCLS):
            accs[m][c, 0] += int(((pred == c) & (y == c)).sum())
            accs[m][c, 1] += int((pred == c).sum())
            accs[m][c, 2] += int((y == c).sum())
        inter = int(((pred == 0) & (y == 0)).sum()); uni = int(((pred == 0) | (y == 0)).sum())
        r[f'{m}_용접선IoU'] = round(inter / max(uni, 1) * 100, 2)
        r[f'{m}_argmax'] = int((pred == 0).sum())
        gt = mms[y == 0]
        sel, p0 = select(prob)
        cp = path_bins(mms[sel], p0[sel])
        if cp is not None and len(gt) >= MIN_PTS:
            d, _ = cKDTree(cp).query(gt, k=1)
            r[f'{m}_편차'] = round(float(d.mean()), 2)
        else:
            r[f'{m}_편차'] = ''
    rows.append(r)

sp = np.array([e[0] for e in ext]); ss = np.array([e[1] for e in ext]); rt = np.array([e[2] for e in ext])
print("=== 점군 구성 ===")
print(f"  전체 촬영 범위  중앙값 {np.median(sp[:,0]):.0f} x {np.median(sp[:,1]):.0f} x {np.median(sp[:,2]):.0f} mm")
print(f"  시편만의 범위   중앙값 {np.median(ss[:,0]):.0f} x {np.median(ss[:,1]):.0f} x {np.median(ss[:,2]):.0f} mm")
print(f"  시편이 차지하는 점 비율  중앙값 {np.median(rt)*100:.1f} %  (나머지는 배경)")
print("  학습 데이터는 배경이 미리 제거되어 시편이 점의 2/3 를 차지했음\n")

names = ['용접선', '모재', '배경']
for m, nm in MODES:
    a = accs[m]
    mi = []
    print(f"=== {nm} ===")
    for c in range(NCLS):
        i, pr, g = a[c]
        iou = i / max(pr + g - i, 1) * 100
        mi.append(iou)
        print(f"  {names[c]:6s} IoU {iou:6.2f} %   정답점 {g:>9,}  예측점 {pr:>9,}")
    tot = a[:, 2].sum(); corr = a[:, 0].sum()
    print(f"  mIoU {np.mean(mi):.2f} %   Accuracy {corr/max(tot,1)*100:.2f} %")
    iv = np.array([r[f'{m}_용접선IoU'] for r in rows if r.get(f'{m}_용접선IoU') != ''])
    am = np.array([r[f'{m}_argmax'] for r in rows if r.get(f'{m}_argmax') != ''])
    dv = np.array([r[f'{m}_편차'] for r in rows if r.get(f'{m}_편차') not in ('', None)])
    print(f"  조건별 용접선 IoU 중앙값 {np.median(iv):.2f} %, 0 % 인 조건 {int((iv==0).sum())} / {len(iv)}건")
    print(f"  argmax 로 용접선이라 찍은 점 중앙값 {int(np.median(am))}점, 0점 {int((am==0).sum())}건")
    if len(dv): print(f"  경로 편차 중앙값 {np.median(dv):.2f} mm")
    print()

with open(os.path.join(RESULT, '해양대66_추론_기준선.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('저장: 해양대66_추론_기준선.csv')
