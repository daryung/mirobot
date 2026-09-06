# -*- coding: utf-8 -*-
"""
추론 용접선 경로 CSV 생성 — 로봇 드로잉 확인용

받는 쪽 상황
  카메라를 달지 않고, 이미 촬영된 3D 점군에서 뽑은 경로를 로봇이 따라 그리는지 확인한다.
  따라서 디프로젝션은 필요 없고, 우리가 3차원 웨이포인트를 바로 넘긴다.

좌표계를 세 가지로 낸다
  raw   : 전달받은 PLY 그대로. 점군과 대조할 때 쓴다
  cam   : (x, -y, -z). RealSense 카메라 좌표계(x 오른쪽, y 아래, z 앞).
          rs2_deproject_pixel_to_point() 가 내는 것과 같은 프레임이다.
          PLY 는 z 가 전부 음수라 그대로 쓰면 카메라 뒤쪽 좌표가 되어 로봇이 못 간다.
  local : 경로 시작점을 원점으로, 경로가 놓인 평면을 XY 로 정렬.
          로봇 작업공간에 올려 형상만 확인할 때 이게 가장 쓰기 쉽다.

단위는 전부 mm 다. Mirobot 이 mm 를 쓴다.
"""
import os, sys, glob, io, csv, argparse
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PNX = "/work/PointNeXt"; sys.path.insert(0, PNX); sys.path.insert(0, "/opt/Pointcept")

ap = argparse.ArgumentParser()
ap.add_argument('--weight', default='kmou_imp_N16.pt')
ap.add_argument('--npts', type=int, default=16384)
ap.add_argument('--grid', type=float, default=0.01)
ap.add_argument('--margin', type=float, default=80.0)
ap.add_argument('--rdp', type=float, default=0.5, help='웨이포인트 단순화 허용 오차 (mm)')
args = ap.parse_args()

MAN = '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable'
OUT, RES = '/out', '/res'
CATS = ["Butt", "Corner", "Edge", "Lap", "Tee"]
NCLS, EVAL_SEED, CONTACT = 3, 42, 5.0
BIN_MM, MIN_PER_BIN, CS, MIN_PTS, KFLOOR = 5.0, 3, 600, 20, 200
MAX_BASE, BASE_PER_WELD, MIN_BASE, BG_RATIO = 6000, 12, 1500, 0.30
os.makedirs(RES, exist_ok=True)


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
            f.seek(pos); arr = np.loadtxt(f, max_rows=n)
    return arr[:, :3].astype(np.float64), arr[:, 6].astype(int)


def wmed(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(np.median(x)) if c[-1] <= 0 else float(x[np.searchsorted(c, c[-1] / 2.0)])


def ma5(y):
    if len(y) < 5: return y
    yp = np.concatenate([np.repeat(y[0], 2), y, np.repeat(y[-1], 2)])
    return np.convolve(yp, np.ones(5) / 5, mode='valid')


def path_bins(P, W):
    """구간연결 + 이동평균 5구간 다듬기. 유효 구간 비율도 함께 낸다."""
    if len(P) < MIN_PTS: return None, 0.0
    c = P.mean(0); _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    u, v, w3 = Vt[0], Vt[1], Vt[2]
    q = P - c; t = q @ u; a = q @ v; b = q @ w3
    nb = max(2, int(np.ceil((t.max() - t.min()) / BIN_MM)))
    e = np.linspace(t.min(), t.max(), nb + 1)
    ti, ai, bi = [], [], []
    for k in range(nb):
        m = (t >= e[k]) & (t <= e[k + 1])
        if m.sum() >= MIN_PER_BIN:
            ti.append(np.median(t[m])); ai.append(wmed(a[m], W[m])); bi.append(wmed(b[m], W[m]))
    ratio = len(ti) / max(nb, 1)
    if len(ti) < 2: return None, ratio
    ti, ai, bi = np.array(ti), np.array(ai), np.array(bi)
    o = np.argsort(ti); ti, ai, bi = ti[o], ai[o], bi[o]
    ai, bi = ma5(ai), ma5(bi)
    ts = np.linspace(ti.min(), ti.max(), CS)
    cp = c + np.outer(ts, u) + np.outer(np.interp(ts, ti, ai), v) + np.outer(np.interp(ts, ti, bi), w3)
    return cp, ratio


def rdp(P, eps):
    """3차원 Douglas-Peucker. 형상을 유지하며 웨이포인트를 줄인다."""
    if len(P) < 3: return np.arange(len(P))
    keep = np.zeros(len(P), bool); keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1: continue
        a, b = P[i], P[j]
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-9:
            dist = np.linalg.norm(P[i + 1:j] - a, axis=1)
        else:
            d = d / L
            q = P[i + 1:j] - a
            dist = np.linalg.norm(q - np.outer(q @ d, d), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m)); stack.append((m, j))
    return np.where(keep)[0]


def to_local(cp):
    """시작점을 원점으로, 경로 평면을 XY 로 맞춘다."""
    c = cp.mean(0)
    _, _, Vt = np.linalg.svd(cp - c, full_matrices=False)
    n = Vt[2]                        # 경로가 놓인 평면의 법선
    x = cp[-1] - cp[0]
    x = x - (x @ n) * n
    nx = np.linalg.norm(x)
    if nx < 1e-9: x = Vt[0]
    else: x = x / nx
    y = np.cross(n, x)
    R = np.stack([x, y, n])          # 행이 새 축
    return (cp - cp[0]) @ R.T


ST = torch.load(os.path.join(OUT, args.weight), map_location='cuda', weights_only=False)
FEAT = ST['feat']
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
net = nn.ModuleList([bb, hd]); net.load_state_dict(ST['model']); net.eval()
print(f"가중치 {args.weight}  입력 {args.npts:,}점  grid {args.grid}  배경 여유 {args.margin:g} mm\n")


def fwd(q, f):
    B, N, _ = q.shape
    p = bb(dict(coord=q.reshape(-1, 3).contiguous(), feat=f.reshape(-1, FEAT).contiguous(),
                grid_size=args.grid, offset=torch.arange(1, B + 1, device=q.device, dtype=torch.long) * N))
    return hd(p.feat).view(B, N, NCLS)


files = sorted(glob.glob(os.path.join(MAN, '**', 'ascii', '*.ply'), recursive=True))
way, summ = [], []
for f in files:
    cond = os.path.basename(f).split('_label')[0]
    xyz, lk = read_ply_ascii(f)
    mm = xyz * 1000.0
    ih = np.where(lk == 1)[0]; iv = np.where(lk == 2)[0]; ig = np.where(lk == 0)[0]
    if len(ih) < 100 or len(iv) < 100: continue
    spec = np.concatenate([ih, iv])
    lo = mm[spec].min(0) - args.margin; hi = mm[spec].max(0) + args.margin
    g = mm[ig]; ig = ig[np.all((g >= lo) & (g <= hi), axis=1)]
    # 정답 (대조용)
    dv, _ = cKDTree(mm[ih]).query(mm[iv], k=1)
    dh, _ = cKDTree(mm[iv]).query(mm[ih], k=1)
    iw = np.concatenate([iv[dv < CONTACT], ih[dh < CONTACT]])
    wset = np.zeros(len(lk), bool); wset[iw] = True
    ib = np.concatenate([ih, iv]); ib = ib[~wset[ib]]

    rng = np.random.default_rng(EVAL_SEED)
    nb = int(np.clip(len(iw) * BASE_PER_WELD, MIN_BASE, MAX_BASE)); nb = min(nb, len(ib))
    sb = rng.choice(ib, nb, replace=False) if nb > 0 else np.empty(0, int)
    ng = min(len(ig), int((len(iw) + nb) * BG_RATIO))
    sg = rng.choice(ig, ng, replace=False) if ng > 0 else np.empty(0, int)
    keep = np.concatenate([iw, sb, sg]); rng.shuffle(keep)

    n = len(keep)
    r2 = np.random.default_rng(EVAL_SEED)
    idx = r2.choice(n, args.npts, replace=False) if n >= args.npts else \
        np.concatenate([np.arange(n), r2.choice(n, args.npts - n, replace=True)])
    sel_pts = keep[idx]
    X = xyz[sel_pts].astype(np.float32); M = mm[sel_pts]
    y = np.full(len(sel_pts), 1, np.int64); y[wset[sel_pts]] = 0; y[lk[sel_pts] == 0] = 2

    c0 = X.mean(0, keepdims=True); q = X - c0
    q = q / (np.max(np.linalg.norm(q, axis=1)) + 1e-8)
    h = (X[:, 1:2] - X[:, 1].min()).astype(np.float32)
    oh = np.zeros((args.npts, 5), np.float32); oh[:, CATS.index('Tee')] = 1
    feat = np.concatenate([q, oh, h], 1).astype(np.float32)
    with torch.no_grad():
        o = fwd(torch.from_numpy(q).unsqueeze(0).cuda(), torch.from_numpy(feat).unsqueeze(0).cuda())
    prob = torch.softmax(o[0], dim=-1).cpu().numpy()

    p0 = prob[:, 0]; am = prob.argmax(1) == 0
    if am.sum() >= KFLOOR:
        sel = am
    else:
        sel = np.zeros(len(p0), bool); sel[np.argsort(-p0)[:KFLOOR]] = True
    cp, ratio = path_bins(M[sel], p0[sel])
    pmean = float(p0[sel].mean())

    inter = int(((prob.argmax(1) == 0) & (y == 0)).sum())
    uni = int(((prob.argmax(1) == 0) | (y == 0)).sum())
    iou = inter / max(uni, 1) * 100
    gt = M[y == 0]
    dev = ''
    if cp is not None and len(gt) >= MIN_PTS:
        d, _ = cKDTree(cp).query(gt, k=1); dev = round(float(d.mean()), 2)

    if cp is None:
        summ.append(dict(condition=cond, status='산출불가', 사유='경로 산출 실패',
                         waypoints=0, path_length_mm='', weld_iou_pct=round(iou, 2),
                         dev_vs_label_mm=dev, pred_points=int(am.sum()),
                         mean_prob=round(pmean, 3), valid_bin_ratio=round(ratio, 3)))
        continue

    # 검출 실패 판정 (문턱값은 NIA 에서 정한 값이라 이 데이터에서는 잠정임)
    st = '정상'
    if pmean < 0.758 or ratio < 0.511: st = '검토필요'

    seg = np.linalg.norm(np.diff(cp, axis=0), axis=1)
    plen = float(seg.sum())
    ki = rdp(cp, args.rdp)
    W = cp[ki]
    cam = W * np.array([1.0, -1.0, -1.0])
    loc = to_local(W)
    for s, (a_, b_, c_) in enumerate(zip(W, cam, loc)):
        way.append(dict(condition=cond, seq=s,
                        x_raw_mm=round(a_[0], 3), y_raw_mm=round(a_[1], 3), z_raw_mm=round(a_[2], 3),
                        x_cam_mm=round(b_[0], 3), y_cam_mm=round(b_[1], 3), z_cam_mm=round(b_[2], 3),
                        x_local_mm=round(c_[0], 3), y_local_mm=round(c_[1], 3), z_local_mm=round(c_[2], 3),
                        status=st))
    summ.append(dict(condition=cond, status=st, 사유='' if st == '정상' else '신뢰 신호 낮음',
                     waypoints=len(W), path_length_mm=round(plen, 1),
                     weld_iou_pct=round(iou, 2), dev_vs_label_mm=dev,
                     pred_points=int(am.sum()), mean_prob=round(pmean, 3),
                     valid_bin_ratio=round(ratio, 3)))

ok = [s for s in summ if s['status'] == '정상']
print(f"조건 {len(summ)}건  정상 {len(ok)}건  검토필요 {len([s for s in summ if s['status']=='검토필요'])}건  "
      f"산출불가 {len([s for s in summ if s['status']=='산출불가'])}건")
pl = np.array([s['path_length_mm'] for s in summ if s['path_length_mm'] != ''])
wp = np.array([s['waypoints'] for s in summ if s['waypoints'] > 0])
iu = np.array([s['weld_iou_pct'] for s in summ])
dv = np.array([s['dev_vs_label_mm'] for s in summ if s['dev_vs_label_mm'] != ''])
print(f"경로 길이  중앙값 {np.median(pl):.1f} mm  범위 {pl.min():.1f} ~ {pl.max():.1f} mm")
print(f"웨이포인트 중앙값 {int(np.median(wp))}개 (허용오차 {args.rdp:g} mm 로 단순화)")
print(f"용접선 IoU 중앙값 {np.median(iu):.2f} %")
if len(dv): print(f"라벨 대비 경로 편차 중앙값 {np.median(dv):.2f} mm")

with open(os.path.join(RES, '추론_용접선_웨이포인트.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(way[0].keys())); w.writeheader(); w.writerows(way)
with open(os.path.join(RES, '추론_용접선_요약.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
print(f"\n저장: 추론_용접선_웨이포인트.csv ({len(way):,}행) / 추론_용접선_요약.csv ({len(summ)}행)")
