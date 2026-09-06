# -*- coding: utf-8 -*-
"""해양대 66조건 3D 시각화 — 조건별 그림 + 요약 그림

조건별 그림 한 장에 담는 것
  좌상  시편 점군 3차원 모습과 추론 경로
  우상  용접선에 직각인 단면. 두 판이 이루는 각도와 경로 위치가 여기서 보인다
  하단  용접선 길이 방향 잔차. 경로가 얼마나 곧게 나왔는지 본다

라벨 세 가지를 구분해서 그린다
  해양대 3mm 밴드   전달받은 원본 라벨. 두 평면의 교선에서 3mm 안쪽
  접촉 5mm          우리가 다시 만든 라벨. 반대쪽 판까지 5mm 안쪽인 표면점
  추론              모델이 용접선으로 분류한 점

교선 기준과 표면 접촉 기준이 왜 다른지가 이 그림의 핵심이다.
교선은 두 판을 무한 평면으로 늘렸을 때 만나는 선이라 실제 판 표면보다 안쪽에 있다.
두께가 두꺼울수록 그 차이가 커지고, D12 에서 교선 라벨이 표면에서 멀어진다.
"""
import os, sys, glob, io, csv, argparse, time
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "/work/PointNeXt"); sys.path.insert(0, "/opt/Pointcept")

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
from matplotlib import font_manager as fm
fm.fontManager.addfont('/wfonts/malgun.ttf')
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ap = argparse.ArgumentParser()
ap.add_argument('--weight', default='kmou_cb_합동NIA_해양대.pt')
ap.add_argument('--npts', type=int, default=8192)
ap.add_argument('--grid', type=float, default=0.01)
ap.add_argument('--margin', type=float, default=80.0)
ap.add_argument('--rdp', type=float, default=0.5)
ap.add_argument('--dpi', type=int, default=130)
ap.add_argument('--limit', type=int, default=0, help='앞에서 몇 조건만 (0 이면 전부)')
args = ap.parse_args()

K6 = '/kmou/06_용접선_3mm_결과_66조건'
MAN = '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable'
OUT = '/out'
DST = '/res/시각화'
D_CON, D_SUM = os.path.join(DST, '조건별'), os.path.join(DST, '요약')
for d in (D_CON, D_SUM): os.makedirs(d, exist_ok=True)

CATS = ["Butt", "Corner", "Edge", "Lap", "Tee"]
NCLS, EVAL_SEED, CONTACT, BAND3 = 3, 42, 5.0, 3.0
BIN_MM, MIN_PER_BIN, CS, MIN_PTS, KFLOOR = 5.0, 3, 600, 20, 200
MAX_BASE, BASE_PER_WELD, MIN_BASE, BG_RATIO = 6000, 12, 1500, 0.30

C_H, C_V = '#a9c4de', '#8a8f98'      # 수평판 / 수직판
C_B3, C_C5 = '#2f6fb5', '#22a06b'    # 해양대 3mm 밴드 / 접촉 5mm
C_PR, C_PATH, C_WP = '#e8833a', '#d02a2a', '#111111'   # 추론점 / 경로 / 웨이포인트
C_TP, C_FN, C_FP = '#1f9d55', '#2f6fb5', '#e8833a'     # 맞힘 / 놓침 / 잘못 잡음


def read_ply_ascii(path, ncol=7):
    with open(path, 'r', encoding='ascii', errors='ignore') as f:
        n = 0
        while True:
            line = f.readline()
            if not line: raise ValueError('헤더 끝 없음')
            s = line.strip()
            if s.startswith('element vertex'): n = int(s.split()[2])
            if s == 'end_header': break
        import pandas as pd
        arr = pd.read_csv(f, sep=r'\s+', header=None, nrows=n).values
    return arr[:, :3].astype(np.float64), arr[:, ncol - 1].astype(int)


def wmed(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(np.median(x)) if c[-1] <= 0 else float(x[np.searchsorted(c, c[-1] / 2.0)])


def ma5(y):
    if len(y) < 5: return y
    yp = np.concatenate([np.repeat(y[0], 2), y, np.repeat(y[-1], 2)])
    return np.convolve(yp, np.ones(5) / 5, mode='valid')


def path_bins(P, W):
    if len(P) < MIN_PTS: return None, 0.0, None
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
    if len(ti) < 2: return None, ratio, None
    ti, ai, bi = np.array(ti), np.array(ai), np.array(bi)
    o = np.argsort(ti); ti, ai, bi = ti[o], ai[o], bi[o]
    raw = (ti.copy(), ai.copy(), bi.copy())      # 다듬기 전. 그림에서 대조용
    ai, bi = ma5(ai), ma5(bi)
    ts = np.linspace(ti.min(), ti.max(), CS)
    cp = c + np.outer(ts, u) + np.outer(np.interp(ts, ti, ai), v) + np.outer(np.interp(ts, ti, bi), w3)
    return cp, ratio, (c, u, v, w3, raw, (ti, ai, bi))


def rdp(P, eps):
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
            d = d / L; q = P[i + 1:j] - a
            dist = np.linalg.norm(q - np.outer(q @ d, d), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k; keep[m] = True
            stack.append((i, m)); stack.append((m, j))
    return np.where(keep)[0]


def to_local(cp):
    """시작점을 원점으로, 경로 평면을 XY 로 맞춘다."""
    c = cp.mean(0)
    _, _, Vt = np.linalg.svd(cp - c, full_matrices=False)
    n = Vt[2]
    x = cp[-1] - cp[0]
    x = x - (x @ n) * n
    nx = np.linalg.norm(x)
    x = Vt[0] if nx < 1e-9 else x / nx
    y = np.cross(n, x)
    return (cp - cp[0]) @ np.stack([x, y, n]).T


def seg_dist(P, a, b):
    """점들에서 선분 ab 까지의 거리."""
    d = b - a; L2 = float(d @ d)
    if L2 < 1e-12: return np.linalg.norm(P - a, axis=1)
    s = np.clip((P - a) @ d / L2, 0.0, 1.0)
    return np.linalg.norm(P - (a + np.outer(s, d)), axis=1)


# ---------------------------------------------------------------- 모델
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


def fwd(q, f):
    B, N, _ = q.shape
    p = bb(dict(coord=q.reshape(-1, 3).contiguous(), feat=f.reshape(-1, FEAT).contiguous(),
                grid_size=args.grid, offset=torch.arange(1, B + 1, device=q.device, dtype=torch.long) * N))
    return hd(p.feat).view(B, N, NCLS)


# ---------------------------------------------------------------- 조건 목록
plys6 = sorted(glob.glob(os.path.join(K6, 'PLY', 'ascii', '*.ply')))
conds = [os.path.basename(p).split('_label')[0] for p in plys6]
man = {os.path.basename(p).split('_label')[0]: p
       for p in glob.glob(os.path.join(MAN, '**', 'ascii', '*.ply'), recursive=True)}
if args.limit: conds, plys6 = conds[:args.limit], plys6[:args.limit]
print(f"가중치 {args.weight}   조건 {len(conds)}건   수동 라벨 있음 {sum(c in man for c in conds)}건")
print(f"입력 {args.npts:,}점  grid {args.grid}  배경 여유 {args.margin:g} mm\n")


def kmou_row(cond):
    p = os.path.join(K6, 'CSV', f'{cond}_weld_path.csv')
    if not os.path.exists(p): return None
    with open(p, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): return r
    return None


rows, way, t0 = [], [], time.time()
for ci, (cond, p6) in enumerate(zip(conds, plys6), 1):
    thick, line, ang = cond.split('_')[1], cond.split('_')[2], int(cond.split('_')[3][1:])
    src = man.get(cond)
    if src is not None:
        xyz, lk = read_ply_ascii(src)          # 0 배경 / 1 수평판 / 2 수직판
        note = ''
    else:
        xyz, c6 = read_ply_ascii(p6)           # 0 배경 / 1 수평판 / 2 수직판 / 3 용접밴드
        lk = c6.copy()
        b = np.where(c6 == 3)[0]; pl = np.where((c6 == 1) | (c6 == 2))[0]
        if len(b) and len(pl):                 # 밴드점을 가까운 판으로 되돌린다
            _, j = cKDTree(xyz[pl]).query(xyz[b], k=1)
            lk[b] = c6[pl[j]]
        note = '수동 라벨 없음. 3mm 밴드점을 가까운 판으로 되돌려 복원'

    mm = xyz * 1000.0
    ih = np.where(lk == 1)[0]; iv = np.where(lk == 2)[0]; ig = np.where(lk == 0)[0]
    if len(ih) < 100 or len(iv) < 100:
        print(f"  [{ci:2d}/{len(conds)}] {cond}  판 점 부족, 건너뜀"); continue
    spec = np.concatenate([ih, iv])
    lo = mm[spec].min(0) - args.margin; hi = mm[spec].max(0) + args.margin
    g = mm[ig]; ig = ig[np.all((g >= lo) & (g <= hi), axis=1)]

    # 접촉 5mm 라벨 (우리 기준)
    dv, _ = cKDTree(mm[ih]).query(mm[iv], k=1)
    dh, _ = cKDTree(mm[iv]).query(mm[ih], k=1)
    iw = np.concatenate([iv[dv < CONTACT], ih[dh < CONTACT]])
    wset = np.zeros(len(lk), bool); wset[iw] = True
    ib = spec[~wset[spec]]

    # 해양대 3mm 밴드 (교선 기준). CSV 의 교선 선분에서 3mm 안쪽
    kr = kmou_row(cond)
    b3 = np.zeros(len(lk), bool); kline = None
    if kr is not None:
        a = np.array([float(kr['start_x_m']), float(kr['start_y_m']), float(kr['start_z_m'])]) * 1000
        b = np.array([float(kr['end_x_m']), float(kr['end_y_m']), float(kr['end_z_m'])]) * 1000
        kline = (a, b)
        b3[spec] = seg_dist(mm[spec], a, b) < BAND3
    pang = float(kr['plane_angle_deg']) if kr else float('nan')

    # 모델 입력 (평가와 같은 방식)
    rng = np.random.default_rng(EVAL_SEED)
    nb = int(np.clip(len(iw) * BASE_PER_WELD, MIN_BASE, MAX_BASE)); nb = min(nb, len(ib))
    sb = rng.choice(ib, nb, replace=False) if nb > 0 else np.empty(0, int)
    ng = min(len(ig), int((len(iw) + nb) * BG_RATIO))
    sg = rng.choice(ig, ng, replace=False) if ng > 0 else np.empty(0, int)
    keep = np.concatenate([iw, sb, sg]); rng.shuffle(keep)
    n = len(keep); r2 = np.random.default_rng(EVAL_SEED)
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
    sel = am if am.sum() >= KFLOOR else np.zeros(len(p0), bool)
    if am.sum() < KFLOOR: sel[np.argsort(-p0)[:KFLOOR]] = True

    cp, ratio, fr = path_bins(M[sel], p0[sel])
    inter = int((am & (y == 0)).sum()); uni = int((am | (y == 0)).sum())
    iou = inter / max(uni, 1) * 100
    gt = M[y == 0]
    dev = float('nan')
    if cp is not None and len(gt) >= MIN_PTS:
        d, _ = cKDTree(cp).query(gt, k=1); dev = float(d.mean())

    plen, nwp, W = float('nan'), 0, None
    if cp is not None:
        plen = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
        W = cp[rdp(cp, args.rdp)]; nwp = len(W)
    st = '산출불가' if cp is None else ('검토필요' if (p0[sel].mean() < 0.758 or ratio < 0.511) else '정상')

    if W is not None:
        cam = W * np.array([1.0, -1.0, -1.0])   # RealSense 카메라 좌표계로 뒤집는다
        loc = to_local(W)
        for s_i, (a_, b_, c_) in enumerate(zip(W, cam, loc)):
            way.append(dict(condition=cond, seq=s_i,
                            x_raw_mm=round(a_[0], 3), y_raw_mm=round(a_[1], 3), z_raw_mm=round(a_[2], 3),
                            x_cam_mm=round(b_[0], 3), y_cam_mm=round(b_[1], 3), z_cam_mm=round(b_[2], 3),
                            x_local_mm=round(c_[0], 3), y_local_mm=round(c_[1], 3), z_local_mm=round(c_[2], 3),
                            status=st))

    # 해양대 교선과 우리 경로가 얼마나 떨어져 있나. 교선은 두 평면을 늘려 만나는 선이라
    # 판이 두꺼울수록 실제 표면보다 안쪽에 놓인다. 이 값이 D12 에서 커지는 것이 라벨 기준을 바꾼 이유다
    kgap = float('nan')
    if cp is not None and kline is not None:
        kgap = float(np.median(seg_dist(cp, kline[0], kline[1])))

    # ------------------------------------------------------------ 그림
    fig = plt.figure(figsize=(13.0, 9.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.32, 1.0], hspace=0.30, wspace=0.20,
                           left=0.055, right=0.975, top=0.90, bottom=0.115)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # (1) 3차원 모습. 모델이 무엇을 용접선으로 골랐는지를 맞고 틀림으로 나눠 칠한다.
    #     맞힘 놓침 잘못잡음 세 가지가 그대로 IoU 의 분자와 분모다
    rs = np.random.default_rng(0)
    for ii, cc, nm in ((ih, C_H, '수평판'), (iv, C_V, '수직판')):
        s = ii if len(ii) <= 6000 else rs.choice(ii, 6000, replace=False)
        ax1.scatter(mm[s, 0], mm[s, 2], mm[s, 1], s=1.0, c=cc, lw=0, alpha=0.55, label=nm)
    tp = am & (y == 0); fn = (~am) & (y == 0); fp = am & (y != 0)
    for msk, cc, nm in ((tp, C_TP, '맞힘'), (fn, C_FN, '놓침'), (fp, C_FP, '잘못 잡음')):
        if msk.sum():
            ax1.scatter(M[msk, 0], M[msk, 2], M[msk, 1], s=3.0, c=cc, lw=0,
                        depthshade=False, label=f'{nm} {int(msk.sum()):,}점')
    if cp is not None:
        # 인식 점 위에 겹치므로 선을 굵게, 웨이포인트는 흰 테두리를 둘러 눈에 띄게 한다
        ax1.plot(cp[:, 0], cp[:, 2], cp[:, 1], c=C_PATH, lw=2.4, label='추론 경로')
        ax1.scatter(W[:, 0], W[:, 2], W[:, 1], s=26, c=C_WP, depthshade=False,
                    edgecolors='white', linewidths=0.8, label=f'웨이포인트 {nwp}개')
    # 축마다 실제 범위에 맞춘다. 세 축을 같은 길이로 잡으면 얇은 시편이 화면에서 눌린다
    lo3, hi3 = mm[spec].min(0), mm[spec].max(0)
    pad = (hi3 - lo3) * 0.06 + 2.0
    lo3, hi3 = lo3 - pad, hi3 + pad
    ax1.set_xlim(lo3[0], hi3[0]); ax1.set_ylim(lo3[2], hi3[2]); ax1.set_zlim(lo3[1], hi3[1])
    ext = hi3 - lo3
    ax1.set_box_aspect((ext[0], ext[2], ext[1]))   # 실제 비율 유지
    ax1.set_xlabel('x (mm)', labelpad=1); ax1.set_ylabel('z (mm)', labelpad=1); ax1.set_zlabel('y (mm)', labelpad=1)
    ax1.tick_params(labelsize=7.5, pad=0.5)
    for a_ in (ax1.xaxis, ax1.yaxis, ax1.zaxis):   # 눈금이 겹쳐 읽히지 않는 것을 막는다
        a_.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax1.view_init(elev=26, azim=-62)
    ax1.set_title('모델이 인식한 용접선', fontsize=11, pad=2)
    lg = ax1.legend(loc='upper center', fontsize=8, framealpha=0.88, borderpad=0.3, ncol=4,
                    columnspacing=0.8, handletextpad=0.35, bbox_to_anchor=(0.5, 0.10))
    for h in lg.legend_handles:
        if hasattr(h, 'set_sizes'): h.set_sizes([22])

    # (2) 용접선 직각 단면
    if fr is not None:
        c, u, v3, w3, raw, sm = fr
        tt = (mm[spec] - c) @ u
        mid = np.median((M[sel] - c) @ u)
        m2 = np.abs(tt - mid) < 8.0
        if m2.sum() < 50: m2 = np.abs(tt - mid) < 20.0
        idx2 = spec[m2]
        vv = (mm[idx2] - c) @ v3; ww = (mm[idx2] - c) @ w3
        isv = np.isin(idx2, iv)
        ax2.scatter(vv[~isv], ww[~isv], s=3, c=C_H, lw=0, label='수평판 단면')
        ax2.scatter(vv[isv], ww[isv], s=3, c=C_V, lw=0, label='수직판 단면')
        bb3 = idx2[b3[idx2]]
        if len(bb3):
            ax2.scatter((mm[bb3] - c) @ v3, (mm[bb3] - c) @ w3, s=9, c=C_B3, lw=0, label='해양대 3mm')
        cw = idx2[wset[idx2]]
        if len(cw):
            ax2.scatter((mm[cw] - c) @ v3, (mm[cw] - c) @ w3, s=9, c=C_C5, lw=0,
                        label='접촉 5mm', zorder=4)
        ps = M[sel]; tp = (ps - c) @ u; mp = np.abs(tp - mid) < 8.0
        if mp.sum():
            ax2.scatter((ps[mp] - c) @ v3, (ps[mp] - c) @ w3, s=22, facecolors='none',
                        edgecolors=C_PR, lw=0.7, label='추론 용접점', zorder=3)
        k = np.argmin(np.abs((cp - c) @ u - mid))
        pv, pw = (cp[k] - c) @ v3, (cp[k] - c) @ w3
        ax2.plot([pv], [pw], marker='x', ms=11, mew=2.4, c=C_PATH, ls='none', label='추론 경로 위치')
        if kline is not None:
            s_ = np.clip((c + mid * u - kline[0]) @ (kline[1] - kline[0]) /
                         max(float((kline[1] - kline[0]) @ (kline[1] - kline[0])), 1e-9), 0, 1)
            kp = kline[0] + s_ * (kline[1] - kline[0])
            ax2.plot([(kp - c) @ v3], [(kp - c) @ w3], marker='+', ms=12, mew=2.2,
                     c=C_B3, ls='none', label='해양대 교선 위치')
        # 접촉부만 보이도록 자른다. 판 전체를 담으면 정작 볼 곳이 몇 픽셀이 된다
        ax2.set_xlim(pv - 42, pv + 42); ax2.set_ylim(pw - 42, pw + 42)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('단면 가로 (mm)'); ax2.set_ylabel('단면 세로 (mm)')
        ax2.set_title(f'용접선 직각 단면 (길이 중앙 ±8 mm)   판 사이 각도 {pang:.1f}°', fontsize=11, pad=4)
        ax2.legend(fontsize=7.5, loc='upper center', ncol=2, columnspacing=0.8,
                   framealpha=0.9, borderpad=0.3, handletextpad=0.4)
        ax2.grid(alpha=0.25, lw=0.5)

        # (3) 길이 방향 잔차
        ti, ai, bi = sm; rti, rai, rbi = raw
        ax3.plot(rti - rti.min(), rai, c='#bbbbbb', lw=1.0, marker='.', ms=3, label='다듬기 전 좌우')
        ax3.plot(rti - rti.min(), rbi, c='#dcc7a8', lw=1.0, marker='.', ms=3, label='다듬기 전 상하')
        ax3.plot(ti - rti.min(), ai, c=C_PATH, lw=1.8, label='이동평균 5구간 좌우')
        ax3.plot(ti - rti.min(), bi, c='#7a4bbf', lw=1.8, label='이동평균 5구간 상하')
        ax3.set_xlabel('용접선 길이 방향 (mm)'); ax3.set_ylabel('경로 기준축 대비 위치 (mm)')
        jag = float(np.abs(np.diff(rai)).mean() + np.abs(np.diff(rbi)).mean())
        jag2 = float(np.abs(np.diff(ai)).mean() + np.abs(np.diff(bi)).mean())
        ax3.set_title(f'경로의 길이 방향 잔차. 회색이 구간연결 원값, 색이 이동평균 5구간을 거친 값   '
                      f'구간 간 흔들림 {jag:.2f} → {jag2:.2f} mm', fontsize=11, pad=4)
        ax3.grid(alpha=0.25, lw=0.5); ax3.legend(fontsize=8, ncol=4, loc='best', framealpha=0.9)
    else:
        for a_ in (ax2, ax3):
            a_.text(0.5, 0.5, '경로 산출 실패', ha='center', va='center', fontsize=13); a_.axis('off')

    fig.suptitle(f'{cond}    판재 두께 {int(thick[1:])} mm · 용접선 {line} · 회전 {ang}°', fontsize=14, y=0.965)
    foot = (f'용접선 IoU {iou:.1f} %      경로 길이 {plen:.1f} mm      웨이포인트 {nwp}개      '
            f'라벨 대비 편차 {dev:.2f} mm      해양대 교선과의 거리 {kgap:.2f} mm      '
            f'유효 구간 {ratio*100:.0f} %      평균 확률 {p0[sel].mean():.3f}      판정 {st}')
    fig.text(0.5, 0.048, foot, ha='center', fontsize=10)
    fig.text(0.5, 0.018, f'접촉 5mm 라벨 {int(wset.sum()):,}점 · 해양대 3mm 밴드 {int(b3.sum()):,}점 · '
                         f'추론 용접점 {int(am.sum()):,}점 / 입력 {args.npts:,}점'
                         + (f'      {note}' if note else ''),
             ha='center', fontsize=8.5, color='#555555')
    fig.savefig(os.path.join(D_CON, f'{cond}.png'), dpi=args.dpi)
    plt.close(fig)

    rows.append(dict(condition=cond, thick=int(thick[1:]), line=line, ang=ang,
                     plane_angle=pang, iou=iou, plen=plen, nwp=nwp, dev=dev, kgap=kgap,
                     ratio=ratio, prob=float(p0[sel].mean()), status=st,
                     n_c5=int(wset.sum()), n_b3=int(b3.sum()), n_pred=int(am.sum()),
                     manual=src is not None))
    if ci % 6 == 0 or ci == len(conds):
        print(f"  [{ci:2d}/{len(conds)}] {cond}  IoU {iou:5.1f} %  {time.time()-t0:5.0f}초")

print(f"\n조건별 그림 {len(rows)}장 저장: {D_CON}")

with open(os.path.join(DST, '추론_용접선_웨이포인트.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w_ = csv.DictWriter(f, fieldnames=list(way[0].keys())); w_.writeheader(); w_.writerows(way)
print(f"웨이포인트 {len(way):,}행 저장: 추론_용접선_웨이포인트.csv")

# ---------------------------------------------------------------- 요약 그림
import matplotlib.ticker as mt
R = rows
THS, LNS = [4, 8, 12], ['L1', 'L2']
ANGS = sorted({r['ang'] for r in R})
CT = {4: '#2f6fb5', 8: '#e8833a', 12: '#d02a2a'}


def pick(th=None, ln=None):
    return [r for r in R if (th is None or r['thick'] == th) and (ln is None or r['line'] == ln)]


# 요약 1 — 두께 × 각도 IoU 히트맵
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, ln in zip(axs, LNS):
    Z = np.full((len(THS), len(ANGS)), np.nan)
    for r in pick(ln=ln): Z[THS.index(r['thick']), ANGS.index(r['ang'])] = r['iou']
    im = ax.imshow(Z, cmap='RdYlGn', vmin=30, vmax=100, aspect='auto')
    ax.set_xticks(range(len(ANGS))); ax.set_xticklabels([f'{a}°' for a in ANGS], fontsize=9)
    ax.set_yticks(range(len(THS))); ax.set_yticklabels([f'{t} mm' for t in THS])
    for i in range(len(THS)):
        for j in range(len(ANGS)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f'{Z[i,j]:.0f}', ha='center', va='center', fontsize=8.5,
                        color='#111111')
    ax.set_title(f'용접선 {ln}', fontsize=11)
    ax.set_xlabel('시편 회전각'); ax.set_ylabel('판재 두께')
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label='용접선 IoU (%)')
fig.suptitle('조건별 용접선 IoU — 두께가 결과를 가르고 회전각은 영향이 작다', fontsize=13, y=0.99)
fig.savefig(os.path.join(D_SUM, '01_두께각도_IoU_히트맵.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# 요약 2 — 회전각에 따른 IoU
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, ln in zip(axs, LNS):
    for th in THS:
        d = sorted(pick(th, ln), key=lambda r: r['ang'])
        if d: ax.plot([r['ang'] for r in d], [r['iou'] for r in d], marker='o', ms=4,
                      c=CT[th], label=f'{th} mm')
    ax.set_title(f'용접선 {ln}', fontsize=11); ax.set_xlabel('시편 회전각 (°)')
    ax.grid(alpha=0.25, lw=0.5); ax.set_ylim(0, 100)
axs[0].set_ylabel('용접선 IoU (%)'); axs[0].legend(fontsize=9, title='판재 두께')
fig.suptitle('회전각에 따른 용접선 IoU — 각도별 뚜렷한 추세는 없다', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(D_SUM, '02_회전각별_IoU.png'), dpi=150)
plt.close(fig)

# 요약 3 — 두께별 분포
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, key, ttl, ylb in zip(axs, ('iou', 'dev', 'plen'),
                             ('용접선 IoU', '라벨 대비 경로 편차', '경로 길이'),
                             ('IoU (%)', '편차 (mm)', '길이 (mm)')):
    dat = [[r[key] for r in pick(th) if not np.isnan(r[key])] for th in THS]
    bp = ax.boxplot(dat, tick_labels=[f'{t} mm' for t in THS], widths=0.55, patch_artist=True)
    for pt, th in zip(bp['boxes'], THS): pt.set_facecolor(CT[th]); pt.set_alpha(0.55)
    for md in bp['medians']: md.set_color('#111111'); md.set_linewidth(1.6)
    ax.set_title(ttl, fontsize=11); ax.set_ylabel(ylb); ax.grid(alpha=0.25, lw=0.5, axis='y')
fig.suptitle('판재 두께별 분포 — 두꺼울수록 IoU 가 떨어지고 편차가 커진다', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(D_SUM, '03_두께별_분포.png'), dpi=150)
plt.close(fig)

# 요약 4 — 교선 기준과 표면 접촉 기준이 왜 다른가
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
w = 0.36
xs = np.arange(len(THS))
b3m = [np.mean([r['n_b3'] for r in pick(th)]) for th in THS]
c5m = [np.mean([r['n_c5'] for r in pick(th)]) for th in THS]
axs[0].bar(xs - w / 2, b3m, w, color=C_B3, label='해양대 3mm 밴드 (교선 기준)')
axs[0].bar(xs + w / 2, c5m, w, color=C_C5, label='접촉 5mm (표면 기준)')
for x, a, b in zip(xs, b3m, c5m):
    axs[0].text(x - w / 2, a, f'{a:,.0f}', ha='center', va='bottom', fontsize=9)
    axs[0].text(x + w / 2, b, f'{b:,.0f}', ha='center', va='bottom', fontsize=9)
axs[0].set_xticks(xs); axs[0].set_xticklabels([f'{t} mm' for t in THS])
axs[0].set_xlabel('판재 두께'); axs[0].set_ylabel('용접선 라벨 점 수 (조건 평균)')
axs[0].set_title('라벨 기준별 용접선 점 수', fontsize=11)
axs[0].legend(fontsize=9); axs[0].grid(alpha=0.25, lw=0.5, axis='y')

kg = [[r['kgap'] for r in pick(th) if not np.isnan(r['kgap'])] for th in THS]
bp = axs[1].boxplot(kg, tick_labels=[f'{t} mm' for t in THS], widths=0.55, patch_artist=True)
for pt, th in zip(bp['boxes'], THS): pt.set_facecolor(CT[th]); pt.set_alpha(0.55)
for md in bp['medians']: md.set_color('#111111'); md.set_linewidth(1.6)
axs[1].axhline(BAND3, c=C_B3, ls='--', lw=1.3, label='해양대 밴드 반경 3 mm')
axs[1].axhline(CONTACT, c=C_C5, ls='--', lw=1.3, label='접촉 기준 5 mm')
for i, d in enumerate(kg, 1):
    if d: axs[1].text(i, np.median(d), f' {np.median(d):.2f}', ha='left', va='center', fontsize=9)
axs[1].set_xlabel('판재 두께'); axs[1].set_ylabel('교선에서 표면 경로까지 거리 (mm)')
axs[1].set_title('교선은 판 표면보다 안쪽에 있다', fontsize=11)
axs[1].legend(fontsize=9); axs[1].grid(alpha=0.25, lw=0.5, axis='y')

fig.suptitle('교선 기준 3mm 밴드가 두꺼운 판에서 실패하는 이유 — 교선이 표면에서 멀어져 라벨이 실제 용접부를 벗어난다',
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(D_SUM, '04_라벨기준_비교.png'), dpi=150)
plt.close(fig)

# 요약 5 — 합동 학습 시드 3개
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
seeds = ['42', '1', '2']
kv = dict(해양대=[82.04, 81.79, 81.91], NIA=[80.45, 80.27, 80.69])
for ax, (nm, v) in zip(axs, kv.items()):
    v = np.array(v); m, sd = v.mean(), v.std(ddof=1)
    ax.bar(seeds, v, color='#2f6fb5' if nm == '해양대' else '#e8833a', width=0.55, alpha=0.85)
    ax.axhline(m, c='#111111', lw=1.2, ls='--')
    ax.fill_between([-0.5, 2.5], m - sd, m + sd, color='#111111', alpha=0.10)
    ax.set_xlim(-0.5, 2.5)
    for i, x in enumerate(v): ax.text(i, x, f'{x:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(v.min() - 1.2, v.max() + 0.9)
    ax.set_xlabel('학습 시드'); ax.set_ylabel('IoU (%)')
    ax.set_title(f'{"해양대 용접선" if nm=="해양대" else "NIA 용접비드"}   {m:.2f} ± {sd:.2f} %', fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
fig.suptitle('합동 학습 재현성 — 시드 3개, 점선이 평균 음영이 표준편차', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(D_SUM, '05_시드3개_재현성.png'), dpi=150)
plt.close(fig)

with open(os.path.join(DST, '조건별_지표.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w_ = csv.DictWriter(f, fieldnames=list(R[0].keys())); w_.writeheader(); w_.writerows(R)

iu = np.array([r['iou'] for r in R])
print(f"요약 그림 5장 저장: {D_SUM}")
print(f"\n용접선 IoU  중앙값 {np.median(iu):.2f} %  평균 {iu.mean():.2f} %  범위 {iu.min():.1f} ~ {iu.max():.1f} %")
for th in THS:
    d = np.array([r['iou'] for r in pick(th)])
    print(f"  {th:2d} mm  중앙값 {np.median(d):5.2f} %  ({len(d)}건)")
print(f"소요 {time.time()-t0:.0f}초")
