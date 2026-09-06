# -*- coding: utf-8 -*-
"""
접촉 밴드 문턱값 재비교 — 배경 크롭과 두께 균형을 적용한 조건에서 다시 잰다.

앞선 비교(3·4·5·6 mm)는 배경을 사무실 전체에서 뽑던 상태에서 잰 것이다.
그때는 점군 범위가 2.9 m 라 복셀이 약 14 mm 가 되어 얇은 접촉부가 뭉개졌고,
밴드를 넓힐수록 그 뭉개짐을 억지로 이기는 쪽으로 결과가 나왔을 가능성이 있다.

배경을 시편 경계 + 80 mm 로 자르면 복셀이 정상 크기가 된다.
그 조건에서는 좁은 밴드로도 충분할 수 있고, 그러면 라벨 잔차를 아끼면서 성능을 얻는다.
해양대에 권고할 밴드 반경을 정하는 근거이므로 다시 잰다.
"""
import os, sys, glob, io, csv, json, time
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PNX = "/work/PointNeXt"; sys.path.insert(0, PNX); sys.path.insert(0, "/opt/Pointcept")

MAN = '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable'
OUT, RES = '/out', '/res'
CATS = ["Butt", "Corner", "Edge", "Lap", "Tee"]
NCLS, NPTS, EVAL_SEED, SEED = 3, 8192, 42, 42
EPOCHS, LR, BATCH, MARGIN_MM = 60, 6e-5, 4, 80.0
MAX_BASE, BASE_PER_WELD, MIN_BASE, BG_RATIO = 6000, 12, 1500, 0.30
TEST_A, VAL_A = {'A040', 'A150'}, {'A020'}
BIN_MM, MIN_PER_BIN, CS, MIN_PTS = 5.0, 3, 600, 20
REP = {'D04': 1, 'D08': 2, 'D12': 3}
THRS = [3.0, 4.0, 5.0, 6.0]


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


def path_bins(P):
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
            ti.append(np.median(t[m])); ai.append(np.median(a[m])); bi.append(np.median(b[m]))
    if len(ti) < 2: return None
    ti, ai, bi = np.array(ti), np.array(ai), np.array(bi)
    o = np.argsort(ti); ti, ai, bi = ti[o], ai[o], bi[o]
    if len(ai) >= 5:
        ap = np.concatenate([np.repeat(ai[0], 2), ai, np.repeat(ai[-1], 2)])
        bp = np.concatenate([np.repeat(bi[0], 2), bi, np.repeat(bi[-1], 2)])
        ai = np.convolve(ap, np.ones(5) / 5, mode='valid')
        bi = np.convolve(bp, np.ones(5) / 5, mode='valid')
    ts = np.linspace(ti.min(), ti.max(), CS)
    return c + np.outer(ts, u) + np.outer(np.interp(ts, ti, ai), v) + np.outer(np.interp(ts, ti, bi), w3)


print("수동 라벨 읽는 중...")
CACHE = []
for f in sorted(glob.glob(os.path.join(MAN, '**', 'ascii', '*.ply'), recursive=True)):
    cond = os.path.basename(f).split('_label')[0]
    xyz, lk = read_ply_ascii(f)
    ih = np.where(lk == 1)[0]; iv = np.where(lk == 2)[0]; ig = np.where(lk == 0)[0]
    if len(ih) < 100 or len(iv) < 100: continue
    mm = xyz * 1000.0
    dv, _ = cKDTree(mm[ih]).query(mm[iv], k=1)
    dh, _ = cKDTree(mm[iv]).query(mm[ih], k=1)
    spec = np.concatenate([ih, iv])
    lo = mm[spec].min(0) - MARGIN_MM; hi = mm[spec].max(0) + MARGIN_MM
    g = mm[ig]
    igc = ig[np.all((g >= lo) & (g <= hi), axis=1)]
    CACHE.append(dict(cond=cond, D=cond.split('_')[1], A=cond.split('_')[3],
                      xyz=xyz, mm=mm, lk=lk, ih=ih, iv=iv, ig=igc, dv=dv, dh=dh))
print(f"  {len(CACHE)}건 (배경은 시편 경계 + {MARGIN_MM:g} mm 로 크롭)\n")

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
ST = torch.load(os.path.join(OUT, 'ptv3_3cls_norgb_8192_lovasz.pt'), map_location='cuda', weights_only=False)
FEAT = ST['feat']


def sample(a, seed=None):
    n = len(a)
    rng = np.random.default_rng(seed) if seed is not None else np.random
    idx = rng.choice(n, NPTS, replace=False) if n >= NPTS else \
        np.concatenate([np.arange(n), rng.choice(n, NPTS - n, replace=True)])
    xyz = a[idx, :3].astype(np.float32); lab = a[idx, 6].astype(np.int64)
    c = xyz.mean(0, keepdims=True); q = xyz - c
    q = q / (np.max(np.linalg.norm(q, axis=1)) + 1e-8)
    h = (xyz[:, 1:2] - xyz[:, 1].min()).astype(np.float32)
    oh = np.zeros((NPTS, 5), np.float32); oh[:, CATS.index('Tee')] = 1
    return q, np.concatenate([q, oh, h], 1).astype(np.float32), lab


def lovasz(logit, y):
    p = torch.softmax(logit, dim=-1); losses = []
    for c in range(NCLS):
        fg = (y == c).float()
        if fg.sum() == 0: continue
        err = (fg - p[:, c]).abs()
        e_s, perm = torch.sort(err, 0, descending=True)
        fg_s = fg[perm]; gts = fg_s.sum()
        jac = 1.0 - (gts - fg_s.cumsum(0)) / (gts + (1 - fg_s).cumsum(0))
        if len(jac) > 1: jac[1:] = jac[1:] - jac[:-1]
        losses.append(torch.dot(e_s, jac))
    return sum(losses) / max(len(losses), 1)


summary = []
for T in THRS:
    print(f"{'='*66}\n접촉 문턱 {T:g} mm (배경 크롭 + 두께 균형)\n{'='*66}")
    ds = {'train': [], 'val': [], 'test': []}
    ns, hws, rss = [], [], []
    byD = {}
    for c in CACHE:
        iw = np.concatenate([c['iv'][c['dv'] < T], c['ih'][c['dh'] < T]])
        wset = np.zeros(len(c['lk']), bool); wset[iw] = True
        ib = np.concatenate([c['ih'], c['iv']]); ib = ib[~wset[ib]]
        rng = np.random.default_rng(SEED)
        nb = int(np.clip(len(iw) * BASE_PER_WELD, MIN_BASE, MAX_BASE)); nb = min(nb, len(ib))
        sb = rng.choice(ib, nb, replace=False) if nb > 0 else np.empty(0, int)
        ng = min(len(c['ig']), int((len(iw) + nb) * BG_RATIO))
        sg = rng.choice(c['ig'], ng, replace=False) if ng > 0 else np.empty(0, int)
        keep = np.concatenate([iw, sb, sg]); rng.shuffle(keep)
        lab = np.full(len(c['lk']), 1, np.int64); lab[wset] = 0; lab[c['lk'] == 0] = 2
        a = np.zeros((len(keep), 7), np.float32)
        a[:, :3] = c['xyz'][keep]; a[:, 6] = lab[keep]
        sp = 'test' if c['A'] in TEST_A else ('val' if c['A'] in VAL_A else 'train')
        ds[sp].append((c['cond'], c['D'], a))
        ns.append(len(iw)); byD.setdefault(c['D'], []).append(len(iw))
        W = c['mm'][iw]
        if len(W) >= MIN_PTS:
            cp = path_bins(W)
            if cp is not None:
                d, _ = cKDTree(cp).query(W, k=1)
                hws.append(float(np.median(d))); rss.append(float(d.mean()))
    ns = np.array(ns)
    print(f"  점 수 중앙값 {int(np.median(ns)):,}  100점 미만 {int((ns<100).sum())}건  "
          f"밴드 반폭 {np.median(hws):.2f} mm  라벨 잔차 {np.median(rss):.2f} mm")

    idx = [i for i in range(len(ds['train'])) for _ in range(REP.get(ds['train'][i][1], 1))]
    bb = PointTransformerV3(
        in_channels=FEAT, order=("z", "z-trans"), stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(48,) * 5,
        dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16), dec_patch_size=(48,) * 4,
        mlp_ratio=4, qkv_bias=True, enable_flash=False,
        upcast_attention=True, upcast_softmax=True, enc_mode=False).cuda()
    hd = nn.Linear(64, NCLS).cuda()
    net = nn.ModuleList([bb, hd]); net.load_state_dict(ST['model'])

    def fwd(q, f):
        B, N, _ = q.shape
        p = bb(dict(coord=q.reshape(-1, 3).contiguous(), feat=f.reshape(-1, FEAT).contiguous(),
                    grid_size=0.01, offset=torch.arange(1, B + 1, device=q.device, dtype=torch.long) * N))
        return hd(p.feat).view(B, N, NCLS)

    def ev(items):
        net.eval(); acc = np.zeros((NCLS, 3), np.int64); per = []
        with torch.no_grad():
            for cond, D, a in items:
                q, f, y = sample(a, EVAL_SEED)
                o = fwd(torch.from_numpy(q).unsqueeze(0).cuda(), torch.from_numpy(f).unsqueeze(0).cuda())
                pr = o[0].argmax(-1).cpu().numpy()
                for cc in range(NCLS):
                    acc[cc, 0] += int(((pr == cc) & (y == cc)).sum())
                    acc[cc, 1] += int((pr == cc).sum()); acc[cc, 2] += int((y == cc).sum())
                it = int(((pr == 0) & (y == 0)).sum()); un = int(((pr == 0) | (y == 0)).sum())
                per.append(dict(조건=cond, D=D, IoU=round(it / max(un, 1) * 100, 2)))
        iou = [acc[c, 0] / max(acc[c, 1] + acc[c, 2] - acc[c, 0], 1) * 100 for c in range(NCLS)]
        return iou, float(np.mean(iou)), per

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = (-1, None); t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        rng = np.random.default_rng(SEED * 1000 + ep)
        order = np.array(idx)[rng.permutation(len(idx))]
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            if len(b) < 2: continue
            qs, fs, ys = [], [], []
            for i in b:
                q, f, y = sample(ds['train'][i][2]); qs.append(q); fs.append(f); ys.append(y)
            o = fwd(torch.from_numpy(np.stack(qs)).cuda(),
                    torch.from_numpy(np.stack(fs)).cuda()).reshape(-1, NCLS)
            yy = torch.from_numpy(np.concatenate(ys)).cuda()
            loss = ce(o, yy) + lovasz(o, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 20 == 0:
            iv_, mv, _ = ev(ds['val'])
            if iv_[0] > best[0]:
                best = (iv_[0], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            print(f"    epoch {ep:>3d}  검증 용접선 IoU {iv_[0]:6.2f} %")
    if best[1] is not None: net.load_state_dict(best[1])
    it_, mt, per = ev(ds['test'])
    perD = {}
    for p in per: perD.setdefault(p['D'], []).append(p['IoU'])
    print(f"  학습 {time.time()-t0:.0f}초  시험 용접선 IoU {it_[0]:.2f} %  mIoU {mt:.2f} %  " +
          '  '.join(f"{D} {np.median(v):.1f}%" for D, v in sorted(perD.items())))
    torch.save(dict(model=net.state_dict(), feat=FEAT, thr=T, crop=MARGIN_MM),
               os.path.join(OUT, f'kmou_final_{T:g}mm.pt'))
    summary.append(dict(문턱=T, 점수중앙=int(np.median(ns)),
                        D04점=int(np.median(byD['D04'])), D08점=int(np.median(byD['D08'])),
                        D12점=int(np.median(byD['D12'])), 미만100=int((ns < 100).sum()),
                        반폭=round(float(np.median(hws)), 2), 잔차=round(float(np.median(rss)), 2),
                        시험IoU=round(it_[0], 2), 시험mIoU=round(mt, 2),
                        D04=round(float(np.median(perD['D04'])), 1),
                        D08=round(float(np.median(perD['D08'])), 1),
                        D12=round(float(np.median(perD['D12'])), 1)))
    del net, bb, hd; torch.cuda.empty_cache()

print(f"\n{'='*104}\n=== 배경 크롭 + 두께 균형 조건에서의 문턱값 비교 ===\n{'='*104}")
print(f"{'문턱':>6s} {'점수(D04/D08/D12)':>24s} {'100미만':>8s} {'반폭':>8s} {'잔차':>8s} "
      f"{'시험IoU':>9s} {'mIoU':>8s} {'D04':>7s} {'D08':>7s} {'D12':>7s}")
for s in summary:
    print(f"{s['문턱']:>5.0f}mm {s['D04점']:>8,} /{s['D08점']:>6,} /{s['D12점']:>5,} {s['미만100']:>7d}건 "
          f"{s['반폭']:>7.2f}mm {s['잔차']:>7.2f}mm {s['시험IoU']:>8.2f}% {s['시험mIoU']:>7.2f}% "
          f"{s['D04']:>6.1f}% {s['D08']:>6.1f}% {s['D12']:>6.1f}%")
print("\n배경 크롭 전 같은 표 (참고)")
print("   3mm   917 /  210 /    0   21건   1.37mm  1.41mm   25.18%          37.7%   4.4%   0.0%")
print("   4mm 1,250 /  269 /  101   10건   1.82mm  1.88mm   26.83%          41.5%  11.8%   0.8%")
print("   5mm 1,518 /  340 /  138    0건   2.25mm  2.33mm   36.14%          51.2%  14.5%   5.4%")
print("   6mm 1,898 /  482 /  163    0건   2.70mm  2.78mm   37.51%          59.6%  15.4%   2.3%")

with open(os.path.join(RES, '해양대66_문턱값_비교_최종.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
print('\n저장: 해양대66_문턱값_비교_최종.csv')
