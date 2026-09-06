# -*- coding: utf-8 -*-
"""
75.77 % 에서 더 올릴 수 있는가 — 한 번에 하나씩만 바꿔 각 지렛대의 몫을 잰다.

기준 (지금까지 최고)
  접촉 5 mm, 배경 크롭 80 mm, grid_size 0.01, 입력 8,192점, 두께 균형
  시험 용접선 IoU 75.77 %  (D04 92.4 / D08 70.2 / D12 48.3 %)

시험할 것
  G50  grid_size 0.005      복셀을 절반으로. 배경 크롭이 컸던 이유가 복셀이었으므로 더 줄여 본다
  G75  grid_size 0.0075     중간값
  CROP 배경 여유 80 -> 30 mm  범위가 줄면 복셀도 같이 줄어든다
  ROT  회전 증강            수평판 법선을 축으로 임의 회전. 촬영 각도 60~120도 공백을 메운다
  N16  입력 16,384점         얇은 접촉 밴드를 더 촘촘히 뽑는다

각 조건은 기준에서 딱 하나만 다르다. 학습 설정(60 epoch, lr 6e-5, batch 4, 시드 42)은 동일.
"""
import os, sys, glob, io, csv, json, time
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PNX = "/work/PointNeXt"; sys.path.insert(0, PNX); sys.path.insert(0, "/opt/Pointcept")

MAN = '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable'
OUT, RES = '/out', '/res'
CATS = ["Butt", "Corner", "Edge", "Lap", "Tee"]
NCLS, EVAL_SEED, SEED = 3, 42, 42
EPOCHS, LR, BATCH = 60, 6e-5, 4
CONTACT = 5.0
MAX_BASE, BASE_PER_WELD, MIN_BASE, BG_RATIO = 6000, 12, 1500, 0.30
TEST_A, VAL_A = {'A040', 'A150'}, {'A020'}
REP = {'D04': 1, 'D08': 2, 'D12': 3}


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
    H = mm[ih]; c0 = H.mean(0)
    _, _, Vt = np.linalg.svd(H - c0, full_matrices=False)
    CACHE.append(dict(cond=cond, D=cond.split('_')[1], A=cond.split('_')[3],
                      xyz=xyz, mm=mm, lk=lk, ih=ih, iv=iv, ig=ig, dv=dv, dh=dh,
                      axis=Vt[2] / np.linalg.norm(Vt[2])))
print(f"  {len(CACHE)}건\n")


def build(margin):
    ds = {'train': [], 'val': [], 'test': []}
    for c in CACHE:
        iw = np.concatenate([c['iv'][c['dv'] < CONTACT], c['ih'][c['dh'] < CONTACT]])
        wset = np.zeros(len(c['lk']), bool); wset[iw] = True
        ib = np.concatenate([c['ih'], c['iv']]); ib = ib[~wset[ib]]
        spec = np.concatenate([c['ih'], c['iv']])
        lo = c['mm'][spec].min(0) - margin; hi = c['mm'][spec].max(0) + margin
        g = c['mm'][c['ig']]
        ig = c['ig'][np.all((g >= lo) & (g <= hi), axis=1)]
        rng = np.random.default_rng(SEED)
        nb = int(np.clip(len(iw) * BASE_PER_WELD, MIN_BASE, MAX_BASE)); nb = min(nb, len(ib))
        sb = rng.choice(ib, nb, replace=False) if nb > 0 else np.empty(0, int)
        ng = min(len(ig), int((len(iw) + nb) * BG_RATIO))
        sg = rng.choice(ig, ng, replace=False) if ng > 0 else np.empty(0, int)
        keep = np.concatenate([iw, sb, sg]); rng.shuffle(keep)
        lab = np.full(len(c['lk']), 1, np.int64); lab[wset] = 0; lab[c['lk'] == 0] = 2
        a = np.zeros((len(keep), 7), np.float32)
        a[:, :3] = c['xyz'][keep]; a[:, 6] = lab[keep]
        sp = 'test' if c['A'] in TEST_A else ('val' if c['A'] in VAL_A else 'train')
        ds[sp].append((c['cond'], c['D'], a, c['axis']))
    return ds


def rotmat(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
ST = torch.load(os.path.join(OUT, 'ptv3_3cls_norgb_8192_lovasz.pt'), map_location='cuda', weights_only=False)
FEAT = ST['feat']


def sample(a, npts, seed=None, axis=None, rot=False):
    n = len(a)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    idx = rng.choice(n, npts, replace=False) if n >= npts else \
        np.concatenate([np.arange(n), rng.choice(n, npts - n, replace=True)])
    xyz = a[idx, :3].astype(np.float64); lab = a[idx, 6].astype(np.int64)
    if rot and axis is not None:
        R = rotmat(axis, rng.uniform(0, 2 * np.pi))
        c = xyz.mean(0, keepdims=True)
        xyz = (xyz - c) @ R.T + c
    xyz = xyz.astype(np.float32)
    c = xyz.mean(0, keepdims=True); q = xyz - c
    q = q / (np.max(np.linalg.norm(q, axis=1)) + 1e-8)
    h = (xyz[:, 1:2] - xyz[:, 1].min()).astype(np.float32)
    oh = np.zeros((npts, 5), np.float32); oh[:, CATS.index('Tee')] = 1
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


RUNS = [
    ('G75',  'grid 0.0075',      dict(grid=0.0075)),
    ('G50',  'grid 0.005',       dict(grid=0.005)),
    ('CROP', '배경 여유 30 mm',   dict(margin=30.0)),
    ('ROT',  '회전 증강',         dict(rot=True)),
    ('N16',  '입력 16,384점',     dict(npts=16384, batch=2)),
]
summary = []
for tag, nm, cfg in RUNS:
    grid = cfg.get('grid', 0.01); margin = cfg.get('margin', 80.0)
    npts = cfg.get('npts', 8192); batch = cfg.get('batch', BATCH); rot = cfg.get('rot', False)
    print(f"{'='*66}\n{tag}  {nm}\n{'='*66}")
    ds = build(margin)
    ex = np.median([np.ptp(a[:, :3], axis=0).max() * 1000 for _, _, a, _ in ds['train']])
    print(f"  점군 최대 변 {ex:.0f} mm,  복셀 약 {ex/2*grid*2:.2f} mm,  입력 {npts:,}점")
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
                    grid_size=grid, offset=torch.arange(1, B + 1, device=q.device, dtype=torch.long) * N))
        return hd(p.feat).view(B, N, NCLS)

    def ev(items):
        net.eval(); acc = np.zeros((NCLS, 3), np.int64); per = []
        with torch.no_grad():
            for cond, D, a, ax in items:
                q, f, y = sample(a, npts, EVAL_SEED)
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
    try:
        for ep in range(1, EPOCHS + 1):
            net.train()
            rng = np.random.default_rng(SEED * 1000 + ep)
            order = np.array(idx)[rng.permutation(len(idx))]
            for s in range(0, len(order), batch):
                b = order[s:s + batch]
                if len(b) < 2: continue
                qs, fs, ys = [], [], []
                for i in b:
                    _, _, a, ax = ds['train'][i]
                    q, f, y = sample(a, npts, axis=ax, rot=rot)
                    qs.append(q); fs.append(f); ys.append(y)
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
                print(f"    epoch {ep:>3d}  검증 {iv_[0]:6.2f} %")
    except torch.cuda.OutOfMemoryError:
        print("    [GPU 메모리 부족] 이 조건은 건너뜀")
        del net, bb, hd; torch.cuda.empty_cache(); continue
    if best[1] is not None: net.load_state_dict(best[1])
    it_, mt, per = ev(ds['test'])
    perD = {}
    for p in per: perD.setdefault(p['D'], []).append(p['IoU'])
    print(f"  학습 {time.time()-t0:.0f}초  시험 {it_[0]:.2f} %  mIoU {mt:.2f} %  " +
          '  '.join(f"{D} {np.median(v):.1f}%" for D, v in sorted(perD.items())))
    torch.save(dict(model=net.state_dict(), feat=FEAT, tag=tag, grid=grid,
                    margin=margin, npts=npts, rot=rot), os.path.join(OUT, f'kmou_imp_{tag}.pt'))
    summary.append(dict(조건=tag, 설명=nm, 시험IoU=round(it_[0], 2), mIoU=round(mt, 2),
                        D04=round(float(np.median(perD['D04'])), 1),
                        D08=round(float(np.median(perD['D08'])), 1),
                        D12=round(float(np.median(perD['D12'])), 1)))
    del net, bb, hd; torch.cuda.empty_cache()

print(f"\n{'='*84}\n=== 개선 시험 (기준: 접촉 5 mm, 크롭 80 mm, grid 0.01, 8,192점, 두께 균형) ===\n{'='*84}")
print(f"{'조건':>6s} {'설명':>18s} {'시험IoU':>9s} {'mIoU':>8s} {'D04':>7s} {'D08':>7s} {'D12':>7s}")
print(f"{'기준':>6s} {'-':>18s} {'75.77':>8s}% {'88.95':>7s}% {'92.4':>6s}% {'70.2':>6s}% {'48.3':>6s}%")
for s in summary:
    print(f"{s['조건']:>6s} {s['설명']:>18s} {s['시험IoU']:>8.2f}% {s['mIoU']:>7.2f}% "
          f"{s['D04']:>6.1f}% {s['D08']:>6.1f}% {s['D12']:>6.1f}%")
with open(os.path.join(RES, '해양대66_개선시험.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
print('\n저장: 해양대66_개선시험.csv')
