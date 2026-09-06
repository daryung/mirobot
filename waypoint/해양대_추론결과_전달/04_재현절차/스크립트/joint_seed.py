# -*- coding: utf-8 -*-
"""
개선안 조합 + NIA 합동 학습

1차에서 각각 남은 것
  N16  입력 16,384점   78.15 %   (D12 54.4 % 로 가장 크게 기여)
  G75  grid 0.0075     77.80 %
  CROP 배경 여유 30 mm  76.01 %
  ROT  회전 증강        71.73 %  기각

G75 와 CROP 은 둘 다 복셀을 줄이는 같은 지렛대라 겹칠 수 있다. N16 은 다른 축이다.

합동 학습
  NIA 1,068건과 해양대 46건을 함께 학습한다. NIA 쪽 클래스 0 은 용접비드,
  해양대 쪽 클래스 0 은 가접 상태 용접선이다. 같은 자리이나 형상이 다르다.
  먼저 0 번을 하나로 합쳐 "용접해야 할 곳" 으로 학습해 본다. 구조가 그대로라
  기존 가중치를 온전히 잇는다. 표본은 23 대 1 이므로 해양대를 반복해 맞춘다.
  평가는 양쪽 시험셋 모두에서 한다. 한쪽만 오르고 다른 쪽이 무너지면 통합 불가다.
"""
import os, sys, glob, io, csv, json, time, argparse
import numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PNX = "/work/PointNeXt"; sys.path.insert(0, PNX); sys.path.insert(0, "/opt/Pointcept")

MAN = '/kmou/04_원본_및_수동라벨/08.10 데이터 셋(생기원)/08.10 데이터 셋(생기원)/수동 라벨링/3D_lable'
NIA = '/work/PointNeXt/data/NIA_Welding3Cls'
OUT, RES = '/out', '/res'
CATS = ["Butt", "Corner", "Edge", "Lap", "Tee"]
_ap = argparse.ArgumentParser(); _ap.add_argument('--seed', type=int, default=42); _args = _ap.parse_args()
NCLS, EVAL_SEED, SEED = 3, 42, _args.seed
EPOCHS, LR = 60, 6e-5
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


print("해양대 수동 라벨 읽는 중...")
CACHE = []
for f in sorted(glob.glob(os.path.join(MAN, '**', 'ascii', '*.ply'), recursive=True)):
    cond = os.path.basename(f).split('_label')[0]
    xyz, lk = read_ply_ascii(f)
    ih = np.where(lk == 1)[0]; iv = np.where(lk == 2)[0]; ig = np.where(lk == 0)[0]
    if len(ih) < 100 or len(iv) < 100: continue
    mm = xyz * 1000.0
    dv, _ = cKDTree(mm[ih]).query(mm[iv], k=1)
    dh, _ = cKDTree(mm[iv]).query(mm[ih], k=1)
    CACHE.append(dict(cond=cond, D=cond.split('_')[1], A=cond.split('_')[3],
                      xyz=xyz, mm=mm, lk=lk, ih=ih, iv=iv, ig=ig, dv=dv, dh=dh))
print(f"  {len(CACHE)}건")

print("NIA 3클래스 읽는 중...")
NIAD = {}
for sp in ('train', 'test'):
    fs = sorted(glob.glob(os.path.join(NIA, sp, '*.npy')))
    NIAD[sp] = [(os.path.basename(x)[:-4], np.load(x)) for x in fs]
    print(f"  {sp} {len(NIAD[sp])}건")
print()


def build_kmou(margin):
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
        ds[sp].append((c['cond'], c['D'], a))
    return ds


from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
ST = torch.load(os.path.join(OUT, 'ptv3_3cls_norgb_8192_lovasz.pt'), map_location='cuda', weights_only=False)
FEAT = ST['feat']


def sample(a, npts, seed=None, cat='Tee'):
    n = len(a)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    idx = rng.choice(n, npts, replace=False) if n >= npts else \
        np.concatenate([np.arange(n), rng.choice(n, npts - n, replace=True)])
    xyz = a[idx, :3].astype(np.float32); lab = a[idx, 6].astype(np.int64)
    c = xyz.mean(0, keepdims=True); q = xyz - c
    q = q / (np.max(np.linalg.norm(q, axis=1)) + 1e-8)
    h = (xyz[:, 1:2] - xyz[:, 1].min()).astype(np.float32)
    oh = np.zeros((npts, 5), np.float32); oh[:, CATS.index(cat)] = 1
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


RUNS = [(f'합동_시드{SEED}', dict(npts=8192, batch=4, grid=0.01, margin=80.0, joint=True))]
summary = []
for tag, cfg in RUNS:
    npts, batch = cfg['npts'], cfg['batch']
    grid, margin, joint = cfg['grid'], cfg['margin'], cfg.get('joint', False)
    print(f"{'='*68}\n{tag}\n{'='*68}")
    ds = build_kmou(margin)
    items = [(c, D, a, 'Tee') for c, D, a in ds['train']]
    idx = [i for i in range(len(items)) for _ in range(REP.get(items[i][1], 1))]
    if joint:
        n0 = len(items)
        for nm, a in NIAD['train']:
            items.append((nm, 'NIA', a, nm.split('_')[0]))
        # 해양대를 NIA 와 비슷한 규모로 반복
        mult = max(1, len(NIAD['train']) // max(len(idx), 1))
        idx = [i for i in idx for _ in range(mult)] + list(range(n0, len(items)))
        print(f"  해양대 {n0}건(반복 {mult}배 -> {len(idx)-len(NIAD['train'])}) + NIA {len(NIAD['train'])}건")

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

    def ev(lst, cat=None, np_=None):
        net.eval(); acc = np.zeros((NCLS, 3), np.int64); per = []
        with torch.no_grad():
            for it in lst:
                if len(it) == 3: cond, D, a = it; cc = cat or 'Tee'
                else: cond, a = it; D = 'NIA'; cc = cond.split('_')[0]
                q, f, y = sample(a, np_ or npts, EVAL_SEED, cc)
                o = fwd(torch.from_numpy(q).unsqueeze(0).cuda(), torch.from_numpy(f).unsqueeze(0).cuda())
                pr = o[0].argmax(-1).cpu().numpy()
                for k in range(NCLS):
                    acc[k, 0] += int(((pr == k) & (y == k)).sum())
                    acc[k, 1] += int((pr == k).sum()); acc[k, 2] += int((y == k).sum())
                it_ = int(((pr == 0) & (y == 0)).sum()); un = int(((pr == 0) | (y == 0)).sum())
                per.append(dict(조건=cond, D=D, IoU=round(it_ / max(un, 1) * 100, 2)))
        iou = [acc[k, 0] / max(acc[k, 1] + acc[k, 2] - acc[k, 0], 1) * 100 for k in range(NCLS)]
        return iou, float(np.mean(iou)), per

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = (-1, None); t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        rng = np.random.default_rng(SEED * 1000 + ep)
        order = np.array(idx)[rng.permutation(len(idx))]
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            if len(b) < 2: continue
            qs, fs, ys = [], [], []
            for i in b:
                _, _, a, cc = items[i]
                q, f, y = sample(a, npts, cat=cc); qs.append(q); fs.append(f); ys.append(y)
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
            print(f"    epoch {ep:>3d}  해양대 검증 {iv_[0]:6.2f} %")
    if best[1] is not None: net.load_state_dict(best[1])
    it_, mt, per = ev(ds['test'])
    perD = {}
    for p in per: perD.setdefault(p['D'], []).append(p['IoU'])
    # NIA 시험셋에서의 성능 (망각 확인)
    ni, nm_, _ = ev(NIAD['test'], np_=8192)
    print(f"  학습 {time.time()-t0:.0f}초")
    print(f"  해양대 시험 용접선 {it_[0]:.2f} %  mIoU {mt:.2f} %  " +
          '  '.join(f"{D} {np.median(v):.1f}%" for D, v in sorted(perD.items())))
    print(f"  NIA 시험  용접비드 {ni[0]:.2f} %  mIoU {nm_:.2f} %   (원래 81.70 % / 92.64 %)")
    torch.save(dict(model=net.state_dict(), feat=FEAT, tag=tag, **cfg),
               os.path.join(OUT, f'kmou_cb_{tag.replace("+","_").replace("(","").replace(")","")}.pt'))
    summary.append(dict(조건=tag, 해양대IoU=round(it_[0], 2), 해양대mIoU=round(mt, 2),
                        D04=round(float(np.median(perD['D04'])), 1),
                        D08=round(float(np.median(perD['D08'])), 1),
                        D12=round(float(np.median(perD['D12'])), 1),
                        NIA비드IoU=round(ni[0], 2), NIAmIoU=round(nm_, 2)))
    del net, bb, hd; torch.cuda.empty_cache()

print(f"\n{'='*96}\n=== 조합 및 합동 학습 결과 ===\n{'='*96}")
print(f"{'조건':>16s} {'해양대IoU':>10s} {'mIoU':>8s} {'D04':>7s} {'D08':>7s} {'D12':>7s} "
      f"{'NIA비드':>9s} {'NIAmIoU':>9s}")
print(f"{'NIA 전용(기존)':>16s} {'0.00':>9s}% {'-':>7s}  {'-':>6s}  {'-':>6s}  {'-':>6s}  "
      f"{'81.70':>8s}% {'92.64':>8s}%")
print(f"{'기준(8192)':>16s} {'75.77':>9s}% {'88.95':>7s}% {'92.4':>6s}% {'70.2':>6s}% {'48.3':>6s}% "
      f"{'미측정':>8s} {'미측정':>8s}")
print(f"{'N16':>16s} {'78.15':>9s}% {'90.04':>7s}% {'92.7':>6s}% {'72.3':>6s}% {'54.4':>6s}% "
      f"{'미측정':>8s} {'미측정':>8s}")
for s in summary:
    print(f"{s['조건']:>16s} {s['해양대IoU']:>9.2f}% {s['해양대mIoU']:>7.2f}% {s['D04']:>6.1f}% "
          f"{s['D08']:>6.1f}% {s['D12']:>6.1f}% {s['NIA비드IoU']:>8.2f}% {s['NIAmIoU']:>8.2f}%")
with open(os.path.join(RES, f'해양대66_합동_시드{SEED}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
print('\n저장: 해양대66_조합_합동.csv')
