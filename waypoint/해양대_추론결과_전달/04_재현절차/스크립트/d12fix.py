# -*- coding: utf-8 -*-
"""
D12 가 안 되는 원인을 밴드 폭 말고 다른 데서 찾는다.

의심 1 — 배경을 사무실 전체에서 뽑고 있었다
  전처리에서 배경 점을 원본 점군 전체(2.9 m 범위)에서 무작위 추출했다.
  그러면 모델에 들어가는 점군의 크기가 시편(309 mm)이 아니라 사무실이 된다.
  PTv3 는 정규화 좌표 기준 grid_size 0.01 로 복셀을 나누므로,
  범위가 2.9 m 면 복셀이 약 14 mm 가 되어 판 두께 12 mm 보다 커진다.
  얇은 접촉부가 복셀 하나에 뭉개진다. 높이 특징도 배경 범위에 끌려간다.
  대책: 배경을 시편 주변으로만 자른다. 실제 운용에서 2D 마스크로 자르는 것과 같은 조건.

의심 2 — 두께별 표본 불균형
  D04 는 시편당 용접선 점이 1,500개, D12 는 138개다. 손실이 D04 에 지배된다.
  대책: 학습 순서에서 D08·D12 를 반복해 세 두께의 기여를 맞춘다.

세 조건을 같은 학습 설정으로 비교한다. 접촉 문턱은 5 mm 로 고정한다.
  A. 현행        : 배경을 전체에서 추출
  B. 배경 크롭    : 시편 경계 + 여유 80 mm 안의 배경만
  C. B + 두께 균형 : 배경 크롭에 두께별 반복 추가
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
EPOCHS, LR, BATCH = 60, 6e-5, 4
CONTACT, MARGIN_MM = 5.0, 80.0
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
    CACHE.append(dict(cond=cond, D=cond.split('_')[1], A=cond.split('_')[3],
                      xyz=xyz, mm=mm, lk=lk, ih=ih, iv=iv, ig=ig, dv=dv, dh=dh))
print(f"  {len(CACHE)}건\n")


def make(c, crop):
    iw = np.concatenate([c['iv'][c['dv'] < CONTACT], c['ih'][c['dh'] < CONTACT]])
    wset = np.zeros(len(c['lk']), bool); wset[iw] = True
    ib = np.concatenate([c['ih'], c['iv']]); ib = ib[~wset[ib]]
    ig = c['ig']
    if crop:
        spec = np.concatenate([c['ih'], c['iv']])
        lo = c['mm'][spec].min(0) - MARGIN_MM
        hi = c['mm'][spec].max(0) + MARGIN_MM
        g = c['mm'][ig]
        ig = ig[np.all((g >= lo) & (g <= hi), axis=1)]
    rng = np.random.default_rng(SEED)
    nb = int(np.clip(len(iw) * BASE_PER_WELD, MIN_BASE, MAX_BASE)); nb = min(nb, len(ib))
    sb = rng.choice(ib, nb, replace=False) if nb > 0 else np.empty(0, int)
    ng = min(len(ig), int((len(iw) + nb) * BG_RATIO))
    sg = rng.choice(ig, ng, replace=False) if ng > 0 else np.empty(0, int)
    keep = np.concatenate([iw, sb, sg]); rng.shuffle(keep)
    lab = np.full(len(c['lk']), 1, np.int64); lab[wset] = 0; lab[c['lk'] == 0] = 2
    a = np.zeros((len(keep), 7), np.float32)
    a[:, :3] = c['xyz'][keep]; a[:, 6] = lab[keep]
    return a, len(iw), (c['mm'][keep].max(0) - c['mm'][keep].min(0)).max()


print("=== 점군 크기 (모델에 들어가는 범위) ===")
for crop, nm in ((False, '현행 (배경 전체)'), (True, '배경 크롭')):
    sp = [make(c, crop)[2] for c in CACHE]
    vox = np.median(sp) / 2 * 0.01 * 2      # 정규화 반경 x grid_size, 대략적인 복셀 크기
    print(f"  {nm:>16s}  최대 변 길이 중앙값 {np.median(sp):>7.0f} mm  ->  복셀 약 {vox:>5.1f} mm")
print("  PTv3 는 정규화 좌표 기준 grid_size 0.01 을 쓴다. 범위가 크면 복셀도 커진다.\n")

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


RUNS = [('A', '현행 (배경 전체)', False, False),
        ('B', '배경 크롭', True, False),
        ('C', '배경 크롭 + 두께 균형', True, True)]
summary = []
for tag, nm, crop, bal in RUNS:
    print(f"{'='*66}\n{tag}. {nm}\n{'='*66}")
    ds = {'train': [], 'val': [], 'test': []}
    for c in CACHE:
        a, nw, _ = make(c, crop)
        sp = 'test' if c['A'] in TEST_A else ('val' if c['A'] in VAL_A else 'train')
        ds[sp].append((c['cond'], c['D'], a))

    idx = list(range(len(ds['train'])))
    if bal:
        cnt = {}
        for i in idx: cnt[ds['train'][i][1]] = cnt.get(ds['train'][i][1], 0) + 1
        rep = {'D04': 1, 'D08': 2, 'D12': 3}
        idx = [i for i in idx for _ in range(rep.get(ds['train'][i][1], 1))]
        print(f"  두께 균형: 학습 표본 {len(ds['train'])} -> {len(idx)}건 (D08 2배, D12 3배)")

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
        if ep % 15 == 0:
            iv_, mv, _ = ev(ds['val'])
            if iv_[0] > best[0]:
                best = (iv_[0], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            print(f"    epoch {ep:>3d}  검증 용접선 IoU {iv_[0]:6.2f} %  mIoU {mv:6.2f} %")
    if best[1] is not None: net.load_state_dict(best[1])
    it_, mt, per = ev(ds['test'])
    perD = {}
    for p in per: perD.setdefault(p['D'], []).append(p['IoU'])
    print(f"  학습 {time.time()-t0:.0f}초  시험 용접선 IoU {it_[0]:.2f} %  mIoU {mt:.2f} %")
    print("    " + '  '.join(f"{D} {np.median(v):.1f}%" for D, v in sorted(perD.items())))
    torch.save(dict(model=net.state_dict(), feat=FEAT, tag=tag),
               os.path.join(OUT, f'kmou_d12_{tag}.pt'))
    summary.append(dict(조건=tag, 설명=nm, 시험IoU=round(it_[0], 2), 시험mIoU=round(mt, 2),
                        D04=round(float(np.median(perD['D04'])), 1),
                        D08=round(float(np.median(perD['D08'])), 1),
                        D12=round(float(np.median(perD['D12'])), 1)))
    del net, bb, hd; torch.cuda.empty_cache()

print(f"\n{'='*80}\n=== D12 개선 시험 (접촉 5 mm 고정) ===\n{'='*80}")
print(f"{'조건':>4s} {'설명':>22s} {'시험IoU':>9s} {'mIoU':>8s} {'D04':>8s} {'D08':>8s} {'D12':>8s}")
for s in summary:
    print(f"{s['조건']:>4s} {s['설명']:>22s} {s['시험IoU']:>8.2f}% {s['시험mIoU']:>7.2f}% "
          f"{s['D04']:>7.1f}% {s['D08']:>7.1f}% {s['D12']:>7.1f}%")
with open(os.path.join(RES, '해양대66_D12개선.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
print('\n저장: 해양대66_D12개선.csv')
