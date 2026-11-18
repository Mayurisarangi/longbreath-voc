import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def iipe(t, d):
    """Irregular Interval Positional Encoding: sinusoidal encoding of actual elapsed time (months)."""
    i = torch.arange(0, d, 2, device=t.device).float()
    ang = t[..., None] / (10000 ** (i / d)) * 3.0
    pe = torch.zeros(*t.shape, d, device=t.device); pe[..., 0::2] = torch.sin(ang); pe[..., 1::2] = torch.cos(ang[..., : d // 2])
    return pe


def intpe(t, d):  # integer positional index (ablation / vanilla transformer)
    idx = torch.round(t / 6.0)
    return iipe(idx * 6.0, d)


def last(h, lens):
    return h[torch.arange(h.size(0)), lens - 1]


class Head(nn.Module):
    def __init__(s, d, hid, p=0.3): super().__init__(); s.f = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, 3))
    def forward(s, z): return s.f(z)


class LSTMNet(nn.Module):
    def __init__(s, fin, h=64): super().__init__(); s.r = nn.LSTM(fin, h, 2, batch_first=True, dropout=0.1); s.h = Head(h, h)
    def forward(s, b): o, _ = s.r(b["x"]); return s.h(last(o, b["lens"]))


class DCell(nn.Module):
    """LSTM cell with learned time-distance decay of the memory (distanced-LSTM style)."""
    def __init__(s, fin, h): super().__init__(); s.c = nn.LSTMCell(fin, h); s.lam = nn.Parameter(torch.zeros(h))
    def forward(s, x, hc, dt): h, c = hc; c = c * torch.exp(-F.softplus(s.lam) * dt[:, None]); return s.c(x, (h, c))


class DLSTMNet(nn.Module):
    def __init__(s, fin, h=64): super().__init__(); s.c1 = DCell(fin, h); s.c2 = DCell(h, h); s.h = Head(h, h); s.H = h
    def forward(s, b):
        x, t = b["x"], b["t"]; N, T, _ = x.shape; z = x.new_zeros(N, s.H); st1 = (z, z); st2 = (z, z); outs = []
        for k in range(T):
            dt = (t[:, k] - t[:, k - 1]) / 12 if k else x.new_zeros(N)
            st1 = s.c1(x[:, k], st1, dt); st2 = s.c2(st1[0], st2, dt); outs.append(st2[0])
        return s.h(last(torch.stack(outs, 1), b["lens"]))


class GRUD(nn.Module):
    """GRU-D (Che et al. 2018): input and hidden-state decay driven by time since last observation, masking."""
    def __init__(s, J=10, h=64):
        super().__init__(); s.J, s.H = J, h
        s.gx = nn.Linear(J, J); s.gh = nn.Linear(J, h); s.cell = nn.GRUCell(2 * J, h); s.h = Head(h, h)
    def forward(s, b):
        v, m, dl, xl = b["v"], b["m"], b["delta"], b["xl"]; N, T, J = v.shape; h = v.new_zeros(N, s.H); outs = []
        for k in range(T):
            gx = torch.exp(-F.relu(s.gx(dl[:, k]))); gh = torch.exp(-F.relu(s.gh(dl[:, k])))
            x = m[:, k] * v[:, k] + (1 - m[:, k]) * gx * xl[:, k]
            h = s.cell(torch.cat([x, m[:, k]], 1), h * gh); outs.append(h)
        return s.h(last(torch.stack(outs, 1), b["lens"]))


class TT(nn.Module):
    """Pure temporal transformer with IIPE (no BiLSTM): [CLS] token + 2-layer encoder."""
    def __init__(s, fin, d=64, ff=128, pe=iipe):
        super().__init__(); s.p = nn.Linear(fin, d); s.d = d; s.pe = pe; s.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        s.e = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, ff, 0.1, batch_first=True), 2); s.h = Head(d, d)
    def forward(s, b):
        x = s.p(b["x"]) + s.pe(b["t"], s.d); N = x.size(0)
        ok = b["valid"] & b["obs"] if b["use_obs"] else b["valid"]
        x = torch.cat([s.cls.expand(N, -1, -1), x], 1); pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool), ~ok], 1)
        return s.h(s.e(x, src_key_padding_mask=pad)[:, 0])


class HTAN(nn.Module):
    """BiLSTM encoder + IIPE + [DRIFT]-token cross-attention transformer decoder + MLP head."""
    def __init__(s, fin, h=192, d=192, ff=384, layers=2, heads=4, hid=128, pe=iipe, use_bilstm=True, use_cross=True):
        super().__init__(); s.pe, s.d, s.ub, s.uc = pe, fin, use_bilstm, use_cross; s.pin = fin
        if use_bilstm:
            s.r = nn.LSTM(fin, h, 2, batch_first=True, bidirectional=True, dropout=0.1); enc = 2 * h
        else: enc = fin
        s.proj = nn.Linear(enc, d)
        if use_cross: s.q = nn.Parameter(torch.randn(1, 1, d) * .02)
        if use_cross: s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, 0.1, batch_first=True), layers)
        s.h = Head(d, hid)
        if not use_bilstm: s.tp = nn.Linear(fin, d)
    def forward(s, b):
        x = b["x"]; N, T, F_ = x.shape
        x = x + s.pe(b["t"], F_) if F_ % 2 == 0 else x + s.pe(b["t"], F_ + 1)[..., :F_]
        ok = b["valid"] & b["obs"] if b["use_obs"] else b["valid"]
        if s.ub:
            pk = nn.utils.rnn.pack_padded_sequence(x, b["lens"].cpu(), batch_first=True, enforce_sorted=False)
            o, _ = s.r(pk); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=T)
        else: o = x
        H = s.proj(o)
        if not s.uc:
            w = ok.float()[..., None]; z = (H * w).sum(1) / w.sum(1).clamp(min=1)
            return s.h(z)
        out = s.dec(s.q.expand(N, -1, -1), H, memory_key_padding_mask=~ok)
        s.last_attn = None
        return s.h(out[:, 0])
    @torch.no_grad()
    def attention(s, b):
        """cross-attention weights of the [DRIFT] token in the last decoder layer (mean over heads)."""
        x = b["x"]; N, T, F_ = x.shape; x = x + s.pe(b["t"], F_)
        pk = nn.utils.rnn.pack_padded_sequence(x, b["lens"].cpu(), batch_first=True, enforce_sorted=False)
        o, _ = s.r(pk); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=T)
        H = s.proj(o); ok = b["valid"]; tgt = s.q.expand(N, -1, -1)
        for l in s.dec.layers[:-1]: tgt = l(tgt, H, memory_key_padding_mask=~ok)
        l = s.dec.layers[-1]; t2 = l.norm1(tgt + l._sa_block(tgt, None, None, False)) if hasattr(l, "_sa_block") else tgt
        _, w = l.multihead_attn(t2, H, H, key_padding_mask=~ok, need_weights=True, average_attn_weights=True)
        return w[:, 0]


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make(name, fin):
    if name == "LSTM": return LSTMNet(fin)
    if name == "DLSTM": return DLSTMNet(fin)
    if name == "GRU-D": return GRUD()
    if name == "TT": return TT(fin)
    if name == "HTAN": return HTAN(fin)
    if name == "HTAN-small": return HTAN(fin, h=16, d=32, ff=64, hid=32)
    if name == "HTAN-noIIPE": return HTAN(fin, pe=intpe)
    if name == "HTAN-noBiLSTM": return HTAN(fin, use_bilstm=False)
    if name == "HTAN-noCross": return HTAN(fin, use_cross=False)
    raise ValueError(name)


def focal(logits, y, g=2.0):
    lp = F.log_softmax(logits, 1); p = lp.exp(); lpt = lp.gather(1, y[:, None])[:, 0]; pt = p.gather(1, y[:, None])[:, 0]
    return (-(1 - pt) ** g * lpt).mean()


def tens(b, idx, dev="cpu"):
    o = {}
    for k, v in b.items():
        if k == "use_obs": o[k] = v
        elif k == "lens": o[k] = torch.as_tensor(v[idx], dtype=torch.long)
        else: o[k] = torch.as_tensor(v[idx])
    return o


def train_predict(name, fin, trb, y_tr, vab, y_va, teb_list, seed=0, epochs=40, patience=10, lr=5e-4, bs=64):
    """trb: list of batches for prefix lengths (augmentation); vab: validation batch at full length; teb_list: batches to predict."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = make(name, fin); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # stack prefix batches
    def cat(bs_):
        o = {}
        for k in bs_[0]:
            if k == "use_obs": o[k] = bs_[0][k]
            else: o[k] = np.concatenate([b[k] for b in bs_])
        return o
    Xtr = cat(trb); ytr = np.concatenate([y_tr] * len(trb)); n = len(ytr); best, bad, state = 1e9, 0, None
    vb = tens(vab, np.arange(len(y_va))); yv = torch.as_tensor(y_va)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; b = tens(Xtr, idx); loss = focal(m(b), torch.as_tensor(ytr[idx]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step(); m.eval()
        with torch.no_grad(): vl = focal(m(vb), yv).item()
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); m.eval(); outs = []
    with torch.no_grad():
        for tb in teb_list:
            b = tens(tb, np.arange(len(tb["lens"]))); outs.append(F.softmax(m(b), 1).numpy())
    return outs, m, nparams(m), ep + 1
