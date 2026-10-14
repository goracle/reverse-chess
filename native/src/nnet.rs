//! rc_net -- CPU forward pass of `net.PolicyValueNet`, for the self-play workers.
//!
//! Same maths as `numpy_net.NumpyNet` / `net.PolicyValueNet.forward` (all f32):
//!   stem conv3x3 + GroupNorm + ReLU, residual blocks, value head (1x1 conv, FC, tanh) and the
//!   pointer policy head (per-square projections gathered per legal action).
//!
//! * The weights arrive as ONE flat f32 blob in a fixed order (see `Net::from_blob`; the Python
//!   side is rustnet.py) and are repacked here: conv weights into 16/8-wide output tiles, linear
//!   weights transposed to [in][out].
//! * Activations are NHWC with a zero border (10x10 pixels) so a 3x3 conv needs no edge cases.
//! * Convolution = register-tiled direct loops (4 pixels x 16 outputs live in 8 AVX2 registers).
//!   The whole forward is compiled twice: once for AVX2+FMA (picked at run time when the CPU has
//!   them) and once for plain x86-64 / other targets.  `RC_NET_SIMD=0` forces the plain one.
//! * A `Net` is immutable after construction and every call allocates its own scratch, so it
//!   may be shared between threads (play.py searches in a thread).
//!
//! C ABI: `rc_net_new`, `rc_net_free`, `rc_net_forward`, `rc_net_error`, `rc_net_simd`.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

const SQ: usize = 64; // squares
const PADW: usize = 10; // padded board width
const GROUPS: usize = 8; // GroupNorm groups
const EPS: f32 = 1e-5; // torch GroupNorm default
const GLOB_CH: usize = 8; // channels of the global 1x1 conv
const FLAT: usize = GLOB_CH * SQ; // its flattened size

thread_local! {
    static LAST_ERR: RefCell<String> = RefCell::new(String::new());
}

pub(crate) fn set_err(msg: String) {
    LAST_ERR.with(|e| *e.borrow_mut() = msg);
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

/// 3x3 conv, no bias, repacked as output tiles: tile weights are [tap 9][cin][lanes].
struct Conv {
    cin: usize,
    cout: usize,
    tiles: Vec<Tile>,
}

struct Tile {
    n0: usize,
    lanes: usize, // 16 or 8
    w: Vec<f32>,
}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Block {
    conv1: Conv,
    norm1: Norm,
    conv2: Conv,
    norm2: Norm,
}

pub struct Net {
    c: usize,      // channels
    h: usize,      // hidden
    hh: usize,     // hidden / 2
    n_gift: usize, // rows of gift_emb
    n_promo: usize,
    cin: usize, // planes + scalars
    planes: usize,
    n_scalars: usize,
    stem: Conv,
    stem_norm: Norm,
    blocks: Vec<Block>,
    glob_conv_wt: Vec<f32>, // [c][8]
    glob_conv_b: Vec<f32>,  // [8]
    glob_fc_wt: Vec<f32>,   // [512][h]
    glob_fc_b: Vec<f32>,
    value_w: Vec<f32>, // [h]
    value_b: f32,
    src_wt: Vec<f32>, // [c][h]
    src_b: Vec<f32>,
    dst_wt: Vec<f32>,       // [c][h]
    glob_proj_wt: Vec<f32>, // [h][h]
    gift_emb: Vec<f32>,     // [n_gift][h]
    promo_emb: Vec<f32>,
    flag_emb: Vec<f32>,   // [4][h]
    act_fc_wt: Vec<f32>,  // [h][hh]
    act_fc_b: Vec<f32>,   // [hh]
    act_out_w: Vec<f32>,  // [hh]
    act_out_b: f32,
}

struct Reader<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [f32], String> {
        if self.pos + n > self.data.len() {
            return Err(format!(
                "weight blob too short at {}: needs {} floats at offset {}, blob has {}",
                what,
                n,
                self.pos,
                self.data.len()
            ));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

/// torch Linear weight (out, in) row-major -> [in][out].
fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

impl Conv {
    /// `w` is a torch Conv2d weight (cout, cin, 3, 3).
    fn new(w: &[f32], cout: usize, cin: usize) -> Conv {
        let mut tiles = Vec::new();
        let mut n0 = 0;
        while n0 < cout {
            let lanes = if cout - n0 >= 16 { 16 } else { 8 };
            let mut pw = vec![0f32; 9 * cin * lanes];
            for tap in 0..9 {
                for ci in 0..cin {
                    for j in 0..lanes {
                        pw[(tap * cin + ci) * lanes + j] = w[((n0 + j) * cin + ci) * 9 + tap];
                    }
                }
            }
            tiles.push(Tile { n0, lanes, w: pw });
            n0 += lanes;
        }
        Conv { cin, cout, tiles }
    }
}

impl Net {
    /// Blob order (all f32, torch layouts), `C` channels, `H` hidden, `I` = planes + scalars:
    ///   stem.weight (C,I,3,3)  stem_norm.weight (C)  stem_norm.bias (C)
    ///   per block: conv1.weight (C,C,3,3) norm1.weight norm1.bias conv2.weight norm2.weight norm2.bias
    ///   glob_conv.weight (8,C)  glob_conv.bias (8)  glob_fc.weight (H,512)  glob_fc.bias (H)
    ///   value_out.weight (H)  value_out.bias (1)
    ///   src_proj.weight (H,C)  src_proj.bias (H)  dst_proj.weight (H,C)  glob_proj.weight (H,H)
    ///   gift_emb (G,H)  promo_emb (P,H)  flag_emb (4,H)
    ///   act_fc.weight (H/2,H)  act_fc.bias (H/2)  act_out.weight (H/2)  act_out.bias (1)
    #[allow(clippy::too_many_arguments)]
    fn from_blob(
        blob: &[f32],
        c: usize,
        blocks: usize,
        h: usize,
        n_gift: usize,
        n_promo: usize,
        planes: usize,
        n_scalars: usize,
    ) -> Result<Net, String> {
        if c == 0 || c % 8 != 0 || c % GROUPS != 0 {
            return Err(format!("channels must be a positive multiple of 8, got {}", c));
        }
        if h == 0 || h % 16 != 0 {
            return Err(format!("hidden must be a positive multiple of 16, got {}", h));
        }
        if n_gift == 0 || n_promo == 0 {
            return Err("embedding tables must not be empty".to_string());
        }
        let cin = planes + n_scalars;
        let hh = h / 2;
        let mut r = Reader { data: blob, pos: 0 };
        let stem = Conv::new(r.take(c * cin * 9, "stem.weight")?, c, cin);
        let stem_norm = Norm {
            gamma: r.take(c, "stem_norm.weight")?.to_vec(),
            beta: r.take(c, "stem_norm.bias")?.to_vec(),
        };
        let mut bl = Vec::new();
        for i in 0..blocks {
            let conv1 = Conv::new(r.take(c * c * 9, &format!("blocks.{}.conv1.weight", i))?, c, c);
            let norm1 = Norm {
                gamma: r.take(c, "norm1.weight")?.to_vec(),
                beta: r.take(c, "norm1.bias")?.to_vec(),
            };
            let conv2 = Conv::new(r.take(c * c * 9, &format!("blocks.{}.conv2.weight", i))?, c, c);
            let norm2 = Norm {
                gamma: r.take(c, "norm2.weight")?.to_vec(),
                beta: r.take(c, "norm2.bias")?.to_vec(),
            };
            bl.push(Block { conv1, norm1, conv2, norm2 });
        }
        let glob_conv_wt = transpose(r.take(GLOB_CH * c, "glob_conv.weight")?, GLOB_CH, c);
        let glob_conv_b = r.take(GLOB_CH, "glob_conv.bias")?.to_vec();
        let glob_fc_wt = transpose(r.take(h * FLAT, "glob_fc.weight")?, h, FLAT);
        let glob_fc_b = r.take(h, "glob_fc.bias")?.to_vec();
        let value_w = r.take(h, "value_out.weight")?.to_vec();
        let value_b = r.take(1, "value_out.bias")?[0];
        let src_wt = transpose(r.take(h * c, "src_proj.weight")?, h, c);
        let src_b = r.take(h, "src_proj.bias")?.to_vec();
        let dst_wt = transpose(r.take(h * c, "dst_proj.weight")?, h, c);
        let glob_proj_wt = transpose(r.take(h * h, "glob_proj.weight")?, h, h);
        let gift_emb = r.take(n_gift * h, "gift_emb.weight")?.to_vec();
        let promo_emb = r.take(n_promo * h, "promo_emb.weight")?.to_vec();
        let flag_emb = r.take(4 * h, "flag_emb.weight")?.to_vec();
        let act_fc_wt = transpose(r.take(hh * h, "act_fc.weight")?, hh, h);
        let act_fc_b = r.take(hh, "act_fc.bias")?.to_vec();
        let act_out_w = r.take(hh, "act_out.weight")?.to_vec();
        let act_out_b = r.take(1, "act_out.bias")?[0];
        if r.pos != blob.len() {
            return Err(format!(
                "weight blob too long: used {} floats, blob has {}",
                r.pos,
                blob.len()
            ));
        }
        Ok(Net {
            c,
            h,
            hh,
            n_gift,
            n_promo,
            cin,
            planes,
            n_scalars,
            stem,
            stem_norm,
            blocks: bl,
            glob_conv_wt,
            glob_conv_b,
            glob_fc_wt,
            glob_fc_b,
            value_w,
            value_b,
            src_wt,
            src_b,
            dst_wt,
            glob_proj_wt,
            gift_emb,
            promo_emb,
            flag_emb,
            act_fc_wt,
            act_fc_b,
            act_out_w,
            act_out_b,
        })
    }
}

// ---------------------------------------------------------------------------
// kernels (all #[inline(always)] so each instantiation is compiled with the caller's
// target features: AVX2+FMA in `forward_avx2`, baseline in `forward_generic`)
// ---------------------------------------------------------------------------

#[inline(always)]
fn fma<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    if F {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// One conv output tile for PX adjacent pixels of one row: PX x W accumulators.
/// `xin` is the padded NHWC input [10*10][cin]; `out` is [64][cout].
#[inline(always)]
unsafe fn conv_tile<const PX: usize, const W: usize, const F: bool>(
    xin: *const f32,
    cin: usize,
    pix: usize,
    w: *const f32,
    out: *mut f32,
    cout: usize,
    n0: usize,
) {
    let r = pix >> 3;
    let c = pix & 7;
    let mut a = [[0f32; W]; PX];
    for tap in 0..9usize {
        let base = xin.add(((r + tap / 3) * PADW + c + tap % 3) * cin);
        let wt = w.add(tap * cin * W);
        for ci in 0..cin {
            let wr = wt.add(ci * W);
            for p in 0..PX {
                let x = *base.add(p * cin + ci);
                for j in 0..W {
                    a[p][j] = fma::<F>(x, *wr.add(j), a[p][j]);
                }
            }
        }
    }
    for p in 0..PX {
        let o = out.add((pix + p) * cout + n0);
        for j in 0..W {
            *o.add(j) = a[p][j];
        }
    }
}

/// Whole 3x3 conv: padded input -> raw output [64][cout].
#[inline(always)]
unsafe fn conv_all<const F: bool>(conv: &Conv, xin: *const f32, out: *mut f32) {
    for t in &conv.tiles {
        let w = t.w.as_ptr();
        if t.lanes == 16 {
            for q in 0..16 {
                conv_tile::<4, 16, F>(xin, conv.cin, q * 4, w, out, conv.cout, t.n0);
            }
        } else {
            for q in 0..8 {
                conv_tile::<8, 8, F>(xin, conv.cin, q * 8, w, out, conv.cout, t.n0);
            }
        }
    }
}

#[inline(always)]
unsafe fn gemv_tile<const W: usize, const F: bool>(
    x: *const f32,
    k: usize,
    wt: *const f32,
    n: usize,
    n0: usize,
    bias: *const f32,
    out: *mut f32,
) {
    let mut a = [0f32; W];
    if !bias.is_null() {
        for j in 0..W {
            a[j] = *bias.add(n0 + j);
        }
    }
    for kk in 0..k {
        let xv = *x.add(kk);
        let wr = wt.add(kk * n + n0);
        for j in 0..W {
            a[j] = fma::<F>(xv, *wr.add(j), a[j]);
        }
    }
    for j in 0..W {
        *out.add(n0 + j) = a[j];
    }
}

/// out[n] = bias[n] + sum_k x[k] * wt[k][n]   (wt row-major [k][n], n a multiple of 8).
#[inline(always)]
unsafe fn gemv<const F: bool>(x: *const f32, k: usize, wt: *const f32, n: usize, bias: *const f32, out: *mut f32) {
    let mut n0 = 0;
    while n0 < n {
        let rem = n - n0;
        if rem >= 32 {
            gemv_tile::<32, F>(x, k, wt, n, n0, bias, out);
            n0 += 32;
        } else if rem >= 16 {
            gemv_tile::<16, F>(x, k, wt, n, n0, bias, out);
            n0 += 16;
        } else {
            gemv_tile::<8, F>(x, k, wt, n, n0, bias, out);
            n0 += 8;
        }
    }
}

#[inline(always)]
fn pad_index(p: usize) -> usize {
    ((p >> 3) + 1) * PADW + (p & 7) + 1
}

/// GroupNorm over `tmp` [64][c] (+ optional residual from a padded buffer, + optional ReLU),
/// written into the interior of the padded buffer `out`.
#[inline(always)]
fn group_norm<const RELU: bool>(
    tmp: &[f32],
    c: usize,
    norm: &Norm,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let cpg = c / GROUPS;
    let n = (SQ * cpg) as f32;
    let mut csum = vec![0f32; c];
    for row in tmp.chunks_exact(c) {
        for (s, v) in csum.iter_mut().zip(row) {
            *s += *v;
        }
    }
    let mut cmean = vec![0f32; c];
    for g in 0..GROUPS {
        let m = csum[g * cpg..(g + 1) * cpg].iter().sum::<f32>() / n;
        for v in &mut cmean[g * cpg..(g + 1) * cpg] {
            *v = m;
        }
    }
    let mut csq = vec![0f32; c];
    for row in tmp.chunks_exact(c) {
        for ((s, v), m) in csq.iter_mut().zip(row).zip(&cmean) {
            let d = *v - *m;
            *s += d * d;
        }
    }
    let mut scale = vec![0f32; c];
    let mut shift = vec![0f32; c];
    for g in 0..GROUPS {
        let var = csq[g * cpg..(g + 1) * cpg].iter().sum::<f32>() / n;
        let rstd = 1.0 / (var + EPS).sqrt();
        for ch in g * cpg..(g + 1) * cpg {
            scale[ch] = norm.gamma[ch] * rstd;
            shift[ch] = norm.beta[ch] - cmean[ch] * scale[ch];
        }
    }
    for p in 0..SQ {
        let src = &tmp[p * c..(p + 1) * c];
        let o = pad_index(p) * c;
        let dst = &mut out[o..o + c];
        match residual {
            Some(res) => {
                let rr = &res[o..o + c];
                for i in 0..c {
                    let v = src[i] * scale[i] + shift[i] + rr[i];
                    dst[i] = if RELU && v < 0.0 { 0.0 } else { v };
                }
            }
            None => {
                for i in 0..c {
                    let v = src[i] * scale[i] + shift[i];
                    dst[i] = if RELU && v < 0.0 { 0.0 } else { v };
                }
            }
        }
    }
}

#[inline(always)]
fn relu_in_place(v: &mut [f32]) {
    for x in v {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

impl Net {
    /// Forward pass.  Inputs were validated by `forward_checked`.
    #[inline(always)]
    unsafe fn forward_impl<const F: bool>(
        &self,
        batch: usize,
        board: &[f32],
        scalars: &[f32],
        counts: &[i32],
        comps: &[u8],
        logits: &mut [f32],
        value: &mut [f32],
    ) {
        let c = self.c;
        let h = self.h;
        let hh = self.hh;
        let cin = self.cin;
        let planes = self.planes;
        let ns = self.n_scalars;

        // padded buffers; borders stay zero because only interiors are ever written
        let mut xin = vec![0f32; PADW * PADW * cin];
        let mut cur = vec![0f32; PADW * PADW * c];
        let mut mid = vec![0f32; PADW * PADW * c];
        let mut tmp = vec![0f32; SQ * c];
        let mut gflat = vec![0f32; FLAT];
        let mut g = vec![0f32; h];
        let mut a_src = vec![0f32; SQ * h];
        let mut a_dst = vec![0f32; SQ * h];
        let mut a_glob = vec![0f32; h];
        let mut hbuf = vec![0f32; h];
        let mut ubuf = vec![0f32; hh];

        let mut act0 = 0usize; // first action of this position
        for b in 0..batch {
            // ---- input planes: board planes + scalars broadcast over the board
            let bd = &board[b * planes * SQ..(b + 1) * planes * SQ];
            let sc = &scalars[b * ns..(b + 1) * ns];
            for p in 0..SQ {
                let o = pad_index(p) * cin;
                for ch in 0..planes {
                    xin[o + ch] = bd[ch * SQ + p];
                }
                for k in 0..ns {
                    xin[o + planes + k] = sc[k];
                }
            }

            // ---- trunk
            conv_all::<F>(&self.stem, xin.as_ptr(), tmp.as_mut_ptr());
            group_norm::<true>(&tmp, c, &self.stem_norm, None, &mut cur);
            for blk in &self.blocks {
                conv_all::<F>(&blk.conv1, cur.as_ptr(), tmp.as_mut_ptr());
                group_norm::<true>(&tmp, c, &blk.norm1, None, &mut mid);
                conv_all::<F>(&blk.conv2, mid.as_ptr(), tmp.as_mut_ptr());
                // x = relu(x + norm2(conv2(y))), in place on `cur`
                let res: Vec<f32> = cur.clone();
                group_norm::<true>(&tmp, c, &blk.norm2, Some(&res), &mut cur);
            }

            // ---- global features and value
            let mut gc = [0f32; GLOB_CH];
            for p in 0..SQ {
                let row = cur.as_ptr().add(pad_index(p) * c);
                gemv::<F>(row, c, self.glob_conv_wt.as_ptr(), GLOB_CH, self.glob_conv_b.as_ptr(), gc.as_mut_ptr());
                for o in 0..GLOB_CH {
                    gflat[o * SQ + p] = if gc[o] < 0.0 { 0.0 } else { gc[o] };
                }
            }
            gemv::<F>(gflat.as_ptr(), FLAT, self.glob_fc_wt.as_ptr(), h, self.glob_fc_b.as_ptr(), g.as_mut_ptr());
            relu_in_place(&mut g);
            let mut v = self.value_b;
            for i in 0..h {
                v += g[i] * self.value_w[i];
            }
            value[b] = v.tanh();

            // ---- policy head: per-square projections, gathered per action
            for p in 0..SQ {
                let row = cur.as_ptr().add(pad_index(p) * c);
                gemv::<F>(row, c, self.src_wt.as_ptr(), h, self.src_b.as_ptr(), a_src.as_mut_ptr().add(p * h));
                gemv::<F>(row, c, self.dst_wt.as_ptr(), h, std::ptr::null(), a_dst.as_mut_ptr().add(p * h));
            }
            gemv::<F>(g.as_ptr(), h, self.glob_proj_wt.as_ptr(), h, std::ptr::null(), a_glob.as_mut_ptr());

            let n_act = counts[b] as usize;
            for i in act0..act0 + n_act {
                let cr = &comps[i * 6..i * 6 + 6];
                let (src, dst) = (cr[0] as usize, cr[1] as usize);
                let flag = cr[2] as usize * 2 + cr[4] as usize;
                let gift = &self.gift_emb[cr[3] as usize * h..(cr[3] as usize + 1) * h];
                let promo = &self.promo_emb[cr[5] as usize * h..(cr[5] as usize + 1) * h];
                let flg = &self.flag_emb[flag * h..(flag + 1) * h];
                let s = &a_src[src * h..(src + 1) * h];
                let d = &a_dst[dst * h..(dst + 1) * h];
                for j in 0..h {
                    let x = s[j] + d[j] + a_glob[j] + gift[j] + promo[j] + flg[j];
                    hbuf[j] = if x < 0.0 { 0.0 } else { x };
                }
                gemv::<F>(hbuf.as_ptr(), h, self.act_fc_wt.as_ptr(), hh, self.act_fc_b.as_ptr(), ubuf.as_mut_ptr());
                // 8 partial sums, so the compiler can vectorise the dot product
                let mut part = [0f32; 8];
                for (k, (u, w)) in ubuf.iter().zip(&self.act_out_w).enumerate() {
                    let u = if *u < 0.0 { 0.0 } else { *u };
                    part[k & 7] += u * *w;
                }
                logits[i] = part.iter().sum::<f32>() + self.act_out_b;
            }
            act0 += n_act;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, batch: usize, board: &[f32], scalars: &[f32], counts: &[i32], comps: &[u8], logits: &mut [f32], value: &mut [f32]) {
        self.forward_impl::<true>(batch, board, scalars, counts, comps, logits, value)
    }

    unsafe fn forward_generic(&self, batch: usize, board: &[f32], scalars: &[f32], counts: &[i32], comps: &[u8], logits: &mut [f32], value: &mut [f32]) {
        self.forward_impl::<false>(batch, board, scalars, counts, comps, logits, value)
    }

    /// Validates the inputs (so the unsafe kernels never see a bad index), then runs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_checked(
        &self,
        batch: usize,
        board: &[f32],
        scalars: &[f32],
        counts: &[i32],
        comps: &[u8],
        logits: &mut [f32],
        value: &mut [f32],
    ) -> Result<(), String> {
        let total: i64 = counts.iter().map(|&n| n as i64).sum();
        if counts.iter().any(|&n| n < 0) {
            return Err("negative action count".to_string());
        }
        if total as usize != logits.len() || comps.len() != logits.len() * 6 {
            return Err(format!(
                "action count mismatch: counts sum to {}, {} comps rows, {} logits slots",
                total,
                comps.len() / 6,
                logits.len()
            ));
        }
        for (i, cr) in comps.chunks_exact(6).enumerate() {
            let (src, dst, demote, gift, ep, promo) = (cr[0], cr[1], cr[2], cr[3], cr[4], cr[5]);
            if src as usize >= SQ || dst as usize >= SQ {
                return Err(format!("action {}: square out of range (src {}, dst {})", i, src, dst));
            }
            if demote > 1 || ep > 1 {
                return Err(format!("action {}: demote/en_passant must be 0 or 1 (got {}, {})", i, demote, ep));
            }
            if gift as usize >= self.n_gift || promo as usize >= self.n_promo {
                return Err(format!("action {}: gift {} / promote {} out of range", i, gift, promo));
            }
        }
        if board.len() != batch * self.planes * SQ || scalars.len() != batch * self.n_scalars || value.len() != batch {
            return Err("board / scalars / value length does not match the batch size".to_string());
        }
        #[cfg(target_arch = "x86_64")]
        {
            if simd_ok() {
                unsafe { self.forward_avx2(batch, board, scalars, counts, comps, logits, value) };
                return Ok(());
            }
        }
        unsafe { self.forward_generic(batch, board, scalars, counts, comps, logits, value) };
        Ok(())
    }
}

fn simd_ok() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let allowed = std::env::var("RC_NET_SIMD").map(|v| v.trim() != "0").unwrap_or(true);
            allowed && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rc_net_abi() -> i32 {
    1
}

/// 1 when the AVX2+FMA kernels are in use, 0 for the portable ones.
#[no_mangle]
pub extern "C" fn rc_net_simd() -> i32 {
    simd_ok() as i32
}

/// Copy the last error message of this thread into `buf` (NUL-terminated, truncated to `cap`);
/// returns the full message length.
///
/// # Safety
/// `buf` must point at `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn rc_net_error(buf: *mut u8, cap: usize) -> usize {
    LAST_ERR.with(|e| {
        let s = e.borrow();
        let b = s.as_bytes();
        if cap > 0 {
            let n = b.len().min(cap - 1);
            std::ptr::copy_nonoverlapping(b.as_ptr(), buf, n);
            *buf.add(n) = 0;
        }
        b.len()
    })
}

/// Build a network from a flat f32 weight blob (order: see `Net::from_blob`).  Returns null on
/// error; `rc_net_error` says why.
///
/// # Safety
/// `blob` must point at `n_floats` readable f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_net_new(
    blob: *const f32,
    n_floats: usize,
    channels: u32,
    blocks: u32,
    hidden: u32,
    n_gift: u32,
    n_promo: u32,
    planes: u32,
    n_scalars: u32,
) -> *mut Net {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let data = std::slice::from_raw_parts(blob, n_floats);
        match Net::from_blob(
            data,
            channels as usize,
            blocks as usize,
            hidden as usize,
            n_gift as usize,
            n_promo as usize,
            planes as usize,
            n_scalars as usize,
        ) {
            Ok(n) => Box::into_raw(Box::new(n)),
            Err(e) => {
                set_err(e);
                std::ptr::null_mut()
            }
        }
    }));
    match r {
        Ok(p) => p,
        Err(_) => {
            set_err("internal panic while building the network".to_string());
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `net` must come from `rc_net_new` (or be null) and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rc_net_free(net: *mut Net) {
    if !net.is_null() {
        drop(Box::from_raw(net));
    }
}

/// Forward pass for `batch` positions whose legal actions are stored back to back
/// (`counts[b]` of them for position b, `n_actions` in total).
///   board   (batch, planes, 8, 8) f32       scalars (batch, n_scalars) f32
///   comps   (n_actions, 6) u8: src, dst, demote, gift, en_passant, promote
///           (canonical frame, i.e. what net.action_components_array returns)
///   logits  (n_actions) f32 out             value (batch) f32 out
/// Returns 0, or -1 (bad input; `rc_net_error` says what) / -99 (internal panic).
///
/// # Safety
/// All pointers must be valid for the sizes above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_net_forward(
    net: *const Net,
    batch: u32,
    board: *const f32,
    scalars: *const f32,
    counts: *const i32,
    n_actions: u32,
    comps: *const u8,
    logits: *mut f32,
    value: *mut f32,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if net.is_null() {
            set_err("null network handle".to_string());
            return -1;
        }
        let net = &*net;
        let b = batch as usize;
        let n = n_actions as usize;
        let board = std::slice::from_raw_parts(board, b * net.planes * SQ);
        let scalars = std::slice::from_raw_parts(scalars, b * net.n_scalars);
        let counts = std::slice::from_raw_parts(counts, b);
        let comps = std::slice::from_raw_parts(comps, n * 6);
        let logits = std::slice::from_raw_parts_mut(logits, n);
        let value = std::slice::from_raw_parts_mut(value, b);
        match net.forward_checked(b, board, scalars, counts, comps, logits, value) {
            Ok(()) => 0,
            Err(e) => {
                set_err(e);
                -1
            }
        }
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_err("internal panic in the forward pass".to_string());
            -99
        }
    }
}
