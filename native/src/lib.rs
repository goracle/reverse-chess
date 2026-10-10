//! rc_engine -- native move generation for Reverse Chess.
//!
//! A line-for-line port of the hot paths of `engine.py`:
//!   * `Game._enumerate`  (legal actions, in exactly the same order), and
//!   * `Game._losing_reason` (pawn_scan / bishop_wall_check dead-end tests).
//!
//! No dependencies, plain C ABI, loaded from Python with ctypes.
//!
//! State wire format (128 bytes): 32 pieces x 4 bytes, in engine.py's piece
//! order (black 0..16 then white 16..32; each side: 8 pawns, king, 2 rooks,
//! 2 bishops, queen, 2 knights).  Per piece: [flags, pos, ptype, 0] with
//! flags bit0 = on board, bit1 = promoted, bit2 = locked; pos = row*8+col;
//! ptype = index into chessassets.PTYPES ("pkrbqn").
//!
//! Action wire format (6 bytes): src, dst, demote, gift, en_passant, promote
//! (same integer codes as env.action_components, but absolute squares).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

const PAWN: u8 = 0;
const KING: u8 = 1;
const ROOK: u8 = 2;
const BISHOP: u8 = 3;
const QUEEN: u8 = 4;
const KNIGHT: u8 = 5;
/// engine.PROMOTION_TYPES = ("q", "n", "b", "r"); promote code c means PROMO_TYPES[c-1].
const PROMO_TYPES: [u8; 4] = [QUEEN, KNIGHT, BISHOP, ROOK];

const KNIGHT_STEPS: [(i32, i32); 8] =
    [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)];
const KING_STEPS: [(i32, i32); 8] =
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAG: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
/// (king_col, rook_col, rook_home_col) after a forward castle.
const CASTLE_SIDES: [(i32, i32, i32); 2] = [(2, 3, 0), (6, 5, 7)];

// ---------------------------------------------------------------------------
// small geometry helpers (team 0 = 'b', team 1 = 'w')
// ---------------------------------------------------------------------------
#[inline]
fn team_of(i: usize) -> usize {
    i >> 4
}
#[inline]
fn other(t: usize) -> usize {
    1 - t
}
#[inline]
fn back_row(t: usize) -> i32 {
    if t == 0 { 0 } else { 7 }
}
#[inline]
fn pawn_home_row(t: usize) -> i32 {
    if t == 0 { 1 } else { 6 }
}
#[inline]
fn promotion_row(t: usize) -> i32 {
    if t == 0 { 7 } else { 0 }
}
#[inline]
fn pawn_step(t: usize) -> i32 {
    if t == 0 { -1 } else { 1 }
}
#[inline]
fn rc(sq: u8) -> (i32, i32) {
    ((sq >> 3) as i32, (sq & 7) as i32)
}
#[inline]
fn sqi(r: i32, c: i32) -> usize {
    (r * 8 + c) as usize
}
#[inline]
fn inb(r: i32, c: i32) -> bool {
    r >= 0 && r < 8 && c >= 0 && c < 8
}

/// bitmask of squares a (team, ptype) piece on `src` could reach by geometry alone
/// (mirror of engine._geom_dsts; bit order == row-major == Python's sorted order).
static GEOM: OnceLock<Vec<u64>> = OnceLock::new();

fn build_geom() -> Vec<u64> {
    let mut t = vec![0u64; 2 * 6 * 64];
    for team in 0..2usize {
        for pt in 0..6u8 {
            for src in 0..64i32 {
                let (sr, sc) = (src >> 3, src & 7);
                let mut m = 0u64;
                for r in 0..8i32 {
                    for c in 0..8i32 {
                        let (dr, dc) = (r - sr, c - sc);
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let ok = match pt {
                            KING => {
                                dr.abs().max(dc.abs()) == 1 || (sr == r && sr == back_row(team))
                            }
                            ROOK => dr == 0 || dc == 0,
                            KNIGHT => {
                                (dr.abs() == 1 && dc.abs() == 2) || (dr.abs() == 2 && dc.abs() == 1)
                            }
                            QUEEN => dr == 0 || dc == 0 || dr.abs() == dc.abs(),
                            BISHOP => dr.abs() == dc.abs(),
                            _ => dr == pawn_step(team) && dc.abs() <= 1,
                        };
                        if ok {
                            m |= 1u64 << (r * 8 + c);
                        }
                    }
                }
                t[(team * 6 + pt as usize) * 64 + src as usize] = m;
            }
        }
    }
    t
}

#[inline]
fn geom(team: usize, pt: u8, src: usize) -> u64 {
    GEOM.get_or_init(build_geom)[(team * 6 + pt as usize) * 64 + src]
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Piece {
    on: bool,
    promoted: bool,
    locked: bool,
    pos: u8,
    ptype: u8,
}

#[derive(Clone, Copy)]
struct State {
    p: [Piece; 32],
    board: [i8; 64], // piece index or -1
}

#[derive(Clone, Copy)]
struct Opt {
    dst: u8,
    demote: bool,
    forced: bool,
    castle: bool,
}

#[derive(Clone, Copy)]
struct Mv {
    mover: usize,
    dst: u8,
    demote: bool,
    castle: bool,
    partner: i32,
    spawn: i32,
    spawn_pos: u8,
    promote_to: Option<u8>,
}

impl State {
    fn from_bytes(b: &[u8]) -> Option<State> {
        if b.len() < 128 {
            return None;
        }
        let blank = Piece { on: false, promoted: false, locked: false, pos: 255, ptype: 0 };
        let mut st = State { p: [blank; 32], board: [-1; 64] };
        for i in 0..32 {
            let f = b[4 * i];
            let pos = b[4 * i + 1];
            let pt = b[4 * i + 2];
            if pt > 5 {
                return None;
            }
            let on = f & 1 != 0;
            st.p[i] = Piece {
                on,
                promoted: f & 2 != 0,
                locked: f & 4 != 0,
                pos: if on { pos } else { 255 },
                ptype: pt,
            };
            if on {
                if pos >= 64 || st.board[pos as usize] >= 0 {
                    return None;
                }
                st.board[pos as usize] = i as i8;
            }
        }
        if !st.p[8].on || !st.p[24].on {
            return None; // both kings are always on the board
        }
        Some(st)
    }

    /// Is piece `i` standing unpromoted on one of its home squares?
    fn at_home(&self, i: usize) -> bool {
        let p = &self.p[i];
        if !p.on || p.promoted {
            return false;
        }
        let (r, c) = rc(p.pos);
        let t = team_of(i);
        let o = i & 15;
        if o < 8 {
            return r == pawn_home_row(t);
        }
        if r != back_row(t) {
            return false;
        }
        match o {
            8 => c == 4,
            9 | 10 => c == 0 || c == 7,
            11 | 12 => c == 2 || c == 5,
            13 => c == 3,
            _ => c == 1 || c == 6,
        }
    }

    fn all_on_board(&self, team: usize) -> bool {
        (0..16).all(|k| self.p[team * 16 + k].on)
    }

    fn king_pos(&self, team: usize) -> (i32, i32) {
        rc(self.p[team * 16 + 8].pos)
    }

    fn path_blocked(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        let (dr, dc) = (b.0 - a.0, b.1 - a.1);
        if !(dr == 0 || dc == 0 || dr.abs() == dc.abs()) {
            return false;
        }
        let (sr, sc) = (dr.signum(), dc.signum());
        let n = dr.abs().max(dc.abs());
        for i in 1..n {
            if self.board[sqi(a.0 + i * sr, a.1 + i * sc)] >= 0 {
                return true;
            }
        }
        false
    }

    /// engine.check_attacking: does piece `i` (where it stands) attack `target`?
    fn check_attacking(&self, i: usize, target: (i32, i32)) -> bool {
        let p = &self.p[i];
        let pos = rc(p.pos);
        let (dr, dc) = (target.0 - pos.0, target.1 - pos.1);
        if dr == 0 && dc == 0 {
            return false;
        }
        match p.ptype {
            KNIGHT => (dr.abs() == 1 && dc.abs() == 2) || (dr.abs() == 2 && dc.abs() == 1),
            KING => dr.abs().max(dc.abs()) == 1,
            PAWN => dr == (if team_of(i) == 0 { 1 } else { -1 }) && dc.abs() == 1,
            pt => {
                let straight = dr == 0 || dc == 0;
                let diagonal = dr.abs() == dc.abs();
                let line = if pt == ROOK {
                    straight
                } else if pt == BISHOP {
                    diagonal
                } else {
                    straight || diagonal
                };
                line && !self.path_blocked(pos, target)
            }
        }
    }

    /// engine._team_attacks: does any piece of `team` attack (tr, tc)?
    fn team_attacks(&self, team: usize, tr: i32, tc: i32) -> bool {
        for &(dr, dc) in KNIGHT_STEPS.iter() {
            let (r, c) = (tr + dr, tc + dc);
            if inb(r, c) {
                let k = self.board[sqi(r, c)];
                if k >= 0 && team_of(k as usize) == team && self.p[k as usize].ptype == KNIGHT {
                    return true;
                }
            }
        }
        for &(dr, dc) in KING_STEPS.iter() {
            let (r, c) = (tr + dr, tc + dc);
            if inb(r, c) {
                let k = self.board[sqi(r, c)];
                if k >= 0 && team_of(k as usize) == team && self.p[k as usize].ptype == KING {
                    return true;
                }
            }
        }
        // a pawn attacks in the forward-chess direction (black down, white up)
        let pr = tr - (if team == 0 { 1 } else { -1 });
        if pr >= 0 && pr < 8 {
            for &pc in [tc - 1, tc + 1].iter() {
                if pc >= 0 && pc < 8 {
                    let k = self.board[sqi(pr, pc)];
                    if k >= 0 && team_of(k as usize) == team && self.p[k as usize].ptype == PAWN {
                        return true;
                    }
                }
            }
        }
        for &(dr, dc) in ORTHO.iter() {
            let (mut r, mut c) = (tr + dr, tc + dc);
            while inb(r, c) {
                let k = self.board[sqi(r, c)];
                if k >= 0 {
                    let q = &self.p[k as usize];
                    if team_of(k as usize) == team && (q.ptype == ROOK || q.ptype == QUEEN) {
                        return true;
                    }
                    break;
                }
                r += dr;
                c += dc;
            }
        }
        for &(dr, dc) in DIAG.iter() {
            let (mut r, mut c) = (tr + dr, tc + dc);
            while inb(r, c) {
                let k = self.board[sqi(r, c)];
                if k >= 0 {
                    let q = &self.p[k as usize];
                    if team_of(k as usize) == team && (q.ptype == BISHOP || q.ptype == QUEEN) {
                        return true;
                    }
                    break;
                }
                r += dr;
                c += dc;
            }
        }
        false
    }

    /// engine._discovered_check: after piece `i` goes to `dst`, does its team attack the enemy king?
    fn discovered_check(&mut self, i: usize, dst: usize, demote: bool) -> bool {
        let src = self.p[i].pos as usize;
        let (old_pt, old_pro) = (self.p[i].ptype, self.p[i].promoted);
        self.board[src] = -1;
        self.board[dst] = i as i8;
        if demote && self.p[i].promoted {
            self.p[i].ptype = PAWN;
            self.p[i].promoted = false;
        }
        let team = team_of(i);
        let (kr, kc) = self.king_pos(other(team));
        let res = self.team_attacks(team, kr, kc);
        self.p[i].ptype = old_pt;
        self.p[i].promoted = old_pro;
        self.board[dst] = -1;
        self.board[src] = i as i8;
        res
    }

    /// engine._king_pawn_ok
    fn king_pawn_ok(&self, team: usize, newpos: Option<(i32, i32)>) -> bool {
        let (tr, tc) = match newpos {
            Some(p) => p,
            None => self.king_pos(team),
        };
        let enemy = other(team);
        let home = pawn_home_row(enemy);
        if tr != home + (if enemy == 0 { 1 } else { -1 }) {
            return true;
        }
        for &pc in [tc - 1, tc + 1].iter() {
            if pc >= 0 && pc < 8 {
                let k = self.board[sqi(home, pc)];
                if k >= 0 {
                    let k = k as usize;
                    if team_of(k) == enemy && self.p[k].ptype == PAWN && self.at_home(k) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// engine._reverse_castle_ok
    fn reverse_castle_ok(&self, by_king: bool, team: usize, src: (i32, i32), dst: (i32, i32)) -> bool {
        let back = back_row(team);
        if src.0 != back || dst.0 != back {
            return false;
        }
        let count = (0..8).filter(|&c| self.board[sqi(back, c)] >= 0).count();
        if count != 2 {
            return false;
        }
        for &(kc, rcol, rook_home) in CASTLE_SIDES.iter() {
            let k = self.board[sqi(back, kc)];
            let r = self.board[sqi(back, rcol)];
            if k < 0 || r < 0 {
                continue;
            }
            let (k, r) = (k as usize, r as usize);
            if self.p[k].ptype == KING
                && self.p[r].ptype == ROOK
                && team_of(k) == team
                && team_of(r) == team
                && !self.p[r].promoted
            {
                if by_king {
                    return src == rc(self.p[k].pos) && dst == (back, 4);
                }
                return src == rc(self.p[r].pos) && dst == (back, rook_home);
            }
        }
        false
    }

    /// engine._pawn_step_ok -> (legal, forced_give)
    fn pawn_step_ok(&self, src: (i32, i32), dst: (i32, i32), team: usize) -> (bool, bool) {
        if dst.0 - src.0 != pawn_step(team) || dst.0 == 0 || dst.0 == 7 {
            return (false, false);
        }
        let sideways = (dst.1 - src.1).abs();
        let forced = sideways == 1;
        let mut legal = forced || sideways == 0;
        if forced && self.all_on_board(other(team)) {
            legal = false;
        }
        (legal, forced)
    }

    /// engine.check_legal_move -> Some((forced_give, reverse_castle)) when legal.
    fn check_legal_move(&mut self, src: usize, dst: usize, as_pawn: bool) -> Option<(bool, bool)> {
        let bi = self.board[src];
        if bi < 0 {
            return None;
        }
        let i = bi as usize;
        let piece = self.p[i];
        let team = team_of(i);
        let pt = if as_pawn { PAWN } else { piece.ptype };
        if src == dst || self.board[dst] >= 0 {
            return None;
        }
        let s = rc(src as u8);
        let d = rc(dst as u8);
        let (dr, dc) = (d.0 - s.0, d.1 - s.1);
        let mut forced = false;
        let mut castle = false;
        let legal;
        match pt {
            KING => {
                let mut l = dr.abs().max(dc.abs()) == 1;
                if !l {
                    l = !piece.promoted && self.reverse_castle_ok(true, team, s, d);
                    castle = l;
                }
                legal = l;
            }
            ROOK => legal = dr == 0 || dc == 0,
            KNIGHT => legal = (dr.abs() == 1 && dc.abs() == 2) || (dr.abs() == 2 && dc.abs() == 1),
            QUEEN => legal = dr == 0 || dc == 0 || dr.abs() == dc.abs(),
            BISHOP => legal = dr.abs() == dc.abs(),
            _ => {
                let (l, f) = self.pawn_step_ok(s, d, team);
                legal = l;
                forced = f;
            }
        }
        if !legal {
            return None;
        }
        // sliding pieces may not jump (knights do; castling handled above)
        if pt != KNIGHT && !castle && self.path_blocked(s, d) {
            if pt == ROOK && self.reverse_castle_ok(false, team, s, d) {
                castle = true; // the rook "jumps" over its king
            } else {
                return None;
            }
        }
        if self.discovered_check(i, dst, as_pawn && piece.ptype != PAWN) {
            return None;
        }
        if !self.king_pawn_ok(team, if piece.ptype == KING { Some(d) } else { None }) {
            return None;
        }
        if castle {
            let home_sq = (back_row(team), 4);
            let en = other(team);
            for k in 0..16 {
                let j = en * 16 + k;
                if self.p[j].on && self.check_attacking(j, home_sq) {
                    return None;
                }
            }
        }
        Some((forced, castle))
    }

    /// engine.piece_moves (uncached).
    fn piece_moves(&mut self, i: usize, first_only: bool, out: &mut Vec<Opt>) {
        let p = self.p[i];
        if p.locked || !p.on {
            return;
        }
        let team = team_of(i);
        let src = p.pos as usize;
        let can_demote = p.promoted && (src as i32 >> 3) == promotion_row(team);
        let mut mask = geom(team, p.ptype, src);
        if can_demote {
            mask |= geom(team, PAWN, src);
        }
        while mask != 0 {
            let dst = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            if self.board[dst] >= 0 {
                continue;
            }
            if let Some((f, c)) = self.check_legal_move(src, dst, false) {
                out.push(Opt { dst: dst as u8, demote: false, forced: f, castle: c });
            }
            if can_demote {
                if let Some((f, c)) = self.check_legal_move(src, dst, true) {
                    out.push(Opt { dst: dst as u8, demote: true, forced: f, castle: c });
                }
            }
            if first_only && !out.is_empty() {
                return;
            }
        }
    }

    /// engine.has_legal_move: does `team` have at least one raw move?
    fn has_legal_move(&mut self, team: usize) -> bool {
        let mut tmp: Vec<Opt> = Vec::with_capacity(2);
        for k in 0..16 {
            let i = team * 16 + k;
            if self.p[i].on {
                tmp.clear();
                self.piece_moves(i, true, &mut tmp);
                if !tmp.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    fn bishop_colour_ok(&self, team: usize, spawn_pos: u8) -> bool {
        let (r, c) = rc(spawn_pos);
        let colour = (r + c) % 2;
        for k in 0..16 {
            let p = &self.p[team * 16 + k];
            if p.on && p.ptype == BISHOP && !p.promoted {
                let (pr, pc) = rc(p.pos);
                if (pr + pc) % 2 == colour {
                    return false;
                }
            }
        }
        true
    }

    /// engine.bishop_wall_check
    fn bishop_wall_check(&self, team: usize) -> bool {
        let mut home_cols = [false; 8];
        for k in 0..16 {
            let i = team * 16 + k;
            let p = &self.p[i];
            if p.on && p.ptype == BISHOP && self.at_home(i) {
                home_cols[(p.pos & 7) as usize] = true;
            }
        }
        let r = pawn_home_row(team);
        let walls: [(usize, [(i32, i32); 2]); 2] =
            [(2, [(r, 1), (r, 3)]), (5, [(r, 4), (r, 6)])];
        for &(col, ref squares) in walls.iter() {
            if home_cols[col] {
                continue;
            }
            let mut walled = 0;
            for &(sr, sc) in squares.iter() {
                let k = self.board[sqi(sr, sc)];
                if k >= 0 {
                    let k = k as usize;
                    if team_of(k) == team && self.p[k].ptype == PAWN && self.at_home(k) {
                        walled += 1;
                    }
                }
            }
            if walled == 2 {
                return false;
            }
        }
        true
    }

    /// engine.pawn_scan (+ tenants.tenant_matching as a bitmask DP over columns).
    fn pawn_scan(&self, team: usize, spawn: Option<(i32, i32)>) -> bool {
        let mut pawns = [(0i32, 0i32); 16];
        let mut n = 0usize;
        let mut locked_cols: u32 = 0;
        for k in 0..16 {
            let i = team * 16 + k;
            let p = &self.p[i];
            if p.on && p.ptype == PAWN && !p.promoted {
                let (r, c) = rc(p.pos);
                if self.at_home(i) {
                    locked_cols |= 1 << c;
                } else {
                    pawns[n] = (r, c);
                    n += 1;
                }
            }
        }
        if let Some(s) = spawn {
            pawns[n] = s;
            n += 1;
        }
        if n == 0 {
            return true;
        }
        let open: u32 = 0xFF & !locked_cols;
        let home_row = pawn_home_row(team);
        const INF: u16 = 10_000;
        let mut dp = [INF; 256];
        dp[0] = 0;
        for idx in 0..n {
            let (row, col) = pawns[idx];
            let reach = (row - home_row).abs();
            let mut ndp = [INF; 256];
            let mut any = false;
            for mask in 0..256usize {
                let base = dp[mask];
                if base >= INF {
                    continue;
                }
                for c in 0..8i32 {
                    let bit = 1usize << c;
                    if open & (bit as u32) == 0 || mask & bit != 0 {
                        continue;
                    }
                    let d = (c - col).abs();
                    if d > reach {
                        continue;
                    }
                    let nm = mask | bit;
                    let cost = base + d as u16;
                    if cost < ndp[nm] {
                        ndp[nm] = cost;
                        any = true;
                    }
                }
            }
            if !any {
                return false;
            }
            dp = ndp;
        }
        let needed = *dp.iter().min().unwrap() as usize;
        let spare = (0..16).filter(|&k| !self.p[other(team) * 16 + k].on).count();
        needed <= spare
    }

    /// engine._pawn_spawn_home_ok
    fn pawn_spawn_home_ok(&self, team: usize, spawn_pos: (i32, i32), newpos: Option<(i32, i32)>) -> bool {
        if spawn_pos.0 != pawn_home_row(team) {
            return true;
        }
        let target = match newpos {
            Some(p) => p,
            None => self.king_pos(other(team)),
        };
        let dr = spawn_pos.0 + (if team == 0 { 1 } else { -1 });
        !(target.0 == dr && (target.1 - spawn_pos.1).abs() == 1)
    }

    /// engine._reverse_ep_possible
    fn reverse_ep_possible(&self, src: (i32, i32), dst: (i32, i32), mover: usize, spawn: usize) -> bool {
        if !(self.p[mover].ptype == PAWN && self.p[spawn].ptype == PAWN) {
            return false;
        }
        let (row, col) = src;
        let row1 = dst.0;
        let st = team_of(spawn);
        let ok = (st == 0 && row == 2 && row1 == 3) || (st == 1 && row == 5 && row1 == 4);
        ok && self.board[sqi(row - 1, col)] < 0 && self.board[sqi(row + 1, col)] < 0
    }

    // -- Piece.update / place / lock bookkeeping, as in engine.Piece ---------
    fn refresh_lock(&mut self, i: usize) {
        let l = self.p[i].ptype == PAWN && self.at_home(i);
        self.p[i].locked = l;
    }

    fn update(&mut self, i: usize, newpos: usize) {
        let old = self.p[i].pos as usize;
        self.board[old] = -1;
        self.board[newpos] = i as i8;
        self.p[i].pos = newpos as u8;
        self.refresh_lock(i);
    }

    /// engine.Game._execute
    fn apply(&mut self, mv: &Mv) {
        let i = mv.mover;
        if mv.partner >= 0 {
            let j = mv.partner as usize;
            let col = if self.p[j].ptype == KING {
                4
            } else if (self.p[j].pos & 7) < 4 {
                0
            } else {
                7
            };
            self.update(j, sqi(back_row(team_of(j)), col));
            self.p[j].locked = true;
        }
        self.update(i, mv.dst as usize);
        if mv.castle {
            self.p[i].locked = true;
        }
        if mv.demote && self.p[i].promoted {
            self.p[i].ptype = PAWN;
            self.p[i].promoted = false;
        }
        if mv.spawn >= 0 {
            let s = mv.spawn as usize;
            self.board[mv.spawn_pos as usize] = s as i8;
            self.p[s].on = true;
            self.p[s].pos = mv.spawn_pos;
            self.refresh_lock(s);
            if let Some(pt) = mv.promote_to {
                if !self.p[s].promoted {
                    self.p[s].ptype = pt;
                    self.p[s].promoted = true;
                    self.p[s].locked = false;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// enumeration
// ---------------------------------------------------------------------------

/// engine._leaves_reply: after `mv`, does the opponent still have a raw move?
fn leaves_reply(st: &mut State, mv: &Mv) -> bool {
    let saved = *st;
    st.apply(mv);
    let r = st.has_legal_move(other(team_of(mv.mover)));
    *st = saved;
    r
}

/// The promotion-independent part of engine._gift_problem (None-ness == true here).
fn gift_base_ok(
    st: &mut State,
    mover: usize,
    src: u8,
    dst: u8,
    spawn: usize,
    spawn_pos: u8,
    had: &mut Option<bool>,
) -> bool {
    let gteam = team_of(spawn);
    if st.p[spawn].on {
        return false;
    }
    if st.board[spawn_pos as usize] >= 0 && spawn_pos != src {
        return false;
    }
    if st.p[spawn].ptype == BISHOP && !st.bishop_colour_ok(gteam, spawn_pos) {
        return false;
    }
    let had_moves = match *had {
        Some(h) => h,
        None => {
            let h = st.has_legal_move(gteam);
            *had = Some(h);
            h
        }
    };
    // Hypothetical position: mover has moved, gift has landed (locks untouched,
    // a gifted pawn stays a plain pawn -- exactly as engine.py does it).
    let saved = *st;
    st.board[src as usize] = -1;
    st.board[dst as usize] = mover as i8;
    st.p[mover].pos = dst;
    st.board[spawn_pos as usize] = spawn as i8;
    st.p[spawn].on = true;
    st.p[spawn].pos = spawn_pos;
    let mut ok = true;
    if had_moves && !st.has_legal_move(gteam) {
        ok = false;
    }
    if ok && !st.bishop_wall_check(gteam) {
        ok = false;
    }
    *st = saved;
    ok
}

fn make_mv(
    st: &State,
    mover: usize,
    src: u8,
    opt: &Opt,
    spawn: i32,
    spawn_pos: u8,
    promote_to: Option<u8>,
) -> Mv {
    let mut partner = -1i32;
    if opt.castle {
        let row = (src >> 3) as i32;
        for c in 0..8 {
            let k = st.board[sqi(row, c)];
            if k >= 0 && k as usize != mover {
                partner = k as i32;
                break;
            }
        }
    }
    Mv { mover, dst: opt.dst, demote: opt.demote, castle: opt.castle, partner, spawn, spawn_pos, promote_to }
}

/// engine._enumerate + the playable filter.  Appends 6 bytes per playable action
/// to `out`; returns (any valid action exists, number of playable actions).
fn enumerate(st: &mut State, team: usize, out: &mut Vec<u8>) -> (bool, usize) {
    let opp = other(team);
    // engine.offboard_by_type: first off-board piece of each type, in piece order
    let mut off: Vec<(u8, usize)> = Vec::with_capacity(6);
    for k in 0..16 {
        let j = opp * 16 + k;
        let p = st.p[j];
        if !p.on && !off.iter().any(|&(t, _)| t == p.ptype) {
            off.push((p.ptype, j));
        }
    }
    let mut had: Option<bool> = None;
    let mut valid_any = false;
    let mut n_play = 0usize;
    let mut opts: Vec<Opt> = Vec::with_capacity(64);

    for k in 0..16 {
        let i = team * 16 + k;
        if !st.p[i].on {
            continue;
        }
        opts.clear();
        st.piece_moves(i, false, &mut opts);
        for oi in 0..opts.len() {
            let opt = opts[oi];
            let src = st.p[i].pos;
            let (src_r, src_c) = rc(src);

            // plain (gift-less) action
            if !opt.forced {
                valid_any = true;
                let mv = make_mv(st, i, src, &opt, -1, 0, None);
                if leaves_reply(st, &mv) {
                    out.extend_from_slice(&[src, opt.dst, opt.demote as u8, 0, 0, 0]);
                    n_play += 1;
                }
            }
            // gifts: forced for diagonal pawn steps, optional for non-pawns, never on a castle
            let mover_pt = st.p[i].ptype;
            if off.is_empty() || opt.castle || !(opt.forced || mover_pt != PAWN) {
                continue;
            }
            for &(gt, sidx) in off.iter() {
                let ep_possible = st.reverse_ep_possible(rc(src), rc(opt.dst), i, sidx);
                let n_ep = if ep_possible { 2 } else { 1 };
                for epi in 0..n_ep {
                    let ep = epi == 1;
                    let spawn_pos: u8 =
                        if ep { sqi(pawn_home_row(opp), src_c) as u8 } else { src };
                    if !gift_base_ok(st, i, src, opt.dst, sidx, spawn_pos, &mut had) {
                        continue; // every promotion variant fails too
                    }
                    // promotion choices (code 0 = stays what it is)
                    let mut promos: [u8; 5] = [0; 5];
                    let mut np = 0usize;
                    if gt == PAWN && !ep {
                        let must = src_r == promotion_row(opp);
                        if !must {
                            promos[np] = 0;
                            np += 1;
                        }
                        for c in 1..=4u8 {
                            promos[np] = c;
                            np += 1;
                        }
                    } else {
                        promos[0] = 0;
                        np = 1;
                    }
                    let mut extra: Option<bool> = None;
                    for pi in 0..np {
                        let promo = promos[pi];
                        let promoting = gt == PAWN && promo != 0;
                        if gt == PAWN && !promoting {
                            let ok = match extra {
                                Some(e) => e,
                                None => {
                                    let sp = rc(spawn_pos);
                                    let e = st.pawn_spawn_home_ok(
                                        opp,
                                        sp,
                                        if mover_pt == KING { Some(rc(opt.dst)) } else { None },
                                    ) && st.pawn_scan(opp, Some(sp))
                                        && sp.0 != back_row(opp);
                                    extra = Some(e);
                                    e
                                }
                            };
                            if !ok {
                                continue;
                            }
                        }
                        valid_any = true;
                        let pt = if promo == 0 { None } else { Some(PROMO_TYPES[(promo - 1) as usize]) };
                        let mv = make_mv(st, i, src, &opt, sidx as i32, spawn_pos, pt);
                        if leaves_reply(st, &mv) {
                            out.extend_from_slice(&[
                                src,
                                opt.dst,
                                opt.demote as u8,
                                gt + 1,
                                ep as u8,
                                promo,
                            ]);
                            n_play += 1;
                        }
                    }
                }
            }
        }
    }
    (valid_any, n_play)
}

/// engine._losing_reason: 0 = none, 1 = pawn wall-off, 2 = pawns, 3 = bishops.
fn losing(st: &State, team: usize) -> i32 {
    let base = team * 16;
    let all_home = (0..16).all(|k| st.p[base + k].on && st.at_home(base + k));
    if !all_home {
        let mut moving = false;
        for k in 0..16 {
            let p = &st.p[base + k];
            if !p.on {
                if p.ptype == PAWN || p.promoted {
                    moving = true;
                    break;
                }
            } else if (p.ptype == PAWN || p.promoted) && !st.at_home(base + k) {
                moving = true;
                break;
            }
        }
        if !moving {
            let outside =
                (0..16).any(|k| st.p[base + k].on && st.p[base + k].ptype != KNIGHT && !st.at_home(base + k));
            if outside {
                return 1;
            }
        }
    }
    if !st.pawn_scan(team, None) {
        return 2;
    }
    if !st.bishop_wall_check(team) {
        return 3;
    }
    0
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rc_abi_version() -> i32 {
    1
}

/// Enumerate the playable actions for `turn` (0 = black, 1 = white).
/// Returns n > 0 (actions written to `out`), 0 (no valid action at all),
/// -2 (valid actions exist but none is playable), -1 (`cap` too small),
/// -3 (bad state), -99 (internal panic).
///
/// # Safety
/// `state` must point at 128 readable bytes and `out` at 6 * `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn rc_enumerate(state: *const u8, turn: u32, out: *mut u8, cap: u32) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let bytes = std::slice::from_raw_parts(state, 128);
        let mut st = match State::from_bytes(bytes) {
            Some(s) => s,
            None => return -3,
        };
        let mut v: Vec<u8> = Vec::with_capacity(6 * 256);
        let (valid_any, n) = enumerate(&mut st, (turn & 1) as usize, &mut v);
        if n == 0 {
            return if valid_any { -2 } else { 0 };
        }
        if n > cap as usize {
            return -1;
        }
        std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
        n as i32
    }));
    r.unwrap_or(-99)
}

/// engine._losing_reason for `team` (0 = black, 1 = white); see `losing`.
///
/// # Safety
/// `state` must point at 128 readable bytes.
#[no_mangle]
pub unsafe extern "C" fn rc_losing(state: *const u8, team: u32) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let bytes = std::slice::from_raw_parts(state, 128);
        match State::from_bytes(bytes) {
            Some(st) => losing(&st, (team & 1) as usize),
            None => -3,
        }
    }));
    r.unwrap_or(-99)
}
