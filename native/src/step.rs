//! rc_step / rc_observe -- the "light game" half of the native engine.
//!
//! `rc_step` plays ONE ply on a packed state and does everything `engine.Game.step`
//! does that does not need Python-side history:
//!
//!   * resolves the action against the pieces (`Game._resolve`) and applies it
//!     (`Game._execute`),
//!   * the fifty-move counter,
//!   * both teams' win conditions (`Game._win_reason`), and the ply cap,
//!   * the position fingerprint that engine.py's repetition table is keyed on
//!     (`snapshot() + turn letter`),
//!   * and, if the game goes on, the next side's playable actions
//!     (`Game._begin_turn`, via the same `enumerate` rc_enumerate uses).
//!
//! The one thing it cannot decide is the repetition draw: that needs the table of
//! positions seen so far, which the caller owns.  So the result is a *verdict before
//! repetition*; the caller applies the repetition test between "50-move" and "ply cap",
//! exactly where `Game.step` does (see fastgame.py).
//!
//! `rc_observe` writes the network input (env.encode_observation) for a packed state.
//!
//! Output buffer of rc_step: a 200-byte header followed by 6 bytes per action.
//!
//! ```text
//!   0..128   new state, same 128-byte wire format as the input
//!   128..192 board fingerprint: Piece.code() of each square (0 = empty), row-major
//!   192      next side to move as a letter, b'b' or b'w'   (128..193 is the repetition key)
//!   193      status      0 live (actions follow)      1 a team has won
//!                        2 draw: both teams satisfy a winning condition
//!                        3 draw: fifty-move rule      4 ply cap reached (repetition still to test)
//!                        5 side to move has no legal move (winner decided)
//!                        6 every legal move would leave the opponent without one (draw)
//!   194      winner: 0 black, 1 white, 255 none
//!   195      win reason: 1 = every piece home; 11/12/13 = the other team is lost
//!            (engine._losing_reason codes 1/2/3)
//!   196..198 quiet plies after the move (u16, little endian)
//!   198..200 unused
//!   200..    6-byte actions (src, dst, demote, gift, en_passant, promote)
//! ```

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    enumerate, losing, make_mv, other, pawn_home_row, sqi, team_of, Opt, State, BISHOP, KING, KNIGHT,
    PAWN, PROMO_TYPES, QUEEN, ROOK,
};

const HDR: usize = 200;

const ST_LIVE: u8 = 0;
const ST_WIN: u8 = 1;
const ST_DRAW_BOTH: u8 = 2;
const ST_DRAW_QUIET: u8 = 3;
const ST_MAXPLY: u8 = 4;
const ST_NO_MOVES: u8 = 5;
const ST_BLOCKED: u8 = 6;

/// rules flag bit 0: Rules.win_requires_all_pieces_on_board
const FLAG_ALL_ON_BOARD: i32 = 1;

impl State {
    /// Inverse of `State::from_bytes` (engine.Game._pack's wire format).
    fn write_bytes(&self, out: &mut [u8]) {
        for i in 0..32 {
            let p = &self.p[i];
            if p.on {
                out[4 * i] = 1 | (if p.promoted { 2 } else { 0 }) | (if p.locked { 4 } else { 0 });
                out[4 * i + 1] = p.pos;
            } else {
                out[4 * i] = 0;
                out[4 * i + 1] = 255;
            }
            out[4 * i + 2] = p.ptype;
            out[4 * i + 3] = 0;
        }
    }

    /// engine.Piece.code of the piece on each square (0 = empty): the repetition fingerprint.
    fn write_snapshot(&self, out: &mut [u8]) {
        for sq in 0..64 {
            let k = self.board[sq];
            out[sq] = if k < 0 {
                0
            } else {
                let i = k as usize;
                let p = &self.p[i];
                1 + ((((p.ptype * 2 + p.promoted as u8) * 2 + p.locked as u8) * 2) + team_of(i) as u8)
            };
        }
    }

    /// engine.Game.check_for_win
    fn check_for_win(&self, team: usize, flags: i32) -> bool {
        if flags & FLAG_ALL_ON_BOARD != 0 && !(self.all_on_board(0) && self.all_on_board(1)) {
            return false;
        }
        (0..16).all(|k| {
            let i = team * 16 + k;
            !self.p[i].on || self.at_home(i)
        })
    }

    /// engine.Game._win_reason as a code: 0 none, 1 team is home, 10 + c the other team is lost.
    fn win_reason(&self, team: usize, flags: i32) -> u8 {
        if self.check_for_win(team, flags) {
            return 1;
        }
        let why = losing(self, other(team));
        if why > 0 {
            10 + why as u8
        } else {
            0
        }
    }
}

/// Everything rc_step does, on safe types.  Returns the number of actions written after
/// the header (0 when the game is decided), or a negative error code.
#[allow(clippy::too_many_arguments)]
fn do_step(
    before: &State,
    turn: usize,
    act: &[u8],
    quiet: i32,
    ply: i32,
    fifty: i32,
    max_plies: i32,
    flags: i32,
    buf: &mut Vec<u8>,
) -> Result<usize, i32> {
    let (src, dst) = (act[0] as usize, act[1] as usize);
    let (demote, gift, ep, promo) = (act[2] != 0, act[3], act[4] != 0, act[5]);
    if src >= 64 || dst >= 64 || gift > 6 || promo > 4 {
        return Err(-4);
    }
    let mut st = *before;
    let mover = match st.board[src] {
        k if k >= 0 => k as usize,
        _ => return Err(-4),
    };
    if team_of(mover) != turn || st.board[dst] >= 0 {
        return Err(-4);
    }

    // Game._resolve: is it a reverse castle?  (answered on the position before the move)
    let castle = match st.check_legal_move(src, dst, demote) {
        Some((_, c)) => c,
        None => return Err(-4),
    };
    let mover_pt = st.p[mover].ptype;
    let progress = gift != 0 || mover_pt == PAWN || demote;

    // Game._build_move: the gifted piece is the first off-board piece of that type.
    let opp = other(turn);
    let (mut spawn, mut spawn_pos) = (-1i32, 0u8);
    if gift != 0 {
        let want = gift - 1;
        let found = (0..16).map(|k| opp * 16 + k).find(|&j| !st.p[j].on && st.p[j].ptype == want);
        match found {
            Some(j) => spawn = j as i32,
            None => return Err(-4),
        }
        spawn_pos = if ep { sqi(pawn_home_row(opp), (src & 7) as i32) as u8 } else { src as u8 };
    }
    let promote_to = if promo != 0 { Some(PROMO_TYPES[(promo - 1) as usize]) } else { None };
    let opt = Opt { dst: dst as u8, demote, forced: false, castle };
    let mv = make_mv(&st, mover, src as u8, &opt, spawn, spawn_pos, promote_to);
    st.apply(&mv);

    // Game.step bookkeeping
    let next = if ep { turn } else { opp }; // reverse en passant: the mover goes again
    let quiet_after = if progress { 0 } else { quiet + 1 };

    buf.clear();
    buf.resize(HDR, 0);
    st.write_bytes(&mut buf[0..128]);
    st.write_snapshot(&mut buf[128..192]);
    buf[192] = if next == 0 { b'b' } else { b'w' };
    buf[194] = 255;
    buf[196..198].copy_from_slice(&(quiet_after.clamp(0, 65535) as u16).to_le_bytes());

    // wins and losses take priority over draws
    let (b_reason, w_reason) = (st.win_reason(0, flags), st.win_reason(1, flags));
    if b_reason != 0 && w_reason != 0 {
        buf[193] = ST_DRAW_BOTH;
        return Ok(0);
    }
    if b_reason != 0 || w_reason != 0 {
        buf[193] = ST_WIN;
        buf[194] = if b_reason != 0 { 0 } else { 1 };
        buf[195] = b_reason | w_reason;
        return Ok(0);
    }
    if quiet_after >= fifty {
        buf[193] = ST_DRAW_QUIET;
        return Ok(0);
    }
    // (the repetition draw belongs here -- the caller tests it)
    if max_plies >= 0 && ply >= max_plies {
        buf[193] = ST_MAXPLY;
        return Ok(0);
    }

    // Game._begin_turn
    let (valid_any, n) = enumerate(&mut st, next, buf);
    if n > 0 {
        buf[193] = ST_LIVE;
        return Ok(n);
    }
    if valid_any {
        buf[193] = ST_BLOCKED;
    } else {
        // no legal move: you lose, unless you are already home
        buf[193] = ST_NO_MOVES;
        buf[194] = if st.check_for_win(next, flags) { next as u8 } else { other(next) as u8 };
    }
    Ok(0)
}

#[no_mangle]
pub extern "C" fn rc_step_abi() -> i32 {
    1
}

/// Play one ply.  Returns n >= 0 (n playable actions follow the 200-byte header in `out`),
/// -1 (`cap` actions would not fit; nothing useful written -- retry with a bigger buffer),
/// -3 (bad state), -4 (the action is not playable here), -99 (internal panic).
///
/// `ply` is the ply count AFTER this move; `max_plies` < 0 means no cap.
///
/// # Safety
/// `state` must point at 128 readable bytes, `action` at 6, and `out` at
/// `200 + 6 * cap` writable bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_step(
    state: *const u8,
    turn: u32,
    action: *const u8,
    quiet: i32,
    ply: i32,
    fifty: i32,
    max_plies: i32,
    flags: i32,
    out: *mut u8,
    cap: u32,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let bytes = std::slice::from_raw_parts(state, 128);
        let before = match State::from_bytes(bytes) {
            Some(s) => s,
            None => return -3,
        };
        let act = std::slice::from_raw_parts(action, 6);
        let mut buf: Vec<u8> = Vec::with_capacity(HDR + 6 * 256);
        match do_step(&before, (turn & 1) as usize, act, quiet, ply, fifty, max_plies, flags, &mut buf) {
            Ok(n) => {
                if n > cap as usize {
                    return -1;
                }
                std::ptr::copy_nonoverlapping(buf.as_ptr(), out, buf.len());
                n as i32
            }
            Err(code) => code,
        }
    }));
    r.unwrap_or(-99)
}

/// env.encode_observation for a packed state, from the point of view of `me` (0 black,
/// 1 white): `board` receives 18 * 64 f32 (planes, then rows flipped for black) and `scal`
/// 17 f32.  Only the scalars that come from the position are written (off-board counts,
/// all-on-board flags, mover-is-black); scalars[10..14] (quiet plies, game length,
/// repetition, extra turn) depend on game history and are left 0 for the caller.
///
/// # Safety
/// `state` must point at 128 readable bytes, `board` at 18*64 and `scal` at 17 writable f32.
#[no_mangle]
pub unsafe extern "C" fn rc_observe(state: *const u8, me: u32, board: *mut f32, scal: *mut f32) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let bytes = std::slice::from_raw_parts(state, 128);
        let st = match State::from_bytes(bytes) {
            Some(s) => s,
            None => return -3,
        };
        let me = (me & 1) as usize;
        let board = std::slice::from_raw_parts_mut(board, 18 * 64);
        let scal = std::slice::from_raw_parts_mut(scal, 17);
        board.fill(0.0);
        scal.fill(0.0);
        for i in 0..32 {
            let p = &st.p[i];
            if !p.on {
                continue;
            }
            let side = if team_of(i) == me { 0 } else { 1 };
            let (mut r, c) = crate::rc(p.pos);
            if me == 0 {
                r = 7 - r; // the board is flipped for black
            }
            let sq = sqi(r, c);
            board[(side * 6 + p.ptype as usize) * 64 + sq] = 1.0;
            if p.promoted {
                board[(12 + side) * 64 + sq] = 1.0;
            }
            if p.locked {
                board[(14 + side) * 64 + sq] = 1.0;
            }
            if st.at_home(i) {
                board[(16 + side) * 64 + sq] = 1.0;
            }
        }
        // off-board counts of the giftable types p, r, b, q, n (own first, then the opponent's)
        const TYPES: [u8; 5] = [PAWN, ROOK, BISHOP, QUEEN, KNIGHT];
        const MAXN: [f64; 5] = [8.0, 2.0, 2.0, 1.0, 2.0];
        let _ = KING;
        let mut k = 0;
        for &team in [me, other(me)].iter() {
            for (t, &ty) in TYPES.iter().enumerate() {
                let count = (0..16).filter(|&j| !st.p[team * 16 + j].on && st.p[team * 16 + j].ptype == ty).count();
                scal[k] = (count as f64 / MAXN[t]) as f32;
                k += 1;
            }
        }
        scal[14] = st.all_on_board(me) as u8 as f32;
        scal[15] = st.all_on_board(other(me)) as u8 as f32;
        scal[16] = (me == 0) as u8 as f32;
        0
    }));
    r.unwrap_or(-99)
}
