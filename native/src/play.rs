//! rc_game -- a whole self-play game in Rust (the worker loop of rollout.py).
//!
//! Before this module a worker played every ply through the Python `engine.Game`: build a dict
//! of `Action` objects for every legal move, bind and execute the one played, re-pack the pieces,
//! encode the observation with a per-piece Python loop, softmax and sample with numpy, and keep
//! the books in a `_Recorder`.  A `Session` does all of that natively, with the pieces the
//! library already had: `do_step` (one ply, wins and draws, the next side's actions),
//! `observe_into` (network input planes) and `Net::forward_checked` (the forward pass).
//!
//! A session owns ONE game and the record of its decisions.  Python drives it in three ways:
//!
//! * `rc_game_run`: play on natively -- the recorded side(s) sample the network `mine`, the
//!   other side samples `opp` (not recorded) -- until the game ends, or until a decision wants
//!   something only Python has (a searched decision, or an opponent that is a rule-based bot).
//! * `rc_game_apply`: play an action Python chose (a bot's move; not recorded).
//! * `rc_game_record_apply`: record a decision Python made (a searched one: the observation,
//!   action rows, log-prob and value come from Python) and play it.
//!
//! The draws are supplied by the caller (`u` for the move sampling, `su` for "is this decision
//! searched"), so a game is a pure function of its inputs and can be replayed exactly by the
//! Python reference in check_nativegame.py.
//!
//! What gets recorded matches `rollout._Recorder`: per decision the board planes (uint8), the
//! scalar features, the legal actions' rows in the mover's frame, the chosen index, the log-prob
//! and value the network gave, the mover and the progress potential `env.potential` of the mover
//! BEFORE the move.
//!
//! Repetition: the table of positions seen lives here; `do_step` cannot decide that draw, so it is
//! applied between the fifty-move rule and the ply cap exactly where `Game.step` / `FastGame.child`
//! do.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::mcts::{self, Cfg, Pos};
use crate::nnet::{set_err, Net};
use crate::step::{
    do_step, observe_into, HDR, ST_BLOCKED, ST_DRAW_BOTH, ST_DRAW_QUIET, ST_MAXPLY, ST_NO_MOVES, ST_WIN,
};
use crate::{other, State};

type Key = [u8; 65];

/// How a finished game ended (`rc_game_info`[1]).
const K_LIVE: i32 = 0;
const K_WIN: i32 = 1; // winner + reason code (1 = every piece home, 11..13 = the other team is lost)
const K_DRAW_BOTH: i32 = 2;
const K_DRAW_QUIET: i32 = 3;
const K_REPEAT: i32 = 4;
const K_MAXPLY: i32 = 5;
const K_NO_MOVES: i32 = 6; // the side to move has no legal move; `winner` says who won
const K_BLOCKED: i32 = 7;

/// Observation scalars that depend on history (the rest come from `observe_into`).
const BOARD_LEN: usize = 18 * 64;
const SCAL_LEN: usize = 17;

pub struct Session {
    st: State,
    turn: usize,
    ply: i32,
    quiet: i32,
    fifty: i32,
    max_plies: i32,
    flags: i32,
    rep_limit: i32,
    extra_turn: bool,
    key: Key,
    rep: u32,
    seen: HashMap<Key, u32>,
    actions: Vec<u8>,
    done: bool,
    kind: i32,
    winner: i32,
    reason: i32,
    // the record
    boards: Vec<u8>,
    scalars: Vec<f32>,
    comps: Vec<u8>,
    counts: Vec<i32>,
    chosen: Vec<i32>,
    logp: Vec<f32>,
    value: Vec<f32>,
    phi: Vec<f32>,
    player: Vec<u8>,
    // scratch
    buf: Vec<u8>,
}

/// Number of the side's pieces that are on the board and not home (`env.burden`).
fn burden(st: &State, team: usize) -> i32 {
    (0..16)
        .filter(|&k| {
            let i = team * 16 + k;
            st.p[i].on && !st.at_home(i)
        })
        .count() as i32
}

/// `env.potential`: the opponent's burden minus ours, over 16.
fn potential(st: &State, team: usize) -> f32 {
    ((burden(st, other(team)) - burden(st, team)) as f64 / 16.0) as f32
}

fn position_key(st: &State, turn: usize) -> Key {
    let mut key = [0u8; 65];
    st.write_snapshot(&mut key[0..64]);
    key[64] = if turn == 0 { b'b' } else { b'w' };
    key
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    fn new(
        st: State,
        turn: usize,
        ply: i32,
        quiet: i32,
        fifty: i32,
        max_plies: i32,
        flags: i32,
        rep_limit: i32,
        extra_turn: bool,
        actions: Vec<u8>,
        seen: HashMap<Key, u32>,
    ) -> Session {
        let key = position_key(&st, turn);
        let rep = seen.get(&key).copied().unwrap_or(0);
        let done = actions.is_empty();
        Session {
            st,
            turn,
            ply,
            quiet,
            fifty,
            max_plies,
            flags,
            rep_limit,
            extra_turn,
            key,
            rep,
            seen,
            actions,
            done,
            kind: K_LIVE,
            winner: -1,
            reason: 0,
            boards: Vec::new(),
            scalars: Vec::new(),
            comps: Vec::new(),
            counts: Vec::new(),
            chosen: Vec::new(),
            logp: Vec::new(),
            value: Vec::new(),
            phi: Vec::new(),
            player: Vec::new(),
            buf: Vec::with_capacity(HDR + 6 * 512),
        }
    }

    fn n_actions(&self) -> usize {
        self.actions.len() / 6
    }

    /// The rules the search needs (`fastgame._Config`).
    fn search_cfg(&self) -> Cfg {
        Cfg {
            fifty: self.fifty,
            max_plies: self.max_plies,
            flags: self.flags,
            rep_limit: self.rep_limit,
            obs_cap: -1,
        }
    }

    /// The live position as a search root (`FastGame.from_native`).
    fn root_pos(&self) -> Pos {
        Pos {
            st: self.st.clone(),
            turn: self.turn,
            ply: self.ply,
            quiet: self.quiet,
            rep: self.rep,
            extra_turn: self.extra_turn,
            key: self.key,
            actions: self.actions.clone(),
            done: self.done,
            winner: self.winner,
        }
    }

    /// Play the `idx`-th legal action: `Game.step`, with the repetition table kept here.
    fn apply(&mut self, idx: usize) -> Result<(), i32> {
        if self.done || idx >= self.n_actions() {
            return Err(-4);
        }
        let mut act = [0u8; 6];
        act.copy_from_slice(&self.actions[6 * idx..6 * idx + 6]);
        let ply = self.ply + 1;
        let n_next = do_step(
            &self.st,
            self.turn,
            &act,
            self.quiet,
            ply,
            self.fifty,
            self.max_plies,
            self.flags,
            &mut self.buf,
        )?;
        let new_state = State::from_bytes(&self.buf[0..128]).ok_or(-3)?;
        let mut key: Key = [0u8; 65];
        key.copy_from_slice(&self.buf[128..193]);
        let letter = self.buf[192];
        let status = self.buf[193];
        let winner = self.buf[194];
        let reason = self.buf[195];
        let quiet = u16::from_le_bytes([self.buf[196], self.buf[197]]) as i32;

        // Game.step counts the position before anything else is decided.
        let total = self.seen.get(&key).copied().unwrap_or(0) + 1;
        self.seen.insert(key, total);

        self.st = new_state;
        self.turn = if letter == b'b' { 0 } else { 1 };
        self.ply = ply;
        self.quiet = quiet;
        self.extra_turn = act[4] != 0;
        self.key = key;
        self.rep = total;
        self.actions.clear();
        self.done = true;
        self.winner = -1;
        self.reason = 0;
        // Wins first, then the fifty-move rule, then repetition, then the ply cap, then the next
        // side's moves: the order of Game.step.
        match status {
            ST_WIN => {
                self.kind = K_WIN;
                self.winner = winner as i32;
                self.reason = reason as i32;
            }
            ST_DRAW_BOTH => self.kind = K_DRAW_BOTH,
            ST_DRAW_QUIET => self.kind = K_DRAW_QUIET,
            _ if total as i32 >= self.rep_limit => self.kind = K_REPEAT,
            ST_MAXPLY => self.kind = K_MAXPLY,
            ST_NO_MOVES => {
                self.kind = K_NO_MOVES;
                self.winner = winner as i32;
            }
            ST_BLOCKED => self.kind = K_BLOCKED,
            _ => {
                self.done = false;
                self.kind = K_LIVE;
                self.actions.extend_from_slice(&self.buf[HDR..HDR + 6 * n_next]);
            }
        }
        Ok(())
    }

    /// `fastgame.FastGame.observation`: the position's planes, the history scalars filled in.
    fn observe(&self, board: &mut [f32], scal: &mut [f32]) {
        observe_into(&self.st, self.turn, board, scal);
        let cap = if self.max_plies > 0 { self.max_plies } else { 400 } as f64;
        scal[10] = (self.quiet as f64 / (self.fifty.max(1) as f64)).min(1.0) as f32;
        scal[11] = (self.ply as f64 / cap).min(1.0) as f32;
        scal[12] = (self.rep as f64 / (self.rep_limit.max(1) as f64)) as f32;
        scal[13] = if self.extra_turn { 1.0 } else { 0.0 };
    }

    /// The legal actions' rows in the mover's frame (`net.action_components_array`).
    fn comps_for_net(&self) -> Vec<u8> {
        let mut c = self.actions.clone();
        if self.turn == 0 {
            for row in c.chunks_exact_mut(6) {
                row[0] ^= 56;
                row[1] ^= 56;
            }
        }
        c
    }

    #[allow(clippy::too_many_arguments)]
    fn record(&mut self, board_u8: &[u8], scal: &[f32], comps: &[u8], idx: usize, logp: f32, value: f32) {
        let n = self.n_actions();
        self.phi.push(potential(&self.st, self.turn));
        self.boards.extend_from_slice(board_u8);
        self.scalars.extend_from_slice(scal);
        self.comps.extend_from_slice(comps);
        self.counts.push(n as i32);
        self.chosen.push(idx as i32);
        self.logp.push(logp);
        self.value.push(value);
        self.player.push(self.turn as u8);
    }

    /// One decision by `net` (`net.choose_action`): evaluate, softmax, sample with the uniform
    /// `u` (or take the best move when `greedy`), record it if asked, play it.
    fn decide(&mut self, net: &Net, u: f64, greedy: bool, record: bool) -> Result<(), i32> {
        let n = self.n_actions();
        if self.done || n == 0 {
            return Err(-4);
        }
        let mut board = vec![0f32; BOARD_LEN];
        let mut scal = [0f32; SCAL_LEN];
        self.observe(&mut board, &mut scal);
        let comps = self.comps_for_net();
        let mut logits = vec![0f32; n];
        let mut value = [0f32; 1];
        if let Err(e) = net.forward_checked(1, &board, &scal, &[n as i32], &comps, &mut logits, &mut value) {
            set_err(e);
            return Err(-6);
        }
        // log-softmax in f32, as net.choose_action does
        let zmax = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for z in logits.iter_mut() {
            *z -= zmax;
        }
        let sum: f32 = logits.iter().map(|z| z.exp()).sum();
        let lse = sum.ln();
        let idx = if greedy {
            let mut best = 0usize;
            for (i, z) in logits.iter().enumerate() {
                if *z > logits[best] {
                    best = i;
                }
            }
            best
        } else {
            // the draw of rng.choice(n, p=probs): cumulative sum, normalised, first entry > u
            let mut cdf = Vec::with_capacity(n);
            let mut acc = 0f64;
            for z in logits.iter() {
                acc += ((*z - lse) as f64).exp();
                cdf.push(acc);
            }
            let total = acc;
            let mut pick = n - 1;
            for (i, c) in cdf.iter().enumerate() {
                if c / total > u {
                    pick = i;
                    break;
                }
            }
            pick
        };
        if record {
            let board_u8: Vec<u8> = board.iter().map(|&x| x as u8).collect();
            self.record(&board_u8, &scal, &comps, idx, logits[idx] - lse, value[0]);
        }
        self.apply(idx)
    }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rc_game_abi() -> i32 {
    1
}

/// 1 when `rc_game_search` (the native tree search) is in this build.
#[no_mangle]
pub extern "C" fn rc_game_search_abi() -> i32 {
    1
}

/// Start a session from a position.  `seen_keys` holds `n_seen` 65-byte repetition keys (board
/// fingerprint + side letter) and `seen_counts` their counts: the positions seen so far
/// (empty for a new game).  `actions` are the `n_actions` legal 6-byte rows of the side to move
/// (none = the game is over).  Returns null on a bad input.
///
/// # Safety
/// Every pointer must be valid for the sizes given (`state`: 128 bytes).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_game_new(
    state: *const u8,
    turn: u32,
    ply: i32,
    quiet: i32,
    fifty: i32,
    max_plies: i32,
    flags: i32,
    rep_limit: i32,
    extra_turn: i32,
    actions: *const u8,
    n_actions: u32,
    seen_keys: *const u8,
    seen_counts: *const u32,
    n_seen: u32,
) -> *mut Session {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let st = match State::from_bytes(std::slice::from_raw_parts(state, 128)) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let acts = if n_actions == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(actions, 6 * n_actions as usize).to_vec()
        };
        let mut seen: HashMap<Key, u32> = HashMap::new();
        if n_seen > 0 {
            let keys = std::slice::from_raw_parts(seen_keys, 65 * n_seen as usize);
            let counts = std::slice::from_raw_parts(seen_counts, n_seen as usize);
            for (k, c) in keys.chunks_exact(65).zip(counts.iter()) {
                let mut key = [0u8; 65];
                key.copy_from_slice(k);
                seen.insert(key, *c);
            }
        }
        let s = Session::new(
            st,
            (turn & 1) as usize,
            ply,
            quiet,
            fifty,
            max_plies,
            flags,
            rep_limit,
            extra_turn != 0,
            acts,
            seen,
        );
        Box::into_raw(Box::new(s))
    }));
    r.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `s` must come from `rc_game_new` (or be null) and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rc_game_free(s: *mut Session) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// Play on natively.  A move is made by `mine` when the mover's bit (bit 0 black, bit 1 white) is
/// set in `record_mask` -- recorded -- and otherwise by `opp` (sampled, not recorded); with a null
/// `opp` the other side's turn is handed back.  `u[j]` is the uniform for the j-th decision of
/// this call and `su[j]` decides whether a recorded decision with a choice is searched: it is
/// when `su[j] < frac`.
///
/// Returns 0 the game is over; 1 a recorded decision is to be searched (nothing was consumed:
/// call `rc_game_position` / `rc_game_record_apply`); 2 the uniforms ran out (call again with new
/// ones); 3 an unrecorded side's turn with no `opp` net (play it with `rc_game_apply`);
/// negative: an error (-6: the network refused, see `rc_net_error`).
///
/// # Safety
/// `mine` (and `opp` unless null) must be live nets; `u` and `su` valid for `n_u` doubles.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_game_run(
    s: *mut Session,
    mine: *const Net,
    opp: *const Net,
    record_mask: u32,
    u: *const f64,
    su: *const f64,
    n_u: u32,
    frac: f64,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || mine.is_null() {
            return -1;
        }
        let s = &mut *s;
        let mine = &*mine;
        let opp = if opp.is_null() { None } else { Some(&*opp) };
        let u = std::slice::from_raw_parts(u, n_u as usize);
        let su = std::slice::from_raw_parts(su, n_u as usize);
        let mut j = 0usize;
        while !s.done {
            if j >= u.len() {
                return 2;
            }
            let recorded = (record_mask >> s.turn) & 1 == 1;
            let res = if recorded {
                if frac > 0.0 && s.n_actions() > 1 && su[j] < frac {
                    return 1;
                }
                s.decide(mine, u[j], false, true)
            } else if let Some(o) = opp {
                s.decide(o, u[j], false, false)
            } else {
                return 3;
            };
            if let Err(code) = res {
                return code;
            }
            j += 1;
        }
        0
    }));
    r.unwrap_or(-99)
}

/// Play the `idx`-th legal action without recording it (a bot's move).  0, or a negative error.
///
/// # Safety
/// `s` must be a live session.
#[no_mangle]
pub unsafe extern "C" fn rc_game_apply(s: *mut Session, idx: u32) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return -1;
        }
        match (*s).apply(idx as usize) {
            Ok(()) => 0,
            Err(c) => c,
        }
    }));
    r.unwrap_or(-99)
}

/// Record a decision made in Python (the position it faced as `board` uint8 (18*64) and `scal`
/// f32 (17), the legal actions' rows in the mover's frame, the chosen index, the log-prob and
/// value the network gave) and play it.  `n` must be the number of legal actions.
///
/// # Safety
/// Pointers valid for the sizes above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_game_record_apply(
    s: *mut Session,
    board: *const u8,
    scal: *const f32,
    comps: *const u8,
    n: u32,
    idx: u32,
    logp: f32,
    value: f32,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return -1;
        }
        let s = &mut *s;
        if s.done || n as usize != s.n_actions() || idx >= n {
            return -4;
        }
        let board = std::slice::from_raw_parts(board, BOARD_LEN);
        let scal = std::slice::from_raw_parts(scal, SCAL_LEN);
        let comps = std::slice::from_raw_parts(comps, 6 * n as usize);
        s.record(board, scal, comps, idx as usize, logp, value);
        match s.apply(idx as usize) {
            Ok(()) => 0,
            Err(c) => c,
        }
    }));
    r.unwrap_or(-99)
}

/// Facts about the game, 16 i32 in `out`:
/// 0 done (0/1)   1 how it ended (0 live, 1 win, 2 draw both won, 3 draw fifty-move, 4 draw
/// repetition, 5 draw ply cap, 6 no legal move (see winner), 7 draw blocked)   2 winner (0 black,
/// 1 white, -1 none)   3 win reason (1 every piece home, 11..13 the other team is lost)
/// 4 side to move (0 black, 1 white)   5 ply   6 quiet plies   7 times this position has occurred
/// 8 extra turn (the last move was a reverse en passant)   9 number of legal actions
/// 10 decisions recorded   11 action rows recorded   12 positions in the repetition table.
///
/// # Safety
/// `out` must hold 16 i32.
#[no_mangle]
pub unsafe extern "C" fn rc_game_info(s: *const Session, out: *mut i32) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return -1;
        }
        let s = &*s;
        let o = std::slice::from_raw_parts_mut(out, 16);
        o.fill(0);
        o[0] = s.done as i32;
        o[1] = s.kind;
        o[2] = s.winner;
        o[3] = s.reason;
        o[4] = s.turn as i32;
        o[5] = s.ply;
        o[6] = s.quiet;
        o[7] = s.rep as i32;
        o[8] = s.extra_turn as i32;
        o[9] = s.n_actions() as i32;
        o[10] = s.counts.len() as i32;
        o[11] = (s.comps.len() / 6) as i32;
        o[12] = s.seen.len() as i32;
        0
    }));
    r.unwrap_or(-99)
}

/// The current position: 128 state bytes, its 65-byte repetition key, and the legal action rows
/// (up to `cap`).  Returns the number of legal actions (they are only written if `cap` allows).
///
/// # Safety
/// `state` 128 bytes, `key` 65 bytes, `actions` 6 * `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn rc_game_position(
    s: *const Session,
    state: *mut u8,
    key: *mut u8,
    actions: *mut u8,
    cap: u32,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return -1;
        }
        let s = &*s;
        let st = std::slice::from_raw_parts_mut(state, 128);
        s.st.write_bytes(st);
        std::ptr::copy_nonoverlapping(s.key.as_ptr(), key, 65);
        let n = s.n_actions();
        if n <= cap as usize {
            std::ptr::copy_nonoverlapping(s.actions.as_ptr(), actions, 6 * n);
        }
        n as i32
    }));
    r.unwrap_or(-99)
}

/// The repetition table (position keys and counts) -- what a search rooted here needs as the
/// positions seen before it.  Writes up to `cap` entries; returns how many there are.
///
/// # Safety
/// `keys` 65 * `cap` bytes, `counts` `cap` u32.
#[no_mangle]
pub unsafe extern "C" fn rc_game_seen(s: *const Session, keys: *mut u8, counts: *mut u32, cap: u32) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return -1;
        }
        let s = &*s;
        if s.seen.len() <= cap as usize {
            for (i, (k, c)) in s.seen.iter().enumerate() {
                std::ptr::copy_nonoverlapping(k.as_ptr(), keys.add(65 * i), 65);
                *counts.add(i) = *c;
            }
        }
        s.seen.len() as i32
    }));
    r.unwrap_or(-99)
}

/// Copy the record out: T decisions, A action rows in all (see `rc_game_info` 10 / 11).
///   boards (T,18,8,8) u8   scalars (T,17) f32   comps (A,6) u8   counts (T) i32   chosen (T) i32
///   logp (T) f32   value (T) f32   phi (T) f32   player (T) u8
///
/// # Safety
/// Every buffer must be valid for the sizes above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_game_export(
    s: *const Session,
    boards: *mut u8,
    scalars: *mut f32,
    comps: *mut u8,
    counts: *mut i32,
    chosen: *mut i32,
    logp: *mut f32,
    value: *mut f32,
    phi: *mut f32,
    player: *mut u8,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() {
            return -1;
        }
        let s = &*s;
        let t = s.counts.len();
        std::ptr::copy_nonoverlapping(s.boards.as_ptr(), boards, s.boards.len());
        std::ptr::copy_nonoverlapping(s.scalars.as_ptr(), scalars, s.scalars.len());
        std::ptr::copy_nonoverlapping(s.comps.as_ptr(), comps, s.comps.len());
        std::ptr::copy_nonoverlapping(s.counts.as_ptr(), counts, t);
        std::ptr::copy_nonoverlapping(s.chosen.as_ptr(), chosen, t);
        std::ptr::copy_nonoverlapping(s.logp.as_ptr(), logp, t);
        std::ptr::copy_nonoverlapping(s.value.as_ptr(), value, t);
        std::ptr::copy_nonoverlapping(s.phi.as_ptr(), phi, t);
        std::ptr::copy_nonoverlapping(s.player.as_ptr(), player, t);
        0
    }));
    r.unwrap_or(-99)
}

/// A searched decision (the `Searcher.search` + `net.search_action` of Python, natively): the
/// network evaluates the position, `noise` of the Dirichlet draw `dir` (n values, or null) is mixed
/// into the root priors, then `sims` PUCT simulations run (`batch` descents per network call).
/// Nothing is played or recorded: Python draws the move from the visits and then calls
/// `rc_game_record_apply` with the board / scalars / action rows written here.
///
/// Outputs (n = number of legal actions, returned):
///   `visits` n i64 -- root visit counts        `probs` n f64 -- the network's own policy
///   `board` 18*64 u8, `scal` 17 f32, `comps` 6n u8 -- the observation, as `rc_game_record_apply` wants it
///   `info` 8 f64: 0 network value of the root, 1 simulations run, 2 index of a move that wins on
///   the spot (-1 none), 3 seconds in the network, 4 seconds stepping positions, 5 network leaves
///
/// Returns n, or a negative error (-4 the game is over / no moves, -6 the network refused).
///
/// # Safety
/// `s`, `net` live; buffers valid for the sizes above (`cap` is the room for `visits`, `probs`
/// and `comps`, in actions: if n > cap nothing is searched and n is returned).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_game_search(
    s: *const Session,
    net: *const Net,
    noise: f64,
    dir: *const f64,
    sims: u32,
    batch: u32,
    c_puct: f64,
    vl: f64,
    cap: u32,
    visits: *mut i64,
    probs: *mut f64,
    board: *mut u8,
    scal: *mut f32,
    comps: *mut u8,
    info: *mut f64,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || net.is_null() {
            return -1;
        }
        let s = &*s;
        let net = &*net;
        let n = s.n_actions();
        if s.done || n == 0 {
            return -4;
        }
        if n > cap as usize {
            return n as i32;
        }
        let cfg = s.search_cfg();
        let root = s.root_pos();
        // the root's own evaluation: value, policy, and what Python records for the decision
        let ev = match mcts::evaluate(net, &cfg, &[&root]) {
            Ok(e) => e,
            Err(c) => return c,
        };
        let value = ev.values[0] as f64;
        let net_probs = ev.probs[0].clone();
        let mut priors = net_probs.clone();
        if noise > 0.0 && n > 1 && !dir.is_null() {
            let d = std::slice::from_raw_parts(dir, n);
            for (p, &x) in priors.iter_mut().zip(d.iter()) {
                *p = (1.0 - noise) * *p + noise * x;
            }
        }
        let out = match mcts::search(
            net,
            &cfg,
            root,
            &s.seen,
            value,
            priors,
            sims as usize,
            batch as usize,
            c_puct,
            vl,
        ) {
            Ok(o) => o,
            Err(c) => return c,
        };
        std::ptr::copy_nonoverlapping(out.visits.as_ptr(), visits, n);
        std::ptr::copy_nonoverlapping(net_probs.as_ptr(), probs, n);
        let b = std::slice::from_raw_parts_mut(board, BOARD_LEN);
        for (d, &x) in b.iter_mut().zip(ev.boards.iter()) {
            *d = x as u8;
        }
        std::ptr::copy_nonoverlapping(ev.scalars.as_ptr(), scal, SCAL_LEN);
        std::ptr::copy_nonoverlapping(ev.comps.as_ptr(), comps, 6 * n);
        let o = std::slice::from_raw_parts_mut(info, 8);
        o.fill(0.0);
        o[0] = value;
        o[1] = out.total as f64;
        o[2] = out.win_idx as f64;
        o[3] = out.t_eval;
        o[4] = out.t_step;
        o[5] = out.evals as f64;
        n as i32
    }));
    r.unwrap_or(-99)
}

/// 1 when `rc_search_run` (the stoppable, time-limited search for play.py) is in this build.
#[no_mangle]
pub extern "C" fn rc_search_abi() -> i32 {
    1
}

/// A search of a position handed over by Python (play.py / review.py: `fastgame.FastGame` fields),
/// run on the CALLING thread until `sims` simulations are done, `deadline_ms` have passed, a move
/// that wins on the spot is found, or `*stop` becomes non-zero (polled between rounds).  ctypes
/// drops the GIL for the call, so the window stays responsive while a Python thread sits in here.
///
/// The root's `root_value` and `priors` (n f64, noise already mixed in) come from the caller.
/// While it runs, after every round, the root's numbers are written into the caller's buffers:
///   `visits` n i64, `w` n f64 (summed from the mover's view), then `info` (8 f64):
///   0 simulations so far (written LAST)   1 index of a move that wins on the spot (-1 none)
///   2 sweep moves checked   3 sweep moves to check   4 seconds in the network   5 seconds
///   stepping positions   6 network leaves   7 1 when finished
/// `sweep` != 0 first plays every root move once looking for an exact result (for at most
/// `sweep_share` of the time budget).  `obs_cap` <= 0: the game-length feature is divided by the
/// rules' cap (400 if none); otherwise by `obs_cap`.
///
/// Returns 0, or a negative error (-4 bad input, -6 the network refused, -99 a panic).
///
/// # Safety
/// Every pointer valid for the sizes given (`state` 128 bytes; `actions` 6 * n; `seen_keys`
/// 65 * n_seen; `stop` one i32, which another thread may write while this runs).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rc_search_run(
    state: *const u8,
    turn: u32,
    ply: i32,
    quiet: i32,
    fifty: i32,
    max_plies: i32,
    flags: i32,
    rep_limit: i32,
    obs_cap: i32,
    extra_turn: i32,
    rep: u32,
    actions: *const u8,
    n_actions: u32,
    seen_keys: *const u8,
    seen_counts: *const u32,
    n_seen: u32,
    net: *const Net,
    root_value: f64,
    priors: *const f64,
    sims: u32,
    batch: u32,
    c_puct: f64,
    vl: f64,
    deadline_ms: f64,
    sweep: i32,
    sweep_share: f64,
    stop: *const i32,
    visits: *mut i64,
    w: *mut f64,
    info: *mut f64,
) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| {
        if state.is_null() || net.is_null() || actions.is_null() || priors.is_null() || n_actions == 0 {
            return -4;
        }
        let net = &*net;
        let st = match State::from_bytes(std::slice::from_raw_parts(state, 128)) {
            Some(s) => s,
            None => return -4,
        };
        let n = n_actions as usize;
        let turn = (turn & 1) as usize;
        let mut seen: HashMap<Key, u32> = HashMap::new();
        if n_seen > 0 {
            let keys = std::slice::from_raw_parts(seen_keys, 65 * n_seen as usize);
            let counts = std::slice::from_raw_parts(seen_counts, n_seen as usize);
            for (k, c) in keys.chunks_exact(65).zip(counts.iter()) {
                let mut key = [0u8; 65];
                key.copy_from_slice(k);
                seen.insert(key, *c);
            }
        }
        let cfg = Cfg { fifty, max_plies, flags, rep_limit, obs_cap };
        let root = Pos {
            st,
            turn,
            ply,
            quiet,
            rep,
            extra_turn: extra_turn != 0,
            key: position_key(&st, turn),
            actions: std::slice::from_raw_parts(actions, 6 * n).to_vec(),
            done: false,
            winner: -1,
        };
        let pri = std::slice::from_raw_parts(priors, n).to_vec();
        let t0 = std::time::Instant::now();
        let budget = if deadline_ms > 0.0 { Some(std::time::Duration::from_secs_f64(deadline_ms / 1000.0)) } else { None };
        let opts = mcts::Opts {
            sims: sims as usize,
            batch: batch as usize,
            c_puct,
            vl,
            sweep: sweep != 0,
            deadline: budget.map(|b| t0 + b),
            sweep_until: budget.map(|b| t0 + b.mul_f64(sweep_share.clamp(0.0, 1.0))),
        };
        let stop_now = || !stop.is_null() && std::ptr::read_volatile(stop) != 0;
        let vis = std::slice::from_raw_parts_mut(visits, n);
        let ww = std::slice::from_raw_parts_mut(w, n);
        let inf = std::slice::from_raw_parts_mut(info, 8);
        inf.fill(0.0);
        inf[1] = -1.0;
        let mut publish = |p: &mcts::Progress| {
            vis.copy_from_slice(p.visits);
            ww.copy_from_slice(p.w);
            inf[1] = p.win_idx as f64;
            inf[2] = p.sweep_done as f64;
            inf[3] = p.sweep_total as f64;
            inf[4] = p.t_eval;
            inf[5] = p.t_step;
            inf[6] = p.evals as f64;
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            inf[0] = p.total as f64;
        };
        match mcts::search_live(net, &cfg, root, &seen, root_value, pri, &opts, &stop_now, &mut publish) {
            Ok(out) => {
                vis.copy_from_slice(&out.visits);
                ww.copy_from_slice(&out.w);
                inf[1] = out.win_idx as f64;
                inf[2] = out.sweep_done as f64;
                inf[3] = out.sweep_total as f64;
                inf[4] = out.t_eval;
                inf[5] = out.t_step;
                inf[6] = out.evals as f64;
                inf[7] = 1.0;
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                inf[0] = out.total as f64;
                0
            }
            Err(c) => c,
        }
    }));
    r.unwrap_or(-99)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The starting position: two kings, everything else off the board (engine.Game.reset).
    fn start_state() -> [u8; 128] {
        let mut b = [0u8; 128];
        for team in 0..2 {
            // per side: 8 pawns, king, 2 rooks, 2 bishops, queen, 2 knights ("pkrbqn")
            let types = [0u8, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 5];
            for (k, &t) in types.iter().enumerate() {
                let i = team * 16 + k;
                b[4 * i + 1] = 255;
                b[4 * i + 2] = t;
            }
        }
        b[4 * 8] = 1;
        b[4 * 8 + 1] = 3 * 8 + 5; // black king (3, 5)
        b[4 * 24] = 1;
        b[4 * 24 + 1] = 4 * 8 + 3; // white king (4, 3)
        b
    }

    fn new_session(max_plies: i32) -> Session {
        let bytes = start_state();
        let mut st = State::from_bytes(&bytes).unwrap();
        let mut acts = Vec::new();
        let (_, n) = crate::enumerate(&mut st, 0, &mut acts);
        assert!(n > 0);
        Session::new(st, 0, 0, 0, 50, max_plies, 0, 3, false, acts, HashMap::new())
    }

    /// Random legal play must always terminate, keep the record consistent and count repeats.
    #[test]
    fn random_games_are_consistent() {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut ended = [0usize; 8];
        for _ in 0..300 {
            let mut s = new_session(300);
            let mut plies = 0;
            while !s.done {
                let n = s.n_actions();
                assert!(n > 0);
                let idx = (rnd() % n as u64) as usize;
                // record a fake decision so the export bookkeeping is exercised
                let comps = s.comps_for_net();
                let board = vec![0u8; BOARD_LEN];
                let scal = [0f32; SCAL_LEN];
                s.record(&board, &scal, &comps, idx, -1.0, 0.0);
                s.apply(idx).expect("a legal action must apply");
                plies += 1;
                assert_eq!(s.ply, plies);
                assert!(plies <= 300);
                assert!(s.rep >= 1);
            }
            ended[s.kind as usize] += 1;
            assert_eq!(s.counts.len(), plies as usize);
            assert_eq!(s.comps.len() / 6, s.counts.iter().map(|&c| c as usize).sum::<usize>());
            assert_eq!(s.boards.len(), plies as usize * BOARD_LEN);
            if s.kind == K_WIN || s.kind == K_NO_MOVES {
                assert!(s.winner == 0 || s.winner == 1);
            }
            if s.kind == K_REPEAT {
                assert!(s.rep as i32 >= 3);
            }
        }
        // 300 games of random play must have ended in more than one way
        assert!(ended.iter().filter(|&&c| c > 0).count() >= 2, "{:?}", ended);
    }

    /// Replaying the same action indices gives the same game (pure function of its inputs).
    #[test]
    fn replay_is_deterministic() {
        let mut s1 = new_session(200);
        let mut picks = Vec::new();
        let mut x = 12345u64;
        while !s1.done {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((x >> 33) % s1.n_actions() as u64) as usize;
            picks.push(idx);
            s1.apply(idx).unwrap();
        }
        let mut s2 = new_session(200);
        for &i in &picks {
            s2.apply(i).unwrap();
        }
        assert!(s2.done);
        assert_eq!((s1.kind, s1.winner, s1.reason, s1.ply, s1.rep), (s2.kind, s2.winner, s2.reason, s2.ply, s2.rep));
        assert_eq!(s1.key, s2.key);
    }

    /// The search's `Pos::child` must agree with `Session::apply` on every ply of a game,
    /// repetition draws included (the base table is the session's own `seen`).
    #[test]
    fn search_positions_follow_the_session() {
        let mut seed = 0xD1B54A32D192ED03u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut buf = Vec::new();
        for _ in 0..100 {
            let mut s = new_session(250);
            let cfg = s.search_cfg();
            while !s.done {
                let idx = (rnd() % s.n_actions() as u64) as usize;
                let before = s.root_pos();
                let seen = s.seen.clone();
                let child = before
                    .child(idx, &cfg, &mut buf, |k| seen.get(k).copied().unwrap_or(0))
                    .expect("a legal action must step");
                s.apply(idx).unwrap();
                let after = s.root_pos();
                assert_eq!(child.turn, after.turn);
                assert_eq!(child.ply, after.ply);
                assert_eq!(child.quiet, after.quiet);
                assert_eq!(child.rep, after.rep);
                assert_eq!(child.extra_turn, after.extra_turn);
                assert_eq!(child.key, after.key);
                assert_eq!(child.done, after.done);
                assert_eq!(child.winner, after.winner);
                assert_eq!(child.actions, after.actions);
            }
        }
    }
}
