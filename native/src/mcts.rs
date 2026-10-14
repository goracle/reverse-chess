//! rc_mcts -- the self-play tree search of mcts.py (`Searcher.search`), natively.
//!
//! A line-for-line port of what `net.search_action` asks of `mcts.Searcher` in training:
//!
//! * the tree holds positions as plain values (`Pos`, the `FastGame` of fastgame.py), each child
//!   made with one `do_step` -- the move applied, wins and draws decided, the next side's moves
//!   enumerated -- and the repetition draw applied exactly where `FastGame.child` applies it
//!   (the count of a position is the number of times it occurred before the root, per the base
//!   table, plus the times it occurred on the line from the root, plus one);
//! * PUCT selection with the first-play-urgency baseline built from real visits only, exact
//!   terminal values (a position that is over is scored +1 / -1 / 0 for the side to move, never by
//!   the network), a node with a winning move is *solved* and descent stops everywhere else;
//! * batched descents with virtual loss (`batch` > 1; each descent in flight counts as a visit
//!   worth `(1 - vl) * q - vl`), the new leaves of a round evaluated with ONE forward pass;
//! * the network is `Net::forward_checked`, softmax in f64 as `SearchEvaluator` does.
//!
//! Not here: the root sweep (`sweep_k` != 0; rollout falls back to the Python searcher), the
//! Dirichlet noise (numpy draws it; it arrives as an array to mix into the root priors) and the
//! time limits / threads of play.py (that search stays Python).
//!
//! Everything here is safe Rust on top of the library's own pieces; the C entry points are in
//! play.rs (`rc_game_search`), because they search from a `Session`'s position.

use std::collections::HashMap;
use std::cmp::Ordering;
use std::time::Instant;

use crate::nnet::{set_err, Net};
use crate::step::{
    do_step, observe_into, HDR, ST_BLOCKED, ST_DRAW_BOTH, ST_DRAW_QUIET, ST_MAXPLY, ST_NO_MOVES, ST_WIN,
};
use crate::State;

pub(crate) type Key = [u8; 65];

/// mcts.FPU_REDUCTION: an unvisited move is assumed this much worse than its parent.
const FPU_REDUCTION: f64 = 0.25;
const NONE: usize = usize::MAX;
const BOARD_LEN: usize = 18 * 64;
const SCAL_LEN: usize = 17;

/// The rules rc_step needs (fastgame._Config).
#[derive(Clone, Copy)]
pub(crate) struct Cfg {
    pub fifty: i32,
    pub max_plies: i32,
    pub flags: i32,
    pub rep_limit: i32,
    /// what the game-length feature is divided by; <= 0: the rules' cap, or 400 when uncapped
    /// (play.py's numpy Advisor normalises by its own `max_plies`, which can differ)
    pub obs_cap: i32,
}

/// One position (fastgame.FastGame): never mutated, `child` makes the next one.
#[derive(Clone)]
pub(crate) struct Pos {
    pub st: State,
    pub turn: usize,
    pub ply: i32,
    pub quiet: i32,
    pub rep: u32,
    pub extra_turn: bool,
    pub key: Key,
    /// the legal actions of the side to move, 6 bytes each (empty when the game is over)
    pub actions: Vec<u8>,
    pub done: bool,
    /// the winner of a finished game (0 black, 1 white), -1 for none / not over
    pub winner: i32,
}

impl Pos {
    pub(crate) fn n_actions(&self) -> usize {
        self.actions.len() / 6
    }

    /// The position after the `idx`-th legal action (`FastGame.child`).  `before(key)` says how
    /// often that position occurred before this move (base table + the line from the root).
    pub(crate) fn child(
        &self,
        idx: usize,
        cfg: &Cfg,
        buf: &mut Vec<u8>,
        before: impl FnOnce(&Key) -> u32,
    ) -> Result<Pos, i32> {
        if self.done || idx >= self.n_actions() {
            return Err(-4);
        }
        let mut act = [0u8; 6];
        act.copy_from_slice(&self.actions[6 * idx..6 * idx + 6]);
        let ply = self.ply + 1;
        let n_next = do_step(&self.st, self.turn, &act, self.quiet, ply, cfg.fifty, cfg.max_plies, cfg.flags, buf)?;
        let st = State::from_bytes(&buf[0..128]).ok_or(-3)?;
        let mut key: Key = [0u8; 65];
        key.copy_from_slice(&buf[128..193]);
        let status = buf[193];
        let winner = buf[194] as i32;
        let quiet = u16::from_le_bytes([buf[196], buf[197]]) as i32;
        let total = before(&key) + 1;
        let mut p = Pos {
            st,
            turn: if buf[192] == b'b' { 0 } else { 1 },
            ply,
            quiet,
            rep: total,
            extra_turn: act[4] != 0,
            key,
            actions: Vec::new(),
            done: true,
            winner: -1,
        };
        // wins, the fifty-move rule, repetition, the ply cap, the next side's moves: Game.step's order
        match status {
            ST_WIN => p.winner = winner,
            ST_DRAW_BOTH | ST_DRAW_QUIET => {}
            _ if total as i32 >= cfg.rep_limit => {}
            ST_MAXPLY => {}
            ST_NO_MOVES => p.winner = winner,
            ST_BLOCKED => {}
            _ => {
                p.done = false;
                p.actions.extend_from_slice(&buf[HDR..HDR + 6 * n_next]);
            }
        }
        Ok(p)
    }

    /// `FastGame.observation`: planes plus the history scalars.
    fn observe(&self, cfg: &Cfg, board: &mut [f32], scal: &mut [f32]) {
        observe_into(&self.st, self.turn, board, scal);
        let cap = if cfg.obs_cap > 0 {
            cfg.obs_cap
        } else if cfg.max_plies > 0 {
            cfg.max_plies
        } else {
            400
        } as f64;
        scal[10] = (self.quiet as f64 / (cfg.fifty.max(1) as f64)).min(1.0) as f32;
        scal[11] = (self.ply as f64 / cap).min(1.0) as f32;
        scal[12] = (self.rep as f64 / (cfg.rep_limit.max(1) as f64)) as f32;
        scal[13] = if self.extra_turn { 1.0 } else { 0.0 };
    }

    /// Exact result of a finished game for the side to move (`mcts.terminal_value`).
    fn terminal_value(&self) -> f64 {
        if self.winner < 0 {
            0.0
        } else if self.winner as usize == self.turn {
            1.0
        } else {
            -1.0
        }
    }
}

/// What a batch of positions looks like to the network, and what it said.
pub(crate) struct EvalBatch {
    pub boards: Vec<f32>,
    pub scalars: Vec<f32>,
    pub comps: Vec<u8>,
    pub values: Vec<f32>,
    /// per position: softmax of the logits over its legal actions, in f64
    pub probs: Vec<Vec<f64>>,
}

/// `SearchEvaluator.analyse_many`: ONE forward pass for all `ps`.
pub(crate) fn evaluate(net: &Net, cfg: &Cfg, ps: &[&Pos]) -> Result<EvalBatch, i32> {
    let b = ps.len();
    let mut boards = vec![0f32; b * BOARD_LEN];
    let mut scalars = vec![0f32; b * SCAL_LEN];
    let mut counts: Vec<i32> = Vec::with_capacity(b);
    let mut comps: Vec<u8> = Vec::new();
    for (k, p) in ps.iter().enumerate() {
        p.observe(cfg, &mut boards[k * BOARD_LEN..(k + 1) * BOARD_LEN], &mut scalars[k * SCAL_LEN..(k + 1) * SCAL_LEN]);
        counts.push(p.n_actions() as i32);
        let start = comps.len();
        comps.extend_from_slice(&p.actions);
        if p.turn == 0 {
            for row in comps[start..].chunks_exact_mut(6) {
                row[0] ^= 56;
                row[1] ^= 56;
            }
        }
    }
    let total = comps.len() / 6;
    let mut logits = vec![0f32; total];
    let mut values = vec![0f32; b];
    if let Err(e) = net.forward_checked(b, &boards, &scalars, &counts, &comps, &mut logits, &mut values) {
        set_err(e);
        return Err(-6);
    }
    let mut probs = Vec::with_capacity(b);
    let mut start = 0usize;
    for &n in counts.iter() {
        let z = &logits[start..start + n as usize];
        start += n as usize;
        let zmax = z.iter().map(|&x| x as f64).fold(f64::NEG_INFINITY, f64::max);
        let mut p: Vec<f64> = z.iter().map(|&x| (x as f64 - zmax).exp()).collect();
        let sum: f64 = p.iter().sum();
        for x in p.iter_mut() {
            *x /= sum;
        }
        probs.push(p);
    }
    Ok(EvalBatch { boards, scalars, comps, values, probs })
}

struct Node {
    pos: Pos,
    turn: usize,
    terminal: bool,
    /// network (or exact, if terminal) value, mover's view
    value: f64,
    priors: Vec<f64>,
    n: Vec<i64>,
    w: Vec<f64>,
    /// virtual losses: batched descents in flight per edge, and their sum
    vl: Vec<i64>,
    nvl: i64,
    total: i64,
    children: Vec<usize>,
    win_idx: Option<usize>,
    pending: bool,
}

impl Node {
    fn expandable(pos: Pos, value: f64, priors: Vec<f64>) -> Node {
        let n = priors.len();
        Node {
            turn: pos.turn,
            pos,
            terminal: false,
            value,
            priors,
            n: vec![0; n],
            w: vec![0.0; n],
            vl: vec![0; n],
            nvl: 0,
            total: 0,
            children: vec![NONE; n],
            win_idx: None,
            pending: false,
        }
    }

    fn pending(pos: Pos) -> Node {
        let mut node = Node::expandable(pos, 0.0, Vec::new());
        node.pending = true;
        node
    }

    fn terminal(pos: Pos) -> Node {
        let value = pos.terminal_value();
        let mut node = Node::expandable(pos, value, Vec::new());
        node.terminal = true;
        node
    }

    /// Give a pending node its network result.
    fn fill(&mut self, value: f64, priors: Vec<f64>) {
        let n = priors.len();
        self.value = value;
        self.priors = priors;
        self.n = vec![0; n];
        self.w = vec![0.0; n];
        self.vl = vec![0; n];
        self.children = vec![NONE; n];
        self.pending = false;
    }
}

type Path = Vec<(usize, usize)>;

struct Tree<'a> {
    net: &'a Net,
    cfg: &'a Cfg,
    base: &'a HashMap<Key, u32>,
    nodes: Vec<Node>,
    c_puct: f64,
    vl: f64,
    buf: Vec<u8>,
    t_eval: f64,
    t_step: f64,
    evals: u64,
}

impl<'a> Tree<'a> {
    /// `mcts._edge_value`: `v` is from `child`'s mover's view; return it from `parent`'s.
    fn edge_value(&self, parent: usize, child: usize, v: f64) -> f64 {
        if self.nodes[child].turn == self.nodes[parent].turn {
            v
        } else {
            -v
        }
    }

    /// `Searcher._select`: PUCT with the FPU baseline from real visits only.
    fn select(&self, ni: usize) -> usize {
        let node = &self.nodes[ni];
        if let Some(w) = node.win_idx {
            return w;
        }
        let sum_w: f64 = node.w.iter().sum();
        let parent_q = if node.total > 0 { sum_w / node.total as f64 } else { node.value };
        let fpu = parent_q - FPU_REDUCTION;
        let total = (node.total + node.nvl) as f64;
        let sqrt_t = (total + 1.0).sqrt();
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..node.n.len() {
            let (q, n_eff) = if node.nvl == 0 {
                let ne = node.n[i];
                (if ne > 0 { node.w[i] / ne as f64 } else { fpu }, ne)
            } else {
                // each descent in flight counts as a visit worth v_virt
                let ne = node.n[i] + node.vl[i];
                let q_now = if node.n[i] > 0 { node.w[i] / node.n[i] as f64 } else { fpu };
                let v_virt = q_now * (1.0 - self.vl) - self.vl;
                (if ne > 0 { (node.w[i] + node.vl[i] as f64 * v_virt) / ne as f64 } else { fpu }, ne)
            };
            let u = self.c_puct * node.priors[i] * sqrt_t / (1.0 + n_eff as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// `Searcher._backup`: add `v` (mover's view at the end of `path`) to every edge on the path
    /// and take back the virtual loss each descent put there.
    fn backup(&mut self, path: &[(usize, usize)], mut v: f64) {
        for &(p, i) in path.iter().rev() {
            let c = self.nodes[p].children[i];
            v = self.edge_value(p, c, v);
            let node = &mut self.nodes[p];
            node.w[i] += v;
            node.n[i] += 1;
            node.total += 1;
            node.vl[i] -= 1;
            node.nvl -= 1;
        }
    }

    fn undo(&mut self, path: &[(usize, usize)]) {
        for &(p, i) in path {
            self.nodes[p].vl[i] -= 1;
            self.nodes[p].nvl -= 1;
        }
    }

    /// The position after edge `i` of the node the path ends at (`FastGame.child`): repetition
    /// counts the base table plus the positions on the line below the root.
    fn make_child(&mut self, path: &[(usize, usize)], ni: usize, i: usize) -> Result<Pos, i32> {
        let nodes = &self.nodes;
        let base = self.base;
        let cfg = self.cfg;
        let line: Vec<&Key> = path[1..].iter().map(|&(n, _)| &nodes[n].pos.key).collect();
        nodes[ni].pos.child(i, cfg, &mut self.buf, |key| {
            base.get(key).copied().unwrap_or(0) + line.iter().filter(|k| ***k == *key).count() as u32
        })
    }

    /// `Searcher._sweep`: play every root move once, best prior first, looking only for a
    /// result.  A move that ends the game enters the tree with its exact value (one simulation);
    /// any other child is dropped again.  Returns (moves checked, moves to check).
    fn sweep(
        &mut self,
        until: Option<Instant>,
        stop: &dyn Fn() -> bool,
        publish: &mut dyn FnMut(&Tree),
    ) -> Result<(usize, usize), i32> {
        let n = self.nodes[0].priors.len();
        let mut order: Vec<usize> = (0..n).collect();
        {
            let pr = &self.nodes[0].priors;
            order.sort_by(|&a, &b| pr[b].partial_cmp(&pr[a]).unwrap_or(Ordering::Equal));
        }
        let mut done = 0usize;
        for (k, &i) in order.iter().enumerate() {
            if stop() || self.nodes[0].win_idx.is_some() {
                break;
            }
            if let Some(u) = until {
                if Instant::now() > u {
                    break;
                }
            }
            let t0 = Instant::now();
            let pos = {
                let nodes = &self.nodes;
                let base = self.base;
                let cfg = self.cfg;
                nodes[0].pos.child(i, cfg, &mut self.buf, |key| base.get(key).copied().unwrap_or(0))?
            };
            self.t_step += t0.elapsed().as_secs_f64();
            done = k + 1;
            if !pos.done {
                continue;
            }
            let idx = self.nodes.len();
            self.nodes.push(Node::terminal(pos));
            self.nodes[0].children[i] = idx;
            let v = self.edge_value(0, idx, self.nodes[idx].value);
            let root = &mut self.nodes[0];
            root.w[i] += v;
            root.n[i] += 1;
            root.total += 1;
            if v >= 1.0 {
                root.win_idx = Some(i);
            }
            if done % 32 == 0 {
                publish(self);
            }
        }
        Ok((done, order.len()))
    }

    /// `Searcher._round`: up to `want` descents, one network call for the new leaves.
    fn round(&mut self, root: usize, want: usize) -> Result<(), i32> {
        let mut leaves: Vec<(Path, usize)> = Vec::new();
        let mut known: Vec<(Path, f64)> = Vec::new(); // descents that ended on a result
        let mut issued = 0usize;
        while issued < want && self.nodes[root].win_idx.is_none() {
            let mut path: Path = Vec::new();
            let mut node = root;
            let mut leaf: Option<usize> = None;
            let v: Option<f64>;
            loop {
                if self.nodes[node].terminal {
                    v = Some(self.nodes[node].value);
                    break;
                }
                if self.nodes[node].pending {
                    v = None; // another descent of this batch got here first
                    break;
                }
                let i = self.select(node);
                path.push((node, i));
                self.nodes[node].vl[i] += 1;
                self.nodes[node].nvl += 1;
                let child = self.nodes[node].children[i];
                if child == NONE {
                    let t0 = Instant::now();
                    let pos = self.make_child(&path, node, i)?;
                    self.t_step += t0.elapsed().as_secs_f64();
                    if pos.done {
                        let idx = self.nodes.len();
                        self.nodes.push(Node::terminal(pos));
                        self.nodes[node].children[i] = idx;
                        let value = self.nodes[idx].value;
                        if self.edge_value(node, idx, value) >= 1.0 {
                            self.nodes[node].win_idx = Some(i);
                        }
                        v = Some(value);
                    } else {
                        let idx = self.nodes.len();
                        self.nodes.push(Node::pending(pos));
                        self.nodes[node].children[i] = idx;
                        leaf = Some(idx);
                        v = Some(0.0);
                    }
                    break;
                }
                node = child;
            }
            match v {
                None => {
                    // collision: stop collecting, flush what we have
                    self.undo(&path);
                    break;
                }
                Some(v) => {
                    issued += 1;
                    match leaf {
                        Some(l) => leaves.push((path, l)),
                        // result known: back it up with the rest, so its virtual loss steers the
                        // other descents away too
                        None => known.push((path, v)),
                    }
                }
            }
        }
        if !leaves.is_empty() {
            let t0 = Instant::now();
            let ev = {
                let ps: Vec<&Pos> = leaves.iter().map(|&(_, l)| &self.nodes[l].pos).collect();
                evaluate(self.net, self.cfg, &ps)?
            };
            self.t_eval += t0.elapsed().as_secs_f64();
            self.evals += leaves.len() as u64;
            for (k, (path, l)) in leaves.iter().enumerate() {
                let value = ev.values[k] as f64;
                self.nodes[*l].fill(value, ev.probs[k].clone());
                self.backup(path, value);
            }
        }
        for (path, v) in known.iter() {
            self.backup(path, *v);
        }
        Ok(())
    }
}

/// What a search leaves behind (all `Searcher.search` + `search_action` read of the root).
pub(crate) struct SearchOut {
    pub visits: Vec<i64>,
    pub w: Vec<f64>,
    pub total: i64,
    pub win_idx: i64,
    pub t_eval: f64,
    pub t_step: f64,
    pub evals: u64,
    pub sweep_done: usize,
    pub sweep_total: usize,
}

/// How a search runs.
#[derive(Clone, Copy)]
pub(crate) struct Opts {
    pub sims: usize,
    pub batch: usize,
    pub c_puct: f64,
    pub vl: f64,
    /// the exact root sweep first (`sweep_k` None in Python)
    pub sweep: bool,
    /// stop when this has passed (checked between rounds)
    pub deadline: Option<Instant>,
    /// the sweep stops when this has passed
    pub sweep_until: Option<Instant>,
}

/// A snapshot of the root for whoever watches a running search.
pub(crate) struct Progress<'a> {
    pub visits: &'a [i64],
    pub w: &'a [f64],
    pub total: i64,
    pub win_idx: i64,
    pub sweep_done: usize,
    pub sweep_total: usize,
    pub t_eval: f64,
    pub t_step: f64,
    pub evals: u64,
}

fn snapshot<'a>(tree: &'a Tree, sweep: (usize, usize)) -> Progress<'a> {
    let root = &tree.nodes[0];
    Progress {
        visits: &root.n,
        w: &root.w,
        total: root.total,
        win_idx: root.win_idx.map(|i| i as i64).unwrap_or(-1),
        sweep_done: sweep.0,
        sweep_total: sweep.1,
        t_eval: tree.t_eval,
        t_step: tree.t_step,
        evals: tree.evals,
    }
}

/// Search from `root` for `sims` simulations (`batch` descents per network call).  The root's
/// value and (noised) priors are given; `base` is the table of positions seen so far, the root's
/// own included.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    net: &Net,
    cfg: &Cfg,
    root: Pos,
    base: &HashMap<Key, u32>,
    root_value: f64,
    root_priors: Vec<f64>,
    sims: usize,
    batch: usize,
    c_puct: f64,
    vl: f64,
) -> Result<SearchOut, i32> {
    let opts = Opts { sims, batch, c_puct, vl, sweep: false, deadline: None, sweep_until: None };
    search_live(net, cfg, root, base, root_value, root_priors, &opts, &|| false, &mut |_| {})
}

/// `search`, for play.py: optionally with the exact root sweep first, a deadline, a stop request
/// polled between rounds, and `publish` called with the root's numbers after every round (and
/// every few sweep moves) so a UI can show the search as it goes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_live(
    net: &Net,
    cfg: &Cfg,
    root: Pos,
    base: &HashMap<Key, u32>,
    root_value: f64,
    root_priors: Vec<f64>,
    opts: &Opts,
    stop: &dyn Fn() -> bool,
    publish: &mut dyn FnMut(&Progress),
) -> Result<SearchOut, i32> {
    if root.done || root_priors.len() != root.n_actions() {
        return Err(-4);
    }
    let sims = opts.sims;
    let mut tree = Tree {
        net,
        cfg,
        base,
        nodes: Vec::with_capacity(sims + 8),
        c_puct: opts.c_puct,
        vl: opts.vl,
        buf: Vec::with_capacity(HDR + 6 * 512),
        t_eval: 0.0,
        t_step: 0.0,
        evals: 0,
    };
    tree.nodes.push(Node::expandable(root, root_value, root_priors));
    let mut sweep = (0usize, 0usize);
    if opts.sweep {
        sweep = tree.sweep(opts.sweep_until, stop, &mut |t: &Tree| publish(&snapshot(t, (0, 0))))?;
        publish(&snapshot(&tree, sweep));
    }
    let batch = opts.batch.max(1);
    let mut idle = 0;
    while tree.nodes[0].win_idx.is_none() && (tree.nodes[0].total as usize) < sims {
        if stop() {
            break;
        }
        if let Some(d) = opts.deadline {
            if Instant::now() >= d {
                break;
            }
        }
        let before = tree.nodes[0].total;
        let want = batch.min(sims - tree.nodes[0].total as usize);
        tree.round(0, want)?;
        publish(&snapshot(&tree, sweep));
        // a round that adds nothing (it cannot, but a hang would be worse than a short search)
        if tree.nodes[0].total == before {
            idle += 1;
            if idle >= 8 {
                break;
            }
        } else {
            idle = 0;
        }
    }
    let root = &tree.nodes[0];
    Ok(SearchOut {
        visits: root.n.clone(),
        w: root.w.clone(),
        total: root.total,
        win_idx: root.win_idx.map(|i| i as i64).unwrap_or(-1),
        t_eval: tree.t_eval,
        t_step: tree.t_step,
        evals: tree.evals,
        sweep_done: sweep.0,
        sweep_total: sweep.1,
    })
}
