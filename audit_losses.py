#!/usr/bin/env python
"""Audit the engine's "no way for all of X's pawns/bishops to get home" losses.  (Run from reverse-chess/.)

    python audit_losses.py heur:gift random                 # who dooms whom, and why
    python audit_losses.py runs/run2/latest.pt heur:nogift  # the net (needs torch) vs a bot
    python audit_losses.py heur:gift heur:nogift --games 400 --probe 20 --playouts 150

Players are "random", "heur:nogift", "heur:gift", "heur:rush" or a checkpoint path.  Games are
played with both colours; every game that ends in a dead-end detection
(pawn_scan / bishop_wall_check / pawn wall-off) is replayed to the position before
the last move and classified:

  victim moved last, no safe alternative   forced: the winner squeezed it (zugzwang)
  victim moved last, safe moves existed    blunder: the victim doomed itself
  winner moved last                        SUSPICIOUS: the winner's own move fired the
                                           detector.  _gift_problem already forbids
                                           gifts that do this, so every one of these
                                           is worth a look; the boards are printed.

Model checks over all plies played (counterexamples would break the detectors'
assumptions):
  * every sideways pawn step is a forced gift (pawn_scan charges one gift per step)
  * how often a bishop is gifted straight onto a home square (the one way the bishop
    wall can be bypassed: it needs an opposing piece standing on that square)

--probe N --playouts K: for N flagged final positions, switch the detectors off and
try K playouts of up to 60 plies each (the "doomed" side walks home, its opponent
plays randomly) looking for any line where the doomed side does get every piece
home.  One hit disproves the detector at that position; no hits proves nothing.
"""
import argparse
import collections
import random

import numpy as np

import engine
from engine import Game, Rules, other_team, back_row, full_name
import heuristic


def make_player(spec):
    """Returns f(game, py_rng, np_rng) -> Action."""
    if spec == "random":
        return lambda g, pr, nr: pr.choice(g.legal_actions())
    if spec.startswith("heur:"):
        mode = spec.split(":", 1)[1]
        return lambda g, pr, nr: heuristic.choose(g, pr, mode)
    import torch
    torch.set_num_threads(2)
    import net as netlib
    model, _ = netlib.load_checkpoint(spec)
    return lambda g, pr, nr: netlib.choose_action(model, g, nr)[0]


def detector_kind(reason):
    for k in ("pawn wall-off", "pawns", "bishops"):
        if k in reason and ("wins:" in reason or "prevents" in reason):
            return {"pawn wall-off": "wall-off", "pawns": "pawns", "bishops": "bishops"}[k]
    return None


def dooms(game, action, team):
    """Would ``team`` be flagged as dead-ended right after ``action``?"""
    move = game.actions[action]
    if move is None:
        move = game._resolve(action)
    game._execute(move)
    try:
        return game._losing_reason(team) is not None
    finally:
        game._undo(move)


def play_game(rules, specs, players, a_team, seed, tally):
    pr, nr = random.Random(seed), np.random.default_rng(seed)
    g = Game(rules)
    who = {a_team: 0, other_team(a_team): 1}
    while not g.done:
        # model checks on the legal actions of every position
        for a in g.legal_actions():
            mover = g.board[a.src[0], a.src[1]]
            if mover.ptype == "p" and not mover.promoted and not a.demote and a.dst[1] != a.src[1]:
                tally["sideways_pawn_steps"] += 1
                if a.gift is None:
                    tally["sideways_pawn_steps_without_gift"] += 1
            if a.gift == "b":
                tally["bishop_gifts"] += 1
                recv = other_team(mover.team)
                if a.src[0] == back_row(recv) and a.src[1] in (2, 5):
                    tally["bishop_gifted_onto_home_square"] += 1
        tally["plies"] += 1
        g.step(players[who[g.turn]](g, pr, nr))
    return g


def classify(rules, g):
    """-> (kind, category, info) for a game that ended in a detector loss, else None."""
    kind = detector_kind(g.end_reason)
    if kind is None or g.winner is None:
        return None
    victim = other_team(g.winner)
    log = list(g.move_log)
    h = Game(rules)
    for a in log[:-1]:
        h.step(a)
    mover = h.turn
    last = log[-1]
    info = {"ascii": h.ascii(), "action": last, "mover": mover, "victim": victim,
            "reason": g.end_reason, "ply": len(log)}
    if mover == victim:
        legal = h.legal_actions()
        safe = [a for a in legal if not dooms(h, a, victim)]
        info["safe_alternatives"] = len(safe)
        info["legal"] = len(legal)
        info["gifts"] = sum(a.gift is not None for a in legal)
        cat = "forced" if not safe else "blunder"
    else:
        cat = "SUSPICIOUS"
    h.close()
    return kind, cat, info, log


def probe(rules, log, victim, playouts, rng):
    """Detectors off; the victim walks home, the other side plays randomly."""
    saved = Game._losing_reason
    Game._losing_reason = lambda self, team: None
    try:
        for k in range(playouts):
            g = Game(rules)
            for a in log:
                g.step(a)
            seq = []
            for _ in range(60):
                if g.done:
                    break
                if g.turn == victim:
                    a = heuristic.choose(g, rng, "nogift", safe=False)
                else:
                    a = rng.choice(g.legal_actions())
                seq.append(a)
                g.step(a)
                if g.check_for_win(victim):
                    return seq
            g.close()
    finally:
        Game._losing_reason = saved
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=3, help="boards to print for SUSPICIOUS cases")
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--playouts", type=int, default=100)
    args = ap.parse_args()

    rules = Rules(max_plies=args.max_plies)
    players = [make_player(args.a), make_player(args.b)]
    tally = collections.Counter()
    cats = collections.Counter()
    flagged, suspicious, squeezed = [], [], []
    results = collections.Counter()
    for i in range(args.games):
        a_team = "b" if i % 2 == 0 else "w"
        g = play_game(rules, (args.a, args.b), players, a_team, args.seed + i, tally)
        results["A wins" if g.winner == a_team else ("draw" if g.winner is None else "B wins")] += 1
        c = classify(rules, g)
        if c is not None:
            kind, cat, info, log = c
            who = "A" if g.winner == a_team else "B"
            cats[(kind, cat, "winner " + who)] += 1
            flagged.append((log, info["victim"]))
            if cat == "forced":
                squeezed.append((kind, info["legal"], info["gifts"]))
            if cat == "SUSPICIOUS":
                suspicious.append(info)
        g.close()

    print("A = %s   B = %s   %d games: %s" % (args.a, args.b, args.games, dict(results)))
    n = sum(cats.values())
    print("\ndetector losses: %d of %d games" % (n, args.games))
    for (kind, cat, who), v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print("  %4d  %-9s %-10s %s" % (v, kind, cat, who))
    if squeezed:
        legal = np.array([x[1] for x in squeezed])
        print("\nforced cases: the victim had %d legal moves at the end (median; min %d, max %d)" % (
            np.median(legal), legal.min(), legal.max()))
        print("  %d of %d had at most 3 legal moves; %d of %d had at most 10" % (
            (legal <= 3).sum(), len(legal), (legal <= 10).sum(), len(legal)))
        print("  median gifting moves among them: %d" % np.median([x[2] for x in squeezed]))
    print("\nmodel checks over %d plies" % tally["plies"])
    print("  sideways pawn steps %d, of which WITHOUT a gift: %d   (must be 0)" % (
        tally["sideways_pawn_steps"], tally["sideways_pawn_steps_without_gift"]))
    print("  bishop gifts %d, of which straight onto a home square: %d" % (
        tally["bishop_gifts"], tally["bishop_gifted_onto_home_square"]))
    for info in suspicious[:args.show]:
        print("\nSUSPICIOUS: %s moved last (ply %d) and %s was flagged: %s" % (
            full_name(info["mover"]), info["ply"], full_name(info["victim"]), info["reason"]))
        print("position before the move:\n" + info["ascii"])
        print("action:", info["action"])

    if args.probe and flagged:
        rng = random.Random(args.seed)
        picks = rng.sample(flagged, min(args.probe, len(flagged)))
        hits = 0
        for log, victim in picks:
            seq = probe(rules, log, victim, args.playouts, rng)
            if seq:
                hits += 1
                print("\nPROBE HIT: %s got every piece home after the detector fired; moves:" % full_name(victim))
                print("  ", seq[:12], "..." if len(seq) > 12 else "")
        print("\nprobe: %d of %d flagged positions refuted (%d playouts each)" % (hits, len(picks), args.playouts))


if __name__ == "__main__":
    main()
