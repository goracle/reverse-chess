#!/usr/bin/env python
"""Automatic game review: find the moves that threw the game away.  No puzzles to author --
the "puzzles" are your own (and the AI's) mistakes, with the move the network preferred.

    python review.py                       # review the newest saved game
    python review.py game.json             # review a particular file
    python review.py game.json --ai ai.npz

For every position of the game the network gives a value (-1 .. +1, from the side to move) and
a favourite move.  A move is marked when the mover's value dropped a lot across it and it was
not the network's own favourite:

    ?!  inaccuracy  (lost 0.15 .. 0.30)       ?  mistake (0.30 .. 0.50)       ??  blunder (0.50+)

With ``--sims N`` (default 30) every position is also searched (MCTS, N simulations, see mcts.py)
and the values and the "best move" come from the search, so a move is only marked when the
*search* sees the value drop, not just the network's raw, jumpy estimate.  ``--sims 0`` is the
bare network (one forward pass per position, a second or two for a whole game).  Either way, treat
the marks as "look here", not as proof.  For every marked move the review also keeps the top three
moves of that position with their value.  play.py shows the same review in its move list (the
Review button, V; it runs by itself when a game ends) using ``Reviewer`` below.
"""
import argparse
import os
import sys
import threading
from typing import Dict, List, Optional

import gamelog
import mcts
import numpy_net
from engine import Action, Game, Rules

DEFAULT_SIMS = 30        # search simulations per position (0: the bare network)
SEARCH_MS = 2000         # ... and at most this long per position

MISTAKES = ((0.50, "??", "blunder"), (0.30, "?", "mistake"), (0.15, "?!", "inaccuracy"))


def classify(loss: float) -> str:
    for limit, mark, _ in MISTAKES:
        if loss >= limit:
            return mark
    return ""


def summarize(turns: List[str], loss: List[float], marks: List[str]) -> Dict[str, Dict]:
    """Per side: moves reviewed, average loss per move (a move that gained value counts as 0)
    and how many of each mark.  ``turns`` may be longer than ``loss`` (a partly reviewed line)."""
    out = {}
    for team in ("w", "b"):
        idx = [i for i in range(min(len(turns), len(loss))) if turns[i] == team]
        out[team] = {
            "moves": len(idx),
            "avg_loss": sum(max(loss[i], 0.0) for i in idx) / len(idx) if idx else 0.0,
            "counts": {mark: sum(1 for i in idx if marks[i] == mark) for _, mark, _ in MISTAKES},
        }
    return out


def summary_text(team: str, info: Dict) -> str:
    """'White 0.08 avg: 2 ?! 1 ? 0 ??' (average loss per move, then the mark counts)"""
    c = info["counts"]
    return "%s %.2f avg: %d ?! %d ? %d ??" % ("White" if team == "w" else "Black", info["avg_loss"],
                                             c["?!"], c["?"], c["??"])


def after_value(advisor, game: Game, action: Action) -> float:
    """The mover's value (-1 .. +1) after ``action``: the network's, or exact if the move ends the game."""
    child = mcts.fast_clone(game)
    try:
        child.step(action)
        if child.done:
            return 0.0 if child.winner is None else 1.0 if child.winner == game.turn else -1.0
        v = advisor.analyse_fresh(child).value
        return v if child.turn == game.turn else -v        # (a reverse en passant keeps the same mover)
    finally:
        child.close()


def review_line(advisor, rules: Rules, actions: List[Action], stop=None, progress=None,
                sims: int = 0, searcher=None) -> Optional[Dict]:
    """Review a whole line.  Returns None if ``stop()`` became true half way.

    ``sims`` > 0 searches every position (see the module doc); 0 uses the bare network.

    result["values"][i]  white's value in the position before move i (the last entry: the end)
    result["best"][i]    the favourite move there (the search's most visited, or the network's)
    result["loss"][i]    how much the mover's value dropped across move i
    result["marks"][i]   "", "?!", "?" or "??"
    result["top"][i]     up to three (action, share, value) there: the share of the search's visits
                        (or the network's probability) and the value for the mover after it, which
                        is None for an unmarked position without search
    result["sims"]       the simulations per position used (0: network only)
    """
    game = Game(rules)
    if sims > 0 and searcher is None:
        searcher = mcts.Searcher(advisor, sims=sims, think_ms=SEARCH_MS)
    searching = searcher is not None and sims > 0

    def looked_at(position, an):
        """(value for the side to move, favourite move, top three) of ``position``."""
        if not searching:
            return an.value, an.top(1)[0][0], [(a, p, None) for a, p in an.top(3)]
        job = searcher.search(position, sims=sims, ms=SEARCH_MS, root_analysis=an)
        if job.error is not None:
            raise RuntimeError(job.error)
        v = job.value()
        best = job.best()
        return (an.value if v is None else v, an.top(1)[0][0] if best is None else best,
                job.top(3) or [(a, p, None) for a, p in an.top(3)])

    values, best, turns, tops = [], [], [], []
    try:
        for i, action in enumerate(actions):
            if stop is not None and stop():
                return None
            turns.append(game.turn)
            v, b, top = looked_at(game, advisor.analyse_fresh(game))
            values.append(v if game.turn == "w" else -v)
            best.append(b)
            tops.append(top)
            game.step(action)
            if progress is not None:
                progress(i + 1, len(actions))
        if game.done:
            values.append(0.0 if game.winner is None else 1.0 if game.winner == "w" else -1.0)
        else:
            v = looked_at(game, advisor.analyse_fresh(game))[0]
            values.append(v if game.turn == "w" else -v)
    finally:
        game.close()
    loss, marks = [], []
    for i, action in enumerate(actions):
        drop = values[i] - values[i + 1] if turns[i] == "w" else values[i + 1] - values[i]
        loss.append(drop)
        marks.append("" if action == best[i] else classify(drop))
    if not searching and any(marks):          # the network's top three, each valued, where it matters
        game = Game(rules)
        try:
            for i, action in enumerate(actions):
                if marks[i]:
                    if stop is not None and stop():
                        return None
                    an = advisor.analyse_fresh(game)
                    tops[i] = [(a, p, after_value(advisor, game, a)) for a, p in an.top(3)]
                game.step(action)
        finally:
            game.close()
    return {"values": values, "best": best, "loss": loss, "marks": marks, "top": tops,
            "sims": sims if searching else 0}


class Reviewer:
    """review_line in a background thread (so the window stays alive).  Poll ``.done``."""

    def __init__(self, advisor, rules: Rules, actions: List[Action], sims: int = 0):
        self.actions = list(actions)
        self.sims = sims
        self.result = None
        self.error = None
        self.done = False
        self.progress = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(advisor, rules), daemon=True)
        self._thread.start()

    def _run(self, advisor, rules):
        try:
            self.result = review_line(advisor, rules, self.actions, stop=lambda: self._stop, sims=self.sims,
                                      progress=lambda i, n: setattr(self, "progress", i / max(n, 1)))
        except Exception as err:               # shown in the window instead of killing the thread silently
            self.error = "%s: %s" % (type(err).__name__, err)
        self.done = True

    def stop(self):
        self._stop = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", help="a saved game (default: the newest one)")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                    help="search simulations per position (default %d; 0 = the bare network)" % DEFAULT_SIMS)
    ap.add_argument("--ai", default=None, help="network file (default: ai.npz / latest.pt next to the program)")
    args = ap.parse_args()
    path = args.game
    if path is None:
        saved = gamelog.saved_games()
        if not saved:
            raise SystemExit("no saved games in %s (press Ctrl+S in play.py to save one)" % gamelog.games_dir())
        path = saved[0]
    ai = args.ai or numpy_net.find_ai_file(os.path.dirname(os.path.abspath(__file__)), os.getcwd())
    if not ai:
        raise SystemExit("no network file found (ai.npz); use --ai")
    rules, actions, meta = gamelog.load_game(path)
    advisor = numpy_net.Advisor(ai)
    out = review_line(advisor, rules, actions, sims=args.sims)
    texts = gamelog.notation_list(rules, actions)
    game = Game(rules)
    print("%s   %d plies   %s   (network %s, %s)" % (
        os.path.basename(path), len(actions), meta.get("result", ""), advisor.name,
        "%d sims per position" % args.sims if args.sims > 0 else "no search"))
    flagged = 0
    turns = []
    for i, action in enumerate(actions):
        team = game.turn
        turns.append(team)
        mark = out["marks"][i]
        line = "%3d %s  %-26s %+.2f" % (i + 1, team, texts[i] + mark, out["values"][i + 1])
        if mark:
            flagged += 1
            line += "   lost %.2f; preferred %s" % (out["loss"][i], gamelog.move_text(game, out["best"][i]))
            line += "   [" + ", ".join("%s %d%% %s" % (gamelog.move_text(game, a), round(100 * share),
                                                       "" if v is None else "%+.2f" % v)
                                        for a, share, v in out["top"][i]) + "]"
        print(line)
        game.step(action)
    print("%d move(s) marked" % flagged)
    summary = summarize(turns, out["loss"], out["marks"])
    for team in ("w", "b"):
        print(summary_text(team, summary[team]))


if __name__ == "__main__":
    main()
