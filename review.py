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

It is the bare network (one forward pass per position, so a whole game takes a second or two),
not a search: treat the marks as "look here", not as proof.  play.py shows the same review
in its move list (the Review button, V; it runs by itself when a game ends) using ``Reviewer`` below.
"""
import argparse
import os
import sys
import threading
from typing import Dict, List, Optional

import gamelog
import numpy_net
from engine import Action, Game, Rules

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


def review_line(advisor, rules: Rules, actions: List[Action], stop=None, progress=None) -> Optional[Dict]:
    """Review a whole line.  Returns None if ``stop()`` became true half way.

    result["values"][i]  white's value in the position before move i (the last entry: the end)
    result["best"][i]    the network's favourite move there
    result["loss"][i]    how much the mover's value dropped across move i
    result["marks"][i]   "", "?!", "?" or "??"
    """
    game = Game(rules)
    values, best, turns = [], [], []
    try:
        for i, action in enumerate(actions):
            if stop is not None and stop():
                return None
            an = advisor.analyse_fresh(game)
            values.append(an.value if game.turn == "w" else -an.value)
            best.append(an.top(1)[0][0])
            turns.append(game.turn)
            game.step(action)
            if progress is not None:
                progress(i + 1, len(actions))
        if game.done:
            values.append(0.0 if game.winner is None else 1.0 if game.winner == "w" else -1.0)
        else:
            values.append(advisor.white_value(game, advisor.analyse_fresh(game)))
    finally:
        game.close()
    loss, marks = [], []
    for i, action in enumerate(actions):
        drop = values[i] - values[i + 1] if turns[i] == "w" else values[i + 1] - values[i]
        loss.append(drop)
        marks.append("" if action == best[i] else classify(drop))
    return {"values": values, "best": best, "loss": loss, "marks": marks}


class Reviewer:
    """review_line in a background thread (so the window stays alive).  Poll ``.done``."""

    def __init__(self, advisor, rules: Rules, actions: List[Action]):
        self.actions = list(actions)
        self.result = None
        self.error = None
        self.done = False
        self.progress = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(advisor, rules), daemon=True)
        self._thread.start()

    def _run(self, advisor, rules):
        try:
            self.result = review_line(advisor, rules, self.actions, stop=lambda: self._stop,
                                      progress=lambda i, n: setattr(self, "progress", i / max(n, 1)))
        except Exception as err:               # shown in the window instead of killing the thread silently
            self.error = "%s: %s" % (type(err).__name__, err)
        self.done = True

    def stop(self):
        self._stop = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", help="a saved game (default: the newest one)")
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
    out = review_line(advisor, rules, actions)
    texts = gamelog.notation_list(rules, actions)
    game = Game(rules)
    print("%s   %d plies   %s   (network %s)" % (os.path.basename(path), len(actions),
                                                   meta.get("result", ""), advisor.name))
    flagged = 0
    turns = []
    for i, action in enumerate(actions):
        team = game.turn
        turns.append(team)
        mark = out["marks"][i]
        line = "%3d %s  %-26s %+.2f" % (i + 1, team, texts[i] + mark, out["values"][i + 1])
        if mark:
            flagged += 1
            line += "   lost %.2f; the network preferred %s" % (out["loss"][i], gamelog.move_text(game, out["best"][i]))
        print(line)
        game.step(action)
    print("%d move(s) marked" % flagged)
    summary = summarize(turns, out["loss"], out["marks"])
    for team in ("w", "b"):
        print(summary_text(team, summary[team]))


if __name__ == "__main__":
    main()
