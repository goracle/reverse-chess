"""Game records for Reverse Chess: notation, save / load, replay.  No pygame in here.

A game is fully determined by its Rules and the list of Actions played (the engine has no
randomness), so a saved game is just that:

    {"format": 1, "rules": {...}, "actions": [{"src": [r, c], "dst": [r, c], ...}, ...],
     "moves": ["Ke7-e8 +Q", ...],            # readable notation, for people (ignored on load)
     "result": "black wins: ...", "meta": {...}}

Files go to the user's data folder next to stats.json (see stats.py):

    Windows  %APPDATA%\\ReverseChess\\games\\
    macOS    ~/Library/Application Support/ReverseChess/games/
    Linux    $XDG_DATA_HOME or ~/.local/share/ReverseChess/games/

Notation (``move_text``): piece letter (none for a pawn, ``*`` after it for a promoted piece),
source-destination, then ``u`` for an un-promotion, `` +Q`` for a piece given back (``+P=Q`` a
pawn given back and promoted to a queen, ``+P ep`` reverse en passant).  Squares are as drawn:
a1 bottom-left, white at the bottom.
"""
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from engine import SIZE, Action, Game, Rules, full_name
from stats import default_path

FORMAT = 1
_LETTER = {"k": "K", "q": "Q", "r": "R", "b": "B", "n": "N", "p": ""}


def sq_name(sq) -> str:
    """(row, col) -> 'a1'..'h8' (white at the bottom, as drawn)."""
    return "%s%d" % ("abcdefgh"[sq[1]], SIZE - sq[0])


def move_text(game: Game, action: Action) -> str:
    """Short notation of ``action``, to be called on the position *before* it is played."""
    piece = game.board[action.src]
    letter = _LETTER[piece.ptype] if piece is not None else "?"
    if piece is not None and piece.promoted:
        letter += "*"
    text = "%s%s-%s" % (letter, sq_name(action.src), sq_name(action.dst))
    if action.demote:
        text += "u"
    if action.gift:
        text += " +" + (_LETTER[action.gift] or "P")
        if action.promote:
            text += "=" + _LETTER[action.promote]
        if action.en_passant:
            text += " ep"
    return text


# -- actions <-> json -----------------------------------------------------

def action_to_dict(a: Action) -> Dict:
    d = {"src": list(a.src), "dst": list(a.dst)}
    if a.demote:
        d["demote"] = True
    if a.gift is not None:
        d["gift"] = a.gift
    if a.en_passant:
        d["en_passant"] = True
    if a.promote is not None:
        d["promote"] = a.promote
    return d


def action_from_dict(d: Dict) -> Action:
    return Action(tuple(d["src"]), tuple(d["dst"]), bool(d.get("demote", False)),
                  d.get("gift"), bool(d.get("en_passant", False)), d.get("promote"))


# -- replay ----------------------------------------------------------------

def replay(rules: Rules, actions: List[Action], upto: Optional[int] = None) -> Game:
    """A fresh Game with the first ``upto`` actions played (all of them by default).
    Raises engine.IllegalActionError if the list is not a legal game under ``rules``."""
    game = Game(rules)
    for action in (actions if upto is None else actions[:upto]):
        game.step(action)
    return game


def notation_list(rules: Rules, actions: List[Action]) -> List[str]:
    """The notation of every action in a line (replays the game once)."""
    game = Game(rules)
    out = []
    for action in actions:
        out.append(move_text(game, action))
        game.step(action)
    game.close()
    return out


# -- files -------------------------------------------------------------------

def games_dir(base: Optional[str] = None) -> str:
    return base or os.path.join(os.path.dirname(default_path()), "games")


def save_game(rules: Rules, actions: List[Action], meta: Optional[Dict] = None,
              path: Optional[str] = None, base: Optional[str] = None) -> str:
    """Write the game to ``path`` (default: a time-stamped file in games_dir()).  Returns the path."""
    game = replay(rules, actions)
    if game.done:
        result = game.end_reason or ("draw" if game.winner is None else "%s wins" % full_name(game.winner))
    else:
        result = "unfinished"
    data = {"format": FORMAT, "rules": dataclasses.asdict(rules),
            "actions": [action_to_dict(a) for a in actions],
            "moves": notation_list(rules, actions), "result": result,
            "meta": dict(meta or {}, saved=time.strftime("%Y-%m-%d %H:%M:%S"))}
    game.close()
    if path is None:
        folder = games_dir(base)
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(folder, "game-%s.json" % stamp)
        n = 1
        while os.path.exists(path):          # two saves in the same second
            n += 1
            path = os.path.join(folder, "game-%s-%d.json" % (stamp, n))
    elif os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return path


def load_game(path: str) -> Tuple[Rules, List[Action], Dict]:
    """(rules, actions, meta) from a file written by save_game.  The line is replayed once to
    make sure it is legal, so a damaged or hand-edited file fails here with a clear error."""
    with open(path) as fh:
        data = json.load(fh)
    if data.get("format") != FORMAT:
        raise ValueError("%s is not a Reverse Chess game file" % os.path.basename(path))
    rules = Rules(**data["rules"])
    actions = [action_from_dict(d) for d in data["actions"]]
    replay(rules, actions).close()
    meta = dict(data.get("meta") or {})
    meta["result"] = data.get("result", "")
    return rules, actions, meta


def saved_games(base: Optional[str] = None) -> List[str]:
    """Saved game files, newest first."""
    folder = games_dir(base)
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    except OSError:
        return []
    return sorted(files, key=lambda p: (os.path.getmtime(p), p), reverse=True)
