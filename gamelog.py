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


def notation_and_turns(rules: Rules, actions: List[Action]) -> Tuple[List[str], List[str]]:
    """(the notation of every action, the team that played it), replaying the line once."""
    game = Game(rules)
    texts, turns = [], []
    try:
        for action in actions:
            texts.append(move_text(game, action))
            turns.append(game.turn)
            game.step(action)
    finally:
        game.close()
    return texts, turns


# -- files -------------------------------------------------------------------

def games_dir(base: Optional[str] = None) -> str:
    return base or os.path.join(os.path.dirname(default_path()), "games")


def game_dict(rules: Rules, actions: List[Action], meta: Optional[Dict] = None) -> Dict:
    """The JSON-able record of a game (what save_game writes to a file)."""
    game = replay(rules, actions)
    if game.done:
        result = game.end_reason or ("draw" if game.winner is None else "%s wins" % full_name(game.winner))
    else:
        result = "unfinished"
    game.close()
    return {"format": FORMAT, "rules": dataclasses.asdict(rules),
            "actions": [action_to_dict(a) for a in actions],
            "moves": notation_list(rules, actions), "result": result,
            "meta": dict(meta or {}, saved=time.strftime("%Y-%m-%d %H:%M:%S"))}


def game_json(rules: Rules, actions: List[Action], meta: Optional[Dict] = None) -> str:
    return json.dumps(game_dict(rules, actions, meta), indent=1)


def move_list_text(turns: List[str], texts: List[str]) -> str:
    """One move per line: ``12 b Pe7-e6 +N`` (ply number, side, notation)."""
    if len(turns) != len(texts):
        raise ValueError("%d sides but %d moves" % (len(turns), len(texts)))
    return "\n".join("%d %s %s" % (i + 1, t, x) for i, (t, x) in enumerate(zip(turns, texts)))


def save_game(rules: Rules, actions: List[Action], meta: Optional[Dict] = None,
              path: Optional[str] = None, base: Optional[str] = None) -> str:
    """Write the game to ``path`` (default: a time-stamped file in games_dir()).  Returns the path."""
    data = game_dict(rules, actions, meta)
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


def parse_game(data) -> Tuple[Rules, List[Action], Dict]:
    """(rules, actions, meta) from the dict of a game record (``game_dict`` / a parsed file).
    The line is replayed once to make sure it is legal, so a damaged or hand-edited record
    fails here with a clear ValueError / IllegalActionError."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("not a Reverse Chess game record")
    try:
        rules = Rules(**data["rules"])
        actions = [action_from_dict(d) for d in data["actions"]]
    except (TypeError, KeyError, IndexError) as err:
        raise ValueError("damaged game record (%s: %s)" % (type(err).__name__, err))
    replay(rules, actions).close()
    meta = dict(data.get("meta") or {})
    meta["result"] = data.get("result", "")
    return rules, actions, meta


def loads_game(text: str) -> Tuple[Rules, List[Action], Dict]:
    """``parse_game`` of JSON text (what Ctrl+Shift+C copies)."""
    try:
        data = json.loads(text)
    except ValueError as err:
        raise ValueError("not JSON (%s)" % err)
    return parse_game(data)


def load_game(path: str) -> Tuple[Rules, List[Action], Dict]:
    """(rules, actions, meta) from a file written by save_game."""
    with open(path) as fh:
        try:
            data = json.load(fh)
        except ValueError as err:
            raise ValueError("%s is not a Reverse Chess game file (%s)" % (os.path.basename(path), err))
    try:
        return parse_game(data)
    except ValueError as err:
        raise ValueError("%s: %s" % (os.path.basename(path), err))


def parse_move_list(rules: Rules, text: str) -> List[Action]:
    """Actions from a move list as Ctrl+C copies it (``12 b Pe7-e6 +N`` per line; the number and
    side are optional).  Every line must be a legal move of the position it is played in;
    raises ValueError naming the first line that is not."""
    game = Game(rules)
    actions: List[Action] = []
    try:
        for n, raw in enumerate(text.splitlines(), 1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) >= 3 and parts[0].isdigit() and parts[1] in ("w", "b"):
                parts = parts[2:]
            wanted = " ".join(parts)
            table: Dict[str, Action] = {}
            for action in game.legal_actions():
                table.setdefault(move_text(game, action), action)
            if wanted not in table:
                raise ValueError("line %d: '%s' is not a legal move there" % (n, wanted))
            actions.append(table[wanted])
            game.step(table[wanted])
            if game.done:
                break
    finally:
        game.close()
    if not actions:
        raise ValueError("no moves found")
    return actions


def saved_games(base: Optional[str] = None) -> List[str]:
    """Saved game files, newest first."""
    folder = games_dir(base)
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    except OSError:
        return []
    return sorted(files, key=lambda p: (os.path.getmtime(p), p), reverse=True)


# -- the session autosave (resume after a crash or a closed window) ---------------

SESSION_FILE = "autosave.json"


def autosave_path(base: Optional[str] = None) -> str:
    """The file play.py keeps the game in progress in: next to stats.json, not in games/ (so it
    never shows up in the Load list)."""
    return os.path.join(base or os.path.dirname(default_path()), SESSION_FILE)


def write_autosave(rules: Rules, actions: List[Action], texts: List[str], variations=(),
                   meta: Optional[Dict] = None, path: Optional[str] = None) -> str:
    """Write the game in progress (and the side lines kept by play.py's variation stack) without
    replaying anything, so it is cheap enough for after every move.  The file is written to a
    temporary name and renamed, so a crash half way never leaves a broken autosave."""
    path = path or autosave_path()
    data = {"format": FORMAT, "rules": dataclasses.asdict(rules),
            "actions": [action_to_dict(a) for a in actions], "moves": list(texts),
            "result": "autosave", "meta": dict(meta or {}, saved=time.strftime("%Y-%m-%d %H:%M:%S")),
            "variations": [{"branch": int(v["branch"]),
                            "actions": [action_to_dict(a) for a in v["line"]]} for v in variations]}
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)
    return path


def read_autosave(path: Optional[str] = None):
    """(rules, actions, meta, variations) of the autosave, or None when there is none.
    ``variations`` is a list of {"line": [Action], "branch": int}; one that no longer replays is
    dropped.  A damaged main line raises ValueError / IllegalActionError."""
    path = path or autosave_path()
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except ValueError as err:
        raise ValueError("autosave is damaged (%s)" % err)
    rules, actions, meta = parse_game(data)
    variations = []
    for v in data.get("variations") or []:
        try:
            line = [action_from_dict(d) for d in v["actions"]]
            replay(rules, line).close()
            variations.append({"line": line, "branch": max(0, min(int(v["branch"]), len(line)))})
        except Exception:                      # a side line is a convenience, never worth failing for
            continue
    return rules, actions, meta, variations


def clear_autosave(path: Optional[str] = None) -> None:
    try:
        os.remove(path or autosave_path())
    except OSError:
        pass
