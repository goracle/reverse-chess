"""Win/loss/draw record for games against the computer (a small JSON file).

One record per finished game: when, which colour the human played, the result *for the human*
('won' / 'lost' / 'drawn', or 'abandoned' if a new game was started first), how many plies,
whether hints were used and which version of the AI it was.  The file lives in the user's
data folder, not next to the program, so it survives updates and works from a packaged app:

    Windows  %APPDATA%\\ReverseChess\\stats.json
    macOS    ~/Library/Application Support/ReverseChess/stats.json
    Linux    $XDG_DATA_HOME or ~/.local/share/ReverseChess/stats.json
"""
import json
import os
import sys
import time
from typing import Dict, List, Optional

RESULTS = ("won", "lost", "drawn")


def default_path() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "ReverseChess", "stats.json")


class Stats:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_path()
        self.games: List[Dict] = []
        self.error = ""                       # shown by the game if the file could not be used
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.games = [g for g in data["games"] if g.get("result") in RESULTS + ("abandoned",)]
        except FileNotFoundError:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            # unreadable: keep it aside rather than overwrite it, and start a new record
            try:
                os.replace(self.path, self.path + ".bad")
                self.error = "stats file was unreadable; kept as stats.json.bad"
            except OSError:
                self.error = "stats file is unreadable and could not be moved aside"

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"games": self.games}, f, indent=1)
            os.replace(tmp, self.path)
            self.error = ""
        except OSError as err:
            self.error = "could not save stats: %s" % err.strerror

    def add(self, human: str, result: str, plies: int, hints: bool, ai_version: Optional[int]):
        """human: 'b' or 'w';  result: 'won' | 'lost' | 'drawn' | 'abandoned'."""
        assert human in ("b", "w") and result in RESULTS + ("abandoned",), (human, result)
        self.games.append({"t": int(time.time()), "human": human, "result": result,
                           "plies": int(plies), "hints": bool(hints), "ai": ai_version})
        self._save()

    def summary(self) -> Dict:
        """Counts overall and per colour, plus how many games used hints / were abandoned."""
        out = {"all": dict.fromkeys(RESULTS, 0), "b": dict.fromkeys(RESULTS, 0),
               "w": dict.fromkeys(RESULTS, 0), "abandoned": 0, "hinted": 0}
        for g in self.games:
            if g["result"] == "abandoned":
                out["abandoned"] += 1
                continue
            out["all"][g["result"]] += 1
            out[g["human"]][g["result"]] += 1
            out["hinted"] += bool(g.get("hints"))
        return out

    def reset(self):
        self.games = []
        self._save()
