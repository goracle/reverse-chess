#!/usr/bin/env python
"""Pygame front-end for Reverse Chess, built on the headless engine.

    python play.py                          # you (black) vs the AI (white)
    python play.py --ai-color b             # you play white
    python play.py --no-ai                  # hot-seat, two humans (the AI can be switched on in the window)
    python play.py --ai path/to/ai.npz      # use a particular network file
    python play.py --black random --white random --delay 100   # watch a game
    python play.py --white mymodule:policy  # plug in your own agent (see below)
    python play.py --white ckpt:runs/run1/latest.pt   # play a network from train.py (needs PyTorch)

How to play
-----------
* Pieces with a legal move have a blue outline.  Click one: its legal
  destinations get a dot.
* Click any other square to be told why that move is illegal.
* Click a destination.  If the move needs a decision (un-promote? which piece
  to give back? reverse en passant? promote the gifted pawn to what?) a menu
  appears in the window.  *Back* returns to the previous question, the X (or
  Esc) closes the menu and keeps your piece selected; nothing is played until
  the last question is answered.
* Everything else lives in the menu bar along the top (File, Game, Analysis, View, Help); the window is
  just the board and one slim side panel, and grows a move-list column only in review mode (or when
  View > Move list > Always show says so).
* The panel on the right shows which pieces are still off the board for each
  team (those are the pieces the *other* side can give back) and how many of
  a team's on-board pieces are home.
* Padlocks: grey = locked for good (home pawns, castled king/rook); orange =
  blocked in right now (e.g. a bishop behind its own pawns).

The AI
------
The computer opponent is the network trained by train.py.  Put its file next to play.py
(``ai.npz``, or a checkpoint such as ``latest.pt``; the newest of ``ai.npz`` / ``ai.pt`` /
``latest.pt`` is used), or point to one with ``--ai``.  A ``.pt`` checkpoint needs PyTorch;
``python export_weights.py latest.pt`` turns it into an ``ai.npz`` that needs only NumPy.
With no file the computer moves at random and the eval bar and hints are unavailable.

* Look-ahead search (S, the MCTS button; see mcts.py): the computer does not just play the
  network's first instinct.  It runs a Monte-Carlo tree search in a background thread (up to
  --sims simulations / --think-ms milliseconds per move), first checking every legal move
  exactly for an immediate win or loss, then averaging the network's value over many lines.
  ``--no-search`` starts with it off (the old policy-sampling opponent).
* Eval bar (E): who is ahead, white at the bottom.  While it is your turn the search keeps
  thinking and the bar becomes the search's value ("search N" in the panel; "forced win"
  when a winning move was found); before that it is the network's raw estimate ("net").
* Hint (H): the three favourite moves, as arrows and in the panel (the network's policy at
  first, then the search's most-visited moves as it deepens).
* AI on/off (A) and which colour(s) the AI plays: Game > Computer plays.
* Your record against the AI (won / lost / drawn, by colour) is kept in a small file in your
  user data folder (see stats.py); a game counts only if the opponent was the AI from the first
  move to the last.  Games where you used hints are counted too, and noted.
* Keys: R = new game, Esc = back out of the menu / deselect, H, E, A as above.

Game history (the third column)
-------------------------------
* The move list shows every move in short notation (K = king, ``*`` = promoted piece, ``+Q`` = a
  queen given back, ``u`` = un-promotion).  Click a move to see the position after it (the top row, "start position", the position before the first move); the mouse
  wheel scrolls.  Left / Right step through the game, Home / End jump to the start / the end.
* Undo (Z or Backspace) takes back your last move (and the computer's reply).  Nothing is thrown
  away until you play a different move: Right / End bring the old moves back.  A game in which you
  took a move back is not counted in your record.
* Flip (F) turns the board round; ``--flip`` starts that way.
* Save (Ctrl+S) writes the game to your user data folder (see gamelog.py); Load (Ctrl+O) opens the
  newest saved game, pressing it again the one before.  ``--load FILE`` opens a particular file,
  ``--autosave`` saves every finished game by itself.  A loaded game that is not over can be
  played on.
* Review (V): the network rates every position and marks the moves that threw the game away
  (``?!`` inaccuracy, ``?`` mistake, ``??`` blunder).  It runs by itself when a game ends or is
  loaded.  M / the Next ? button jump to the position *before* the next marked move and draw the
  network's preferred move (green) and the move that was played (red); at a position where it is
  your turn you can try to find something better.  Shift+M / Ctrl+Left jump to the previous marked
  move, Ctrl+Right to the next.  The gift of a move is drawn as a translucent piece on the square it
  lands on (hints too).  P plays the network's move instead of the one played there (the rest of the
  old line is dropped).  The graph under the move list is the network's view of the whole game
  (white up); click it to jump; under it, each side's average loss per move and its mark counts.
  ``python review.py`` prints the same.
* Copy (Ctrl+C) puts the whole move list on the clipboard, one move per line; Ctrl+Shift+C the game
  as JSON (what Save writes).  Paste (Ctrl+V) opens either of those from the clipboard, and a
  ``.json`` game file dropped onto the window opens too.
* Variations: a different move from an earlier position (or P, the network's move) no longer throws
  the old line away: it is kept, with its review, and Tab swaps to it (Shift+Tab: the oldest kept
  line).  The board shows where the two lines part; the current line is kept in its place, so Tab
  again comes back.  Up to 12 lines are kept for the game.
* Review details: with ``--review-sims N`` (default 30) every position is searched, so a move is only
  marked when the search sees the value drop (``--review-sims 0``: the bare network).  At a marked
  position the panel lists the top three moves there with their share and value, and the move that
  was played.  Every move-list row has a small eval bar (white's share after that move; hover for the
  number); hover a dot on the graph for the loss, click it to jump to the position before that move.
* Resume: the game in progress is written to your user data folder (autosave.json, next to the saved
  games) after every move, side lines included, and reopened the next time you start play.py
  if it was not over.  R (new game) discards it; ``--no-resume`` turns all of this off.

Play feel and window
--------------------
* C = rank / file labels (flip-aware), U = move sounds, N = pick another network file for the AI
  (next to play.py, runs/*/, nets/, --net-dir), Ctrl +/- / Ctrl+0 / Ctrl+1 = window size (also --scale).
* Move menu: keys 1-9, Up / Down + Enter, Backspace.  Shift+click on the destination confirms even a
  single-variant move.  The highlighted option's gift is ghosted on the board.
* Premove: on the computer's turn click your piece and its destination; right-click / Esc cancels.
* Game > Think time sets the AI's think time (the simulation cap scales with it).

Custom agents
-------------
``--black`` / ``--white`` accept ``human``, ``ai`` (the network above), ``random``,
``ckpt:PATH`` (a trained network that samples from its policy), ``ckpt-greedy:PATH`` (same,
but always the most likely move) or ``module:function``.
The function is called as ``function(game)`` and must return one of
``game.legal_actions()``.  A trained network fits here directly.

Only pygame and numpy are needed; there is no tkinter and no pop-up window.
"""
import argparse
import glob
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pygame

import gamelog
import mcts
import numpy_net
import review as reviewlib
from chessassets import TEAMS, PTYPES, image_path
from engine import SIZE, Game, IllegalActionError, Rules, full_name, other_team, pawn_home_row
from stats import Stats, default_path as stats_default_path

# --------------------------------------------------------------------------
# Layout and colours
# --------------------------------------------------------------------------
SQ = 100                         # pixels per square
BOARD_PX = SQ * SIZE
BAR_W = 34                       # the eval bar, between the board and the panel
PANEL_X = BOARD_PX + BAR_W
PANEL_W = 252
MOVES_X = PANEL_X + PANEL_W
MOVES_W = 220                    # the move list / game history column (only on screen in review mode)
MENU_H = 30                      # the menu bar along the top
BOARD_Y = MENU_H                 # top edge of the board
ROW_H = 20
EVAL_COL_W = 34                  # the little eval bar at the right of each move-list row
MAX_VARIATIONS = 12              # side lines kept by the variation stack (Tab)
THINK_STEPS = (100, 250, 500, 1000, 2500, 5000, 10000, 20000)   # ms: the AI think-time slider's stops
PICK_ROWS = 14                   # rows of the network picker that show at once
PREMOVE_COLOUR = (70, 140, 255, 200)
STATUS_H = 84
COMPACT_W = MOVES_X              # window width: board, eval bar and side panel
WIN_W = MOVES_X + MOVES_W        # ... and with the move-list column (review mode)
WIN_H = BOARD_Y + BOARD_PX + STATUS_H
STRIP_H = 312                    # tall layout: height of the strip of controls under the board
MENU_TITLES = ("File", "Game", "Analysis", "View", "Help")

LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
SELECT = (230, 220, 60)
LAST = (205, 210, 106)
DOT = (40, 120, 220)
MOVABLE = (60, 150, 255)
LOCKED = (95, 95, 110)
BLOCKED = (230, 110, 40)
PANEL_BG = (38, 38, 42)
STATUS_BG = (24, 24, 28)
TEXT = (235, 235, 235)
MUTED = (150, 150, 155)
BAD = (255, 190, 110)
GOOD = (130, 220, 130)
BUTTON = (62, 62, 70)
BUTTON_HOVER = (84, 84, 96)
BUTTON_ON = (52, 110, 170)
BUTTON_OFF = (46, 46, 52)
HINT_COLOURS = ((60, 200, 90, 215), (240, 200, 60, 190), (235, 135, 60, 170))
HINT_TEXT = ((130, 220, 130), (240, 205, 90), (240, 160, 100))
HINT_WIDTHS = (15, 11, 8)
MARK_COLOURS = {"??": (240, 90, 80), "?": (240, 150, 70), "?!": (235, 200, 80)}
REVIEW_BEST = (60, 200, 90, 215)
REVIEW_PLAYED = (235, 80, 70, 190)

NAMES = {'p': 'pawn', 'k': 'king', 'r': 'rook', 'b': 'bishop', 'q': 'queen', 'n': 'knight'}

# A move may need these decisions, asked in this order.  Each is a field of
# engine.Action; a decision is only asked when the legal actions disagree on it.
DIMENSIONS = ('demote', 'gift', 'en_passant', 'promote')


def make_layout():
    """Rectangles of the clickable parts of the side panel and of the move-list column (the latter is only on
    screen while the window is expanded).  Fixed; only their labels change."""
    x0 = PANEL_X + 14
    w = PANEL_W - 28
    R = pygame.Rect
    return {
        'x0': x0, 'w': w,
        'net': R(x0, BOARD_Y + 40, w, 38),   # the AI's name: click to pick another network
        'eval_text_y': BOARD_Y + 372,
        'hint_y': BOARD_Y + 392,
        'stats_y': BOARD_Y + 560,
        # the move-list column
        'm_prev': R(MOVES_X + 14, BOARD_Y + 44, 92, 28), 'm_next': R(MOVES_X + 114, BOARD_Y + 44, 92, 28),
        'm_list': R(MOVES_X + 6, BOARD_Y + 134, MOVES_W - 12, ROW_H * 27),
        'm_graph': R(MOVES_X + 14, BOARD_Y + 134 + ROW_H * 27 + 12, MOVES_W - 28, 66),
    }


def make_strip_layout():
    """Rectangles of the tall layout's control strip.  The strip is part of the canvas, under the landscape
    picture (y from WIN_H), and is only shown when the window uses the tall layout: the title, the AI's name
    and the two piece trays across the top, the record on the left and the evaluation / hints on the right."""
    y = WIN_H
    R = pygame.Rect
    return {
        'x0': 14, 'w': 360, 'hint_x': 430, 'hint_w': PANEL_X - 430 - 14, 'stats_x': 14,
        'net': R(250, y + 6, PANEL_X - 250 - 14, 38),
        'eval_text_y': y + 152, 'hint_y': y + 174, 'stats_y': y + 152,
    }


def menu_item(label, run=None, key="", checked=None, enabled=True, sub=None, keep=False):
    """One entry of a drop-down menu (a separator is just ``None``).  ``sub``: a list of entries for a sub-menu;
    ``keep``: leave the menu open after the click."""
    return {"label": label, "run": run, "key": key, "checked": checked, "enabled": enabled, "sub": sub, "keep": keep}


# --------------------------------------------------------------------------
# Choosing between actions that share the same src and dst
# --------------------------------------------------------------------------

def _order(value):
    """Stable display order: None/False first, then True, then piece types."""
    if value is None or value is False:
        return -1
    if value is True:
        return 0
    return PTYPES.index(value) + 1


def next_choice(candidates):
    """First decision on which ``candidates`` still disagree, or None.

    Returns ``(dimension, sorted_values)``.  Choosing a value narrows the
    candidates; call again until None (then exactly one action is left).
    """
    for dim in DIMENSIONS:
        values = []
        for action in candidates:
            value = getattr(action, dim)
            if value not in values:
                values.append(value)
        if len(values) > 1:
            return dim, sorted(values, key=_order)
    return None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def sq_name(sq):
    """(row, col) -> 'a1'..'h8' (white at the bottom, as drawn)."""
    return "%s%d" % ("abcdefgh"[sq[1]], SIZE - sq[0])


def clipboard_put(text):
    """Put ``text`` on the system clipboard.  Tries pygame's clipboard, then the usual command line
    tools; raises RuntimeError (with every reason) if none of them worked."""
    reasons = []
    try:
        import pygame.scrap
        if not pygame.scrap.get_init():
            pygame.scrap.init()
        if hasattr(pygame.scrap, "put_text"):
            pygame.scrap.put_text(text)
        else:
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        return
    except Exception as err:          # pygame.error, NotImplementedError, ... -- reported below if all fail
        reasons.append("pygame.scrap: %s" % err)
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"],
                ["pbcopy"], ["clip"]):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5)
            return
        except (OSError, subprocess.SubprocessError) as err:
            reasons.append("%s: %s" % (cmd[0], err))
    raise RuntimeError("no working clipboard (%s)" % "; ".join(reasons))


def clipboard_get():
    """The text on the system clipboard.  Tries pygame's clipboard, then the usual command line
    tools; raises RuntimeError (with every reason) if there is nothing to read."""
    reasons = []
    try:
        import pygame.scrap
        if not pygame.scrap.get_init():
            pygame.scrap.init()
        if hasattr(pygame.scrap, "get_text"):
            text = pygame.scrap.get_text()
        else:
            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
            text = raw.decode("utf-8", "replace") if raw else ""
        text = (text or "").replace("\x00", "")
        if text.strip():
            return text
        reasons.append("pygame.scrap: empty")
    except Exception as err:          # pygame.error, NotImplementedError, ... -- reported below if all fail
        reasons.append("pygame.scrap: %s" % err)
    for cmd in (["wl-paste", "--no-newline"], ["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "--clipboard", "--output"], ["pbpaste"],
                ["powershell", "-command", "Get-Clipboard"]):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            out = subprocess.run(cmd, check=True, timeout=5, capture_output=True).stdout
        except (OSError, subprocess.SubprocessError) as err:
            reasons.append("%s: %s" % (cmd[0], err))
            continue
        text = out.decode("utf-8", "replace")
        if text.strip():
            return text
        reasons.append("%s: empty" % cmd[0])
    raise RuntimeError("nothing to paste (%s)" % "; ".join(reasons))


def describe(game, action):
    """One line for an action: 'rook a3-a1, give back a knight'."""
    piece = game.board[action.src]
    text = "%s %s-%s" % (NAMES[piece.ptype] if piece is not None else "piece",
                         sq_name(action.src), sq_name(action.dst))
    notes = []
    if action.demote:
        notes.append("un-promote")
    if action.gift:
        notes.append("give back a %s" % NAMES[action.gift])
    if action.en_passant:
        notes.append("reverse en passant")
    if action.promote:
        notes.append("promote it to a %s" % NAMES[action.promote])
    return text + (", " + ", ".join(notes) if notes else "")


def wrap(text, font, width):
    """Split ``text`` into lines no wider than ``width`` pixels."""
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if font.size(trial)[0] <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def draw_padlock(surface, x, y, colour):
    """A small padlock whose top-left corner is (x, y)."""
    pygame.draw.arc(surface, colour, pygame.Rect(x + 3, y, 10, 14), 0, math.pi, 3)
    pygame.draw.rect(surface, colour, (x, y + 7, 16, 12), border_radius=2)
    pygame.draw.circle(surface, (20, 20, 20), (x + 8, y + 13), 2)


def draw_arrow(surface, start, end, colour, width):
    """A thick arrow from start to end (pixel coordinates)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    head = min(width * 2 + 8, length * 0.6)
    base = (end[0] - ux * head, end[1] - uy * head)
    pygame.draw.line(surface, colour, start, base, width)
    px, py = -uy, ux
    pygame.draw.polygon(surface, colour, [end, (base[0] + px * head * 0.55, base[1] + py * head * 0.55),
                                          (base[0] - px * head * 0.55, base[1] - py * head * 0.55)])


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

def load_agent(spec, rng):
    """None means human; otherwise a function game -> Action."""
    if spec == 'human':
        return None
    if spec == 'random':
        return lambda game: rng.choice(game.legal_actions())
    if spec.startswith(('ckpt:', 'ckpt-greedy:')):    # a network trained by train.py
        kind, _, path = spec.partition(':')
        from net import make_agent                    # imports torch only when asked for
        return make_agent(path, greedy=(kind == 'ckpt-greedy'),
                          rng=np.random.default_rng(rng.randrange(2 ** 32)))
    module, _, name = spec.partition(':')
    if not name:
        raise SystemExit("agent must be 'human', 'ai', 'random' or 'module:function', got %r" % spec)
    return getattr(importlib.import_module(module), name)


def app_dir():
    """Folder holding the program (and ai.npz): works from source and from a PyInstaller bundle."""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def auto_scale(full=(WIN_W, WIN_H)):
    """The window scale that fits the desktop (at most 1): 1114 x 914 pixels at 1.0, 1334 x 914 in review mode
    (``full``: the window's size at 1.0)."""
    try:
        width, height = pygame.display.get_desktop_sizes()[0]
    except Exception:                                  # old pygame, no display: no opinion
        return 1.0
    if width <= 0 or height <= 0:
        return 1.0
    return max(0.4, min(1.0, (height - 90) / full[1], (width - 30) / full[0]))


def find_tiers(extra_dirs=()):
    """Difficulty levels written by ``tiers.py``: ``[{name, path, search, loss_rate}]``, easiest first.  Read from
    ``nets/tiers.json`` next to play.py, in the working directory or in ``extra_dirs``; entries whose file is
    missing are left out."""
    for base in [app_dir(), os.getcwd()] + list(extra_dirs):
        manifest = os.path.join(base or "", "nets", "tiers.json")
        if not os.path.isfile(manifest):
            continue
        try:
            with open(manifest, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        tiers = []
        for entry in data.get("tiers", []):
            path = entry.get("path") or ""
            path = path if os.path.isabs(path) else os.path.join(base, path)
            if os.path.isfile(path):
                tiers.append({"name": str(entry.get("name", os.path.basename(path))), "path": path,
                              "search": bool(entry.get("search", False)),
                              "loss_rate": entry.get("loss_rate")})
        if tiers:
            return tiers
    return []


def think_label(ms):
    return "%d ms" % ms if ms < 1000 else "%.1f s" % (ms / 1000.0)


def find_nets(extra_dirs=()):
    """Network files that could be the AI: ``[{path, rel, version, mtime}]``, grouped by folder, newest
    version first.  Looks next to play.py, in the working directory and in ``extra_dirs``, each for
    ``*.npz`` / ``*.pt`` and under ``runs/*/`` and ``nets/`` (train.py's ``iter_NNNNN.pt`` snapshots)."""
    bases = []
    for base in [app_dir(), os.getcwd()] + list(extra_dirs):
        if base and os.path.isdir(base) and os.path.realpath(base) not in [os.path.realpath(b) for b in bases]:
            bases.append(base)
    found = {}
    for base in bases:
        for pattern in ("*.npz", "*.pt", os.path.join("runs", "*", "*.pt"), os.path.join("runs", "*", "*.npz"),
                        os.path.join("nets", "*.npz"), os.path.join("nets", "*.pt")):
            for path in glob.glob(os.path.join(base, pattern)):
                name = os.path.basename(path)
                if ".tmp." in name or not os.path.isfile(path):
                    continue
                found.setdefault(os.path.realpath(path), (path, os.path.relpath(path, base)))
    items = []
    for real, (path, rel) in found.items():
        version = None
        match = re.search(r"iter_(\d+)", os.path.basename(path))
        if match:
            version = int(match.group(1))
        elif path.lower().endswith(".npz"):
            try:
                with np.load(path, allow_pickle=False) as data:
                    if "meta_version" in data.files and int(data["meta_version"]) >= 0:
                        version = int(data["meta_version"])
            except Exception:                          # not one of ours: still listed, loading will say why
                pass
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        items.append({"path": path, "real": real, "rel": rel, "version": version, "mtime": mtime})
    items.sort(key=lambda it: (os.path.dirname(it["real"]), -(it["version"] if it["version"] is not None else -1),
                               -it["mtime"], it["real"]))
    return items[:500]


def synth_sounds():
    """The three move sounds (``move``, ``gift``, ``end``) as pygame Sounds matching the open mixer, or {}
    if there is no audio output.  They are made with numpy, so there is no sound file to ship."""
    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()
        rate, fmt, channels = pygame.mixer.get_init()
    except pygame.error:
        return {}
    if fmt != -16 or channels not in (1, 2):
        return {}
    rng = np.random.default_rng(7)

    def note(partials, dur, knock=0.0):
        t = np.arange(int(rate * dur)) / float(rate)
        wave = sum(amp * np.sin(2 * np.pi * f * t) * np.exp(-t * decay) for f, amp, decay in partials)
        if knock:
            wave = wave + knock * rng.standard_normal(len(t)) * np.exp(-t * 70)
        return wave * np.minimum(1.0, t / 0.002)             # a 2 ms attack: no click

    def make(wave):
        data = (np.clip(wave, -1.0, 1.0) * 0.45 * 32767).astype(np.int16)
        if channels == 2:
            data = np.repeat(data[:, None], 2, axis=1)
        return pygame.mixer.Sound(buffer=np.ascontiguousarray(data).tobytes())

    try:
        return {
            "move": make(note([(190, 1.0, 40), (380, 0.4, 60)], 0.12, knock=0.3)),
            "gift": make(np.concatenate([note([(660, 0.8, 14)], 0.11), note([(990, 0.8, 11)], 0.2)])),
            "end": make(np.concatenate([note([(392, 0.8, 7)], 0.2), note([(294, 0.8, 5)], 0.2),
                                        note([(196, 0.9, 4)], 0.45)])),
        }
    except (pygame.error, ValueError):
        return {}


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

class App:
    def __init__(self, rules, delay_ms, rng, advisor, ai_note, custom, ai_on, ai_sides,
                 stats, show_eval, searcher=None, flipped=False, autosave=False,
                 review_sims=0, session=False, scale=None, sound=True, coords=True, net_dirs=(),
                 auto_orient=True, tall=False):
        pygame.init()
        pygame.display.set_caption("Reverse Chess")
        # See "the window" below: a scaled canvas under a menu bar drawn at window level.
        self.expanded = False
        self.win_w, self.win_h = COMPACT_W, WIN_H
        self.canvas = self.screen = pygame.Surface((self.win_w, self.win_h + STRIP_H), 0, 32)
        self.window = None
        self.os_window = None
        self.size_wanted, self.size_deadline = None, 0.0
        self.last_size, self.refused, self.seen_size = None, None, None
        self.requested, self.resizing_until = set(), 0.0
        self.regions, self.mode = [], "landscape"
        self.tall = bool(tall)
        self.strip = make_strip_layout()
        self.scale = 1.0
        self.view = pygame.Rect(0, MENU_H, self.win_w, self.win_h - MENU_H)   # where the canvas sits in the window
        self.fit_screen() if scale is None else self.apply_scale(scale)
        self.font = pygame.font.SysFont(None, 28)
        self.font_big = pygame.font.SysFont(None, 38)
        self.font_small = pygame.font.SysFont(None, 20)
        self.font_status = pygame.font.SysFont(None, 25)
        self.font_btn = pygame.font.SysFont(None, 23)
        self._sprites = {}
        self.layout = make_layout()
        self.rules = rules
        self.delay_ms = delay_ms
        self.rng = rng
        self.advisor = advisor        # numpy_net.Advisor or None (no AI file)
        self.home_advisor = advisor
        self.settings_path, self.settings_saved, self.settings_next = None, None, 0.0
        self.eval_advisor = advisor   # what the eval bar, hints and review use: the main (insane) AI, whatever difficulty plays
        self.ai_note = ai_note        # why there is no advisor, shown in the panel
        self.custom = custom          # {'b': fn|None, 'w': fn|None}: agents given on the command line
        self.ai_on = ai_on
        self.ai_sides = set(ai_sides)
        self.stats = stats
        self.show_eval = show_eval and advisor is not None
        self.searcher = searcher if advisor is not None else None   # mcts.Searcher or None
        self.use_search = self.searcher is not None
        self.search_job = None
        self.ana_searcher, self.ana_job = None, None     # the main AI's search for the eval bar / hints (see analysis_searcher)
        self.default_ai = self.make_default_ai()
        self.base_think = self.searcher.think_ms if self.searcher is not None else 2500
        self.base_sims = self.searcher.sims if self.searcher is not None else 600
        self.think_ms = self.base_think   # the think-time slider (scales the search's simulation cap with it)
        self.stats_summary = stats.summary() if stats is not None else None
        self.confirm_reset_until = 0
        self.flipped = flipped        # board drawn from black's side
        self.auto_orient = auto_orient  # turn the board so the human's colour is at the bottom
        self.autosave = autosave      # save every finished game (--autosave)
        self.review_sims = review_sims  # search simulations per position in the review (0: network only)
        self.session = session        # keep the game in progress on disk after every move (resume)
        self.variations = []
        self.tooltip = None           # (text, colour) for the hovered graph mark / eval chip, set while drawing
        self.load_index = -1          # which saved game the Load button opened last
        self.coords = coords          # rank / file labels on the board edge (C)
        self.sound_on = sound         # move sounds (U)
        self.sounds = None            # built on first use (synth_sounds); {} = no audio
        self.net_dirs = list(net_dirs)
        self.picker = None            # the open network picker (N), see open_picker
        self.help_open = False        # modal in-app shortcut / control reference (F1 or ?)
        self.premove = None           # (src, dst) queued while the computer thinks
        self.pre_src = None           # the piece picked for a premove, waiting for its destination
        self.menu_sel = None          # the menu option highlighted from the keyboard
        self.menu_open = None         # index (into MENU_TITLES) of the open drop-down menu, or None
        self.menu_sub = None          # label of the entry whose sub-menu is open
        self.moves_mode = 'auto'      # the move-list column: 'auto' (review / browsing), 'on' or 'off'
        self.tiers = find_tiers(self.net_dirs)
        self.refresh_agents()
        self.new_game()
        self.orient()
        if stats is not None and stats.error:
            self.message = stats.error

    # -- the window: a menu bar at the top, and below it the game, scaled to fit --------------
    # The game is drawn on a canvas.  Its top part is the "wide" picture: win_w x win_h logical pixels, the board,
    # the eval bar and the side panel in a row (COMPACT_W wide, WIN_W with the move-list column, see
    # sync_window), of which the top MENU_H rows are unused: the menu bar is drawn straight onto the window
    # afterwards, at full window width and at its own fixed size, like any other Linux program's.  Under it
    # the canvas holds the control strip of the "tall" layout (View > Layout), the same controls arranged
    # for a window that is taller than wide.  The window keeps the shape of whichever layout is in use: when
    # the user drags an edge or a corner, the other dimension follows (on_resize), so there is never a blank
    # strip, and every resize makes the board, and everything else, bigger or smaller.
    def content_size(self, tall=None):
        """(width, height) in canvas pixels of what the layout shows, without the menu bar."""
        tall = self.tall if tall is None else tall
        if tall:
            return PANEL_X + (MOVES_W if self.expanded else 0), BOARD_PX + STATUS_H + STRIP_H
        return self.win_w, self.win_h - MENU_H

    def window_size_for(self, factor, tall=None):
        """Window pixels that show the layout at ``factor`` times its size (the menu bar is not scaled)."""
        cw, ch = self.content_size(tall)
        return int(round(cw * factor)), MENU_H + int(round(ch * factor))

    def set_window_size(self, size):
        """Resize the open window.  This resizes the existing SDL window in place; calling set_mode again
        instead re-creates the window, which blanks / re-animates it (a flicker and a fade on most desktops)."""
        size = (max(1, int(size[0])), max(1, int(size[1])))
        self.last_size = self.seen_size = size
        if self.window is not None:
            try:
                if self.os_window is None:
                    from pygame._sdl2 import video as sdl_video
                    self.os_window = sdl_video.Window.from_display_module()
                if tuple(self.window.get_size()) != size:
                    self.requested = {size}
                    self.os_window.size = size
                    self.size_wanted, self.size_deadline = size, time.time() + 0.5
                return
            except Exception:              # no such API / no such window (headless): the old way still works
                self.os_window = None
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
        try:
            from pygame._sdl2 import video as sdl_video
            self.os_window = sdl_video.Window.from_display_module()
        except Exception:
            self.os_window = None
        try:
            self.os_window.minimum_size = (320, MENU_H + 240)      # (pygame-ce only)
        except Exception:
            pass

    def on_resize(self, size):
        """The user resized (or maximised) the window.  The window manager owns the size while it is being
        dragged, so nothing is asked of it: the game is simply re-fitted and centred in whatever it gives us."""
        self.resizing_until = time.time() + 0.35          # (drawing is cheaper while this lasts)
        self.seen_size = (int(size[0]), int(size[1]))

    def apply_scale(self, factor):
        """Resize the window to show the layout at ``factor`` (1.0 = 1114 x 914, or 1334 x 914 with the
        move-list column, in the wide layout)."""
        factor = max(0.4, min(4.0, factor))
        size = self.window_size_for(factor)
        self.set_window_size(size)
        self.fit_view(size)

    def fit_screen(self):
        """The largest zoom (at most 1) at which the whole window fits the desktop."""
        self.apply_scale(auto_scale(self.window_size_for(1.0) if self.tall else (WIN_W, WIN_H)))

    def set_tall(self, tall):
        """View > Layout: the side panel beside the board (wide), or under it (tall)."""
        if tall == self.tall:
            return
        self.tall = tall
        self.refused = None
        try:
            dw, dh = pygame.display.get_desktop_sizes()[0]
        except Exception:
            dw, dh = 10 ** 6, 10 ** 6
        cw, ch = self.content_size()
        s = min(self.scale, max(0.4, (dh - 90 - MENU_H) / float(ch)), max(0.4, (dw - 30) / float(cw)))
        self.apply_scale(s)

    def fit_view(self, size=None):
        """Lay the canvas out in the window (see ``compute_regions``)."""
        ww, wh = size or self.window.get_size()
        self.compute_regions(ww, max(1, wh - MENU_H))

    def compute_regions(self, ww, ah):
        """Decide how the window shows the canvas.  Wide layout: the landscape picture as it is.  Tall
        layout: the board and the status line on top at the full window width, and under them the control
        strip (the canvas rows under WIN_H), with the move-list column, when there is one, at the right.
        Every part is a ``(canvas rect, (x, y) in the window, scale)``.  (While the quick reference is open
        the wide picture is always used, fitted into the window: it is a modal page made for it.)"""
        tall = self.tall and not getattr(self, 'help_open', False)
        exp = self.expanded
        R = pygame.Rect
        if tall:
            cw, ch = self.content_size(True)
            s = min(ww / float(cw), ah / float(ch))
            top_h = BOARD_PX + STATUS_H
            regions = [(R(0, BOARD_Y, PANEL_X, top_h), (0, MENU_H), s),
                       (R(0, self.win_h, PANEL_X, STRIP_H), (0, MENU_H + int(round(top_h * s))), s)]
            if exp:
                regions.append((R(MOVES_X, BOARD_Y, MOVES_W, self.win_h - BOARD_Y), (int(round(PANEL_X * s)), MENU_H), s))
            self.mode = "portrait"
        else:
            s = min(ww / float(self.win_w), ah / float(self.win_h - MENU_H))
            ox = max(0, (ww - int(round(self.win_w * s))) // 2)          # centred: spare room is split evenly
            oy = max(0, (ah - int(round((self.win_h - MENU_H) * s))) // 2)
            regions = [(R(0, MENU_H, self.win_w, self.win_h - MENU_H), (ox, MENU_H + oy), s)]
            self.mode = "landscape"
        self.regions = regions
        self.scale = s
        w = max(int(round(r.w * sc)) + dx for r, (dx, dy), sc in regions)
        h = max(int(round(r.h * sc)) + dy for r, (dx, dy), sc in regions) - MENU_H
        self.view = pygame.Rect(0, MENU_H, max(1, w), max(1, h))

    def want_expanded(self):
        """Is the move-list column needed?  By default only while a review exists or runs, or while an
        earlier position is on the board (View > Move list overrides this)."""
        if self.moves_mode == 'on':
            return True
        if self.moves_mode == 'off':
            return False
        return self.review is not None or self.reviewer is not None or self.browsing

    def sync_window(self):
        """Make room for the move-list column, or give it back, keeping the zoom."""
        want = self.want_expanded()
        if want == self.expanded:
            return
        self.expanded = want
        self.win_w = WIN_W if want else COMPACT_W
        self.canvas = self.screen = pygame.Surface((self.win_w, self.win_h + STRIP_H), 0, 32)
        size = self.window_size_for(self.scale)
        self.set_window_size(size)
        self.fit_view(size)

    def zoom(self, factor):
        self.apply_scale(self.scale * factor)
        self.message = "Window scale %d%%  (Ctrl +/- resize, Ctrl+0 fit the screen, Ctrl+1 100%%)." % round(100 * self.scale)

    def present(self):
        """Show the finished frame: the canvas parts arranged and scaled into the window, the menu bar on top."""
        window = pygame.display.get_surface()
        if self.size_wanted is not None:
            if window.get_size() == self.size_wanted:
                self.size_wanted = None
            elif time.time() > self.size_deadline:        # the window manager said no (maximised, tiled ...)
                self.refused = self.seen_size = tuple(window.get_size())
                self.requested = set()
                self.size_wanted = None
        self.window = window
        self.fit_view()
        window.fill(PANEL_BG)
        fast = time.time() < self.resizing_until          # while dragging: cheap scaling, smooth once it stops
        scale_fn = pygame.transform.scale if fast else pygame.transform.smoothscale
        for src, (dx, dy), sc in self.regions:
            src = src.clip(self.canvas.get_rect())
            size = (max(1, int(round(src.w * sc))), max(1, int(round(src.h * sc))))
            part = self.canvas.subsurface(src).copy()
            window.blit(part if size == src.size else scale_fn(part, size), (dx, dy))
        self.draw_menubar(window)
        self.draw_dropdown(window)
        pygame.display.flip()

    def to_logical(self, pos):
        """A window pixel -> the canvas pixel under it (nowhere on the canvas: far outside it)."""
        for src, (dx, dy), sc in self.regions:
            if dx <= pos[0] < dx + int(round(src.w * sc)) and dy <= pos[1] < dy + int(round(src.h * sc)):
                return (src.x + int(math.floor((pos[0] - dx) / sc)), src.y + int(math.floor((pos[1] - dy) / sc)))
        return (-9999, -9999)

    def raw_mouse(self):
        """The pointer in window pixels (the menu bar and its drop-downs live there, unscaled)."""
        return pygame.mouse.get_pos()

    def mouse(self):
        return self.to_logical(pygame.mouse.get_pos())

    # -- who plays what --------------------------------------------------
    def agent_for(self, team):
        if not (self.ai_on and team in self.ai_sides):
            return None
        return self.custom.get(team) or self.default_ai

    def refresh_agents(self):
        self.agents = {team: self.agent_for(team) for team in TEAMS}

    def stats_eligible(self):
        """True when this setup is 'a human against the network', the only thing the record counts."""
        if self.stats is None or self.advisor is None or not self.ai_on or len(self.ai_sides) != 1:
            return False
        return self.custom.get(next(iter(self.ai_sides))) is None

    def human_colour(self):
        return other_team(next(iter(self.ai_sides)))

    def make_default_ai(self):
        """The computer's move function for the network now loaded (random moves with none)."""
        advisor, rng = self.advisor, self.rng
        if advisor is None:
            return lambda game: rng.choice(game.legal_actions())
        np_rng = np.random.default_rng(rng.randrange(2 ** 32))
        return lambda game: advisor.choose(game, np_rng)

    def set_advisor(self, advisor, tier=False):
        """Swap the network that plays, searches, evaluates and reviews.  A difficulty level (``tier``) changes
        only who plays: the eval bar, hints and review keep using the main AI (ai.npz)."""
        self.stop_search()
        self.advisor = advisor
        if not tier or self.eval_advisor is None:
            self.eval_advisor = advisor
            self.ana_searcher = self.ana_job = None
        self.ai_note = ""
        self.default_ai = self.make_default_ai()
        if self.searcher is None:
            self.searcher = mcts.Searcher(advisor, sims=self.sims_for(self.think_ms), think_ms=self.think_ms)
            self.use_search = True
        else:
            self.searcher.advisor = advisor
        if not tier:                                    # (a difficulty level leaves hints, eval and review alone)
            self.hint = None
            self.bar_pos = None
            self.drop_review()                          # those ratings were the old network's
            for line in self.variations:
                line["review"] = None
        if self.game.ply > 0 and not self.game.done:
            self.stats_ok = False                       # the opponent changed part-way
        elif self.game.ply == 0:
            self.stats_ok = self.stats_eligible()
        self.refresh_agents()
        self.cancel()
        self.clear_premove()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        self.message = "The AI is now %s." % advisor.name

    # -- board coordinates, sound, think time ----------------------------------
    def toggle_coords(self):
        self.coords = not self.coords

    def toggle_sound(self):
        if self.sound_on:
            self.sound_on = False
            return
        if self.sounds is None:
            self.sounds = synth_sounds()
        if not self.sounds:
            self.message = "No audio output available, so no sound."
            return
        self.sound_on = True
        self.sounds["move"].play()

    def play_sound(self, action):
        if not self.sound_on:
            return
        if self.sounds is None:
            self.sounds = synth_sounds()
        name = "end" if self.game.done else "gift" if action.gift else "move"
        sound = self.sounds.get(name)
        if sound is not None:
            sound.play()

    def sims_for(self, think_ms):
        """The search's simulation cap that goes with ``think_ms`` (both grow together, so the slider
        does something whether this machine is limited by time or by the cap)."""
        return max(10, int(round(self.base_sims * think_ms / float(self.base_think))))

    def set_think(self, ms):
        self.think_ms = ms
        if self.searcher is not None:
            self.searcher.think_ms = ms
            self.searcher.sims = self.sims_for(ms)
        if self.ana_searcher is not None:
            self.ana_searcher.think_ms = ms
            self.ana_searcher.sims = self.sims_for(ms)

    def think_enabled(self):
        return self.searcher is not None

    # -- premove: queue a move while the computer thinks ---------------------
    def premove_possible(self):
        game = self.game
        return (not game.done and not self.browsing and self.agents[game.turn] is not None
                and self.agents[other_team(game.turn)] is None)

    def clear_premove(self):
        self.premove = self.pre_src = None

    def right_click(self):
        if self.menu_open is not None:
            self.close_menus()
        elif self.premove is not None or self.pre_src is not None:
            self.clear_premove()
            self.message = "Premove cancelled."

    def premove_click(self, pos):
        """A click on the computer's turn: pick one of your pieces, then where it goes."""
        x, y = pos
        if not (0 <= x < BOARD_PX and BOARD_Y <= y < BOARD_Y + BOARD_PX) or not self.premove_possible():
            return
        mine = other_team(self.game.turn)
        sq = self.xy_sq(x, y)
        piece = self.game.board[sq]
        own = piece is not None and piece.team == mine
        if self.pre_src is None:
            if own:
                self.pre_src = sq
                self.message = "Premove: now click where it goes (right-click cancels)."
            else:
                self.message = "It is the computer's turn.  Click one of your pieces to queue a premove."
        elif sq == self.pre_src:
            self.clear_premove()
            self.message = ""
        elif own:
            self.pre_src = sq                           # picked another piece instead
        else:
            self.premove = (self.pre_src, sq)
            self.pre_src = None
            self.message = "Premove %s-%s queued; it is played if still legal (right-click cancels)." % (
                sq_name(self.premove[0]), sq_name(sq))

    def run_premove(self):
        """It is the human's turn and a premove is waiting: play it, or say why not."""
        src, dst = self.premove
        self.clear_premove()
        candidates = [a for a in self.by_src.get(src, []) if a.dst == dst]
        if not candidates:
            self.message = "Premove %s-%s is not legal now." % (sq_name(src), sq_name(dst))
            return
        self.selected = src
        self.candidates = candidates
        self.history = []
        self.message = ""
        self.resolve()                                  # a move that needs a decision opens its menu

    # -- the network picker (N) ------------------------------------------------
    def open_picker(self):
        self.close_menus()
        items = find_nets(self.net_dirs)
        if not items:
            self.message = ("No network files found (looked next to play.py, in runs/*/ and nets/, and in "
                            "--net-dir folders).")
            return
        cur = os.path.realpath(self.advisor.path) if self.advisor is not None else None
        sel = next((i for i, it in enumerate(items) if it["real"] == cur), 0)
        self.picker = {"items": items, "sel": sel, "top": max(0, sel - PICK_ROWS // 2), "cur": cur}
        self.picker_visible()

    def close_picker(self):
        self.picker = None

    def picker_visible(self):
        """Scroll the picker so the selected row shows."""
        pk = self.picker
        rows = min(PICK_ROWS, len(pk["items"]))
        pk["top"] = max(0, min(pk["top"], len(pk["items"]) - rows))
        if pk["sel"] < pk["top"]:
            pk["top"] = pk["sel"]
        elif pk["sel"] >= pk["top"] + rows:
            pk["top"] = pk["sel"] - rows + 1

    def picker_rects(self):
        """(panel, one rect per visible row, X button) of the picker."""
        n = min(PICK_ROWS, len(self.picker["items"]))
        width, row = 700, 30
        height = 62 + n * row + 30
        panel = pygame.Rect((BOARD_PX - width) // 2, BOARD_Y + (BOARD_PX - height) // 2, width, height)
        rows = [pygame.Rect(panel.x + 14, panel.y + 56 + i * row, width - 28, row - 2) for i in range(n)]
        return panel, rows, pygame.Rect(panel.right - 46, panel.y + 10, 34, 34)

    def picker_scroll(self, dy):
        pk = self.picker
        pk["top"] -= 3 * dy
        pk["top"] = max(0, min(pk["top"], max(0, len(pk["items"]) - PICK_ROWS)))

    def picker_key(self, key):
        pk = self.picker
        n = len(pk["items"])
        if key == pygame.K_ESCAPE:
            self.close_picker()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.picker_choose(pk["sel"])
        elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_HOME, pygame.K_END):
            step = {pygame.K_UP: -1, pygame.K_DOWN: 1, pygame.K_PAGEUP: -PICK_ROWS, pygame.K_PAGEDOWN: PICK_ROWS}
            pk["sel"] = (0 if key == pygame.K_HOME else n - 1 if key == pygame.K_END
                         else max(0, min(n - 1, pk["sel"] + step[key])))
            self.picker_visible()

    def click_picker(self, pos):
        panel, rows, x_rect = self.picker_rects()
        if x_rect.collidepoint(pos) or not panel.collidepoint(pos):
            self.close_picker()
            return
        for i, rect in enumerate(rows):
            if rect.collidepoint(pos):
                self.picker_choose(self.picker["top"] + i)
                return

    def picker_choose(self, index):
        path = self.picker["items"][index]["path"]
        self.close_picker()
        try:
            advisor = numpy_net.Advisor(path, self.advisor.max_plies if self.advisor is not None else 300)
        except Exception as err:                        # a damaged file, a .pt without PyTorch, ...
            self.message = "Could not load %s: %s" % (os.path.basename(path), err)
            return
        self.set_advisor(advisor)

    def menu_geometry(self, n, with_back=False):
        if self.mode == "portrait":                 # the tall layout: the menu takes the control strip's place
            return menu_rects_tall(n, with_back)
        return menu_rects(n, with_back, self.win_w, self.win_h)

    # -- the move menu from the keyboard, and a preview of the highlighted option -----------
    def menu_key(self, key):
        """Keys while a move menu is open: 1-9 choose, Up / Down + Enter choose, Backspace goes back."""
        dim, values = self.pending
        digits = {pygame.K_1 + i: i for i in range(9)}
        digits.update({pygame.K_KP1 + i: i for i in range(9)})
        if key in digits:
            if digits[key] < len(values):
                self.choose(digits[key])
        elif key in (pygame.K_UP, pygame.K_DOWN):
            step = -1 if key == pygame.K_UP else 1
            self.menu_sel = (len(values) - 1 if step < 0 else 0) if self.menu_sel is None \
                else (self.menu_sel + step) % len(values)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.choose(self.menu_sel if self.menu_sel is not None else 0)
        elif key == pygame.K_BACKSPACE:
            self.menu_back() if self.history else self.close_menu()

    def menu_preview_action(self):
        """The action the highlighted (hovered, or keyboard-selected) menu option would lead to, or None.
        With a choice of variants it is the plainest one (no reverse en passant, no promotion)."""
        if self.pending is None:
            return None
        dim, values = self.pending
        _, rects, _, _ = self.menu_geometry(len(values), bool(self.history))
        mouse = self.mouse()
        index = next((i for i, rect in enumerate(rects) if rect.collidepoint(mouse)), self.menu_sel)
        if index is None or index >= len(values):
            return None
        pool = self.candidates if dim == 'confirm' else [a for a in self.candidates if getattr(a, dim) == values[index]]
        if len({a.gift for a in pool}) != 1:          # which piece is given is not settled yet: nothing to show
            return None
        return min(pool, key=lambda a: (a.en_passant, a.promote is not None))

    # -- state -----------------------------------------------------------
    def new_game(self):
        self.game = Game(self.rules)
        self.selected = None          # square of the piece being moved
        self.candidates = []          # actions for the chosen src -> dst
        self.pending = None           # (dimension, values) while a menu is open
        self.history = []             # candidate lists before each answered question (for Back)
        self.message = ""
        self.hint = None
        self.hint_used = False
        self.recorded = False
        self.stats_ok = self.stats_eligible()
        self.bar_pos = None
        self.clear_premove()
        self.stop_search()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        self.orient()
        self.line = []                # every action of the game, and the current position in it:
        self.texts = []               # (notation of each action, the team that played it)
        self.turns = []
        self.pos = 0                  # pos < len(line) = looking at an earlier position
        self.moves_top = None         # first visible row of the move list (None = follow the game)
        self.variations = []          # lines cut off by a different move, newest first (see stash_line)
        self.drop_review()
        self.reindex()

    def restart(self):
        """Start a new game; one abandoned part-way is noted in the record."""
        self.clear_session()
        if self.stats_ok and not self.recorded and not self.game.done and self.game.ply > 0:
            self.stats.add(self.human_colour(), "abandoned", self.game.ply, self.hint_used,
                           self.advisor.version)
            self.stats_summary = self.stats.summary()
        self.new_game()

    def reindex(self):
        """Group the side to move's legal actions by source square."""
        self.by_src = defaultdict(list)
        for action in self.game.legal_actions():
            self.by_src[action.src].append(action)
        # Pieces that are not (permanently) locked but cannot move right now.
        # For the side to move this means "no playable action"; for the other
        # side, "no raw move".  Computed once per ply, not once per frame.
        self.blocked = set()
        game = self.game
        if not game.done:
            for team in TEAMS:
                for piece in game.on_board(team):
                    if piece.locked:
                        continue
                    if team == game.turn:
                        stuck = piece.pos not in self.by_src
                    else:
                        stuck = not game.piece_moves(piece, first_only=True)
                    if stuck:
                        self.blocked.add(piece.pos)

    def cancel(self):
        """Back to the board with nothing selected."""
        self.selected = None
        self.candidates = []
        self.pending = None
        self.menu_sel = None
        self.history = []
        self.message = ""

    def close_menu(self):
        """Close the move menu without playing; the piece stays selected."""
        self.candidates = []
        self.pending = None
        self.menu_sel = None
        self.history = []
        self.message = ""

    def menu_back(self):
        """Undo the last answer in the menu (only offered when there is one)."""
        self.candidates = self.history.pop()
        self.pending = next_choice(self.candidates)
        self.menu_sel = None

    def escape(self):
        if self.pending is not None:
            self.close_menu()
        elif self.premove is not None or self.pre_src is not None:
            self.clear_premove()
            self.message = ""
        else:
            self.cancel()

    def play(self, action):
        text, team = gamelog.move_text(self.game, action), self.game.turn
        try:
            self.game.step(action)
        except IllegalActionError as err:
            self.message = str(err)
        else:
            self.message = ""
            self.play_sound(action)
            if self.pos < len(self.line) and self.line[self.pos] == action:
                # the move that is already next in the line: nothing branches, nothing is lost
                self.pos += 1
                self.moves_top = None
                self.cancel_keep_message()
                self.reindex()
                self.ai_due = pygame.time.get_ticks() + self.delay_ms
                return
            if self.pos < len(self.line):      # a different move from an earlier position
                self.stash_line(self.pos)      # the old line is kept (Tab brings it back)
                self.truncate(self.pos)
                self.stats_ok = False          # took a move back: not a clean game against the AI
            self.line.append(action)
            self.texts.append(text)
            self.turns.append(team)
            self.pos += 1
            self.moves_top = None
        self.cancel_keep_message()
        self.reindex()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        self.save_session()
        if self.game.done:
            self.record_finished()
            if self.autosave:
                self.save()
            if self.advisor is not None:
                self.start_review()

    # -- game history: step, undo, flip, save, load, review --------------
    @property
    def browsing(self):
        return self.pos < len(self.line)

    def truncate(self, n):
        """Forget every move after the first ``n`` (the review keeps what is still true)."""
        del self.line[n:], self.texts[n:], self.turns[n:]
        rev = self.review
        if rev is not None:
            for key in ("best", "loss", "marks", "top"):
                if key in rev:
                    rev[key] = rev[key][:n]
            rev["values"] = rev["values"][:n + 1]
        elif self.reviewer is not None:
            self.reviewer.stop()
            self.reviewer = None

    def jump(self, n):
        """Show the position after the first ``n`` moves of the game (nothing is lost)."""
        n = max(0, min(len(self.line), n))
        if n == self.pos:
            return
        self.stop_search()
        self.hint = None
        self.clear_premove()
        old = self.game
        self.game = gamelog.replay(self.rules, self.line, n)
        old.close()
        self.pos = n
        self.moves_top = None
        self.cancel()
        self.reindex()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms

    def step_back(self):
        self.jump(self.pos - 1)

    def step_forward(self):
        self.jump(self.pos + 1)

    def takeback(self):
        """Back to the last position where it was a human's turn (so the computer's reply goes too)."""
        if self.pos == 0:
            self.message = "Nothing to take back."
            return
        target = self.pos - 1
        for n in range(self.pos - 1, -1, -1):
            if self.agents[self.turns[n]] is None:
                target = n
                break
        self.jump(target)
        self.message = "Taken back; Right / End bring the moves back, a new move starts another line (Tab)."

    def flip(self):
        self.flipped = not self.flipped

    def orient(self):
        """With exactly one human side, draw the board from that side (your pieces at the bottom, moving
        up).  Two humans or two computers: leave the board as it is.  F still flips it by hand."""
        if self.auto_orient and self.ai_on and len(self.ai_sides) == 1:
            self.flipped = self.human_colour() == 'b'

    def game_meta(self):
        meta = {"ai": self.advisor.name if self.advisor is not None else None,
                "ai_sides": sorted(self.ai_sides) if self.ai_on else []}
        return meta

    def save(self):
        if not self.line:
            self.message = "Nothing to save yet."
            return
        try:
            path = gamelog.save_game(self.rules, self.line, self.game_meta())
        except OSError as err:
            self.message = "Could not save the game: %s" % err
            return
        self.load_index = -1
        self.message = "Saved " + path

    def load(self, path=None):
        """Open ``path`` (default: the newest saved game; again: the one before it)."""
        if path is None:
            saved = gamelog.saved_games()
            if not saved:
                self.message = "No saved games yet (Ctrl+S saves the current one)."
                return
            self.load_index = (self.load_index + 1) % len(saved)
            path = saved[self.load_index]
            where = " (%d of %d)" % (self.load_index + 1, len(saved))
        else:
            where = ""
        try:
            rules, actions, meta = gamelog.load_game(path)
        except (OSError, ValueError, KeyError, IllegalActionError) as err:
            self.message = "Could not load %s: %s" % (os.path.basename(path), err)
            return
        self.open_game(rules, actions, "Loaded %s%s" % (os.path.basename(path), where))

    def open_game(self, rules, actions, message, review=True, variations=None):
        """Make ``actions`` (a legal line under ``rules``) the game on the board, at its end."""
        texts, turns = gamelog.notation_and_turns(rules, actions)
        self.rules = rules
        self.new_game()
        self.stats_ok = False                       # not a game played against the AI just now
        self.line, self.texts, self.turns = list(actions), texts, turns
        self.game.close()
        self.game = gamelog.replay(rules, actions)
        self.pos = len(actions)
        self.recorded = True
        self.variations = [dict(v, review=None) for v in (variations or [])][:MAX_VARIATIONS]
        self.reindex()
        self.message = message
        self.save_session()
        if review and self.advisor is not None and actions:
            self.start_review()

    def paste_text(self, text):
        """Open a game from text: the JSON that Ctrl+Shift+C copies, or a move list as Ctrl+C copies it."""
        text = text.strip()
        if not text:
            self.message = "The clipboard is empty."
            return
        try:
            if text.startswith("{"):
                rules, actions, _ = gamelog.loads_game(text)
            else:
                rules, actions = self.rules, gamelog.parse_move_list(self.rules, text)
        except (ValueError, KeyError, IllegalActionError) as err:
            self.message = "Could not paste a game: %s" % err
            return
        self.open_game(rules, actions, "Pasted a game of %d moves." % len(actions))

    def paste(self):
        try:
            text = clipboard_get()
        except RuntimeError as err:
            self.message = "Could not paste: %s" % err
            return
        self.paste_text(text)

    def drop_file(self, path):
        """A file dropped on the window."""
        if path.lower().endswith(".json"):
            self.load(path)
        else:
            self.message = "Drop a saved game (.json) here, or paste one with Ctrl+V."

    # -- the session: resume after a crash or a closed window -----------------
    def save_session(self):
        """Write the game in progress (and its side lines) to disk; called after every move."""
        if not self.session or not self.line:
            return
        try:
            gamelog.write_autosave(self.rules, self.line, self.texts, self.variations, self.game_meta())
        except OSError:
            pass                                    # never interrupt a game over a full disk

    def clear_session(self):
        if self.session:
            gamelog.clear_autosave()

    def resume(self):
        """Reopen the game the last session left unfinished.  True if there was one."""
        try:
            got = gamelog.read_autosave()
        except (OSError, ValueError, IllegalActionError) as err:
            self.message = "Could not resume the last game: %s" % err
            return False
        if got is None:
            return False
        rules, actions, _, variations = got
        if rules != self.rules or not actions:
            return False                            # other rules this time: start fresh
        game = gamelog.replay(rules, actions)
        over = game.done
        game.close()
        if over:
            return False
        self.open_game(rules, actions, "Resumed the unfinished game from last time (%d moves).  "
                       "R starts a new one." % len(actions), review=False, variations=variations)
        return True

    # -- variations: lines cut off by a different move are kept ---------------
    def stash_line(self, branch):
        """Keep the current line, which a different move at ``branch`` is about to cut off."""
        if len(self.line) <= branch:
            return
        self.keep_line(list(self.line), branch, dict(self.review) if self.review is not None else None)

    def keep_line(self, line, branch, review):
        self.variations = [v for v in self.variations if v["line"] != line]
        self.variations.insert(0, {"line": line, "branch": branch, "review": review})
        del self.variations[MAX_VARIATIONS:]

    def swap_variation(self, step=1):
        """Tab: trade the current line for the most recently kept one (Shift+Tab: the oldest).  The
        current line is kept in its place, so Tab again comes back; the board shows the position
        where the two lines part."""
        if not self.variations:
            self.message = ("No other line yet: play a different move from an earlier position "
                            "and the old line is kept here.")
            return
        v = self.variations.pop(0 if step > 0 else -1)
        fork = 0
        while fork < min(len(self.line), len(v["line"])) and self.line[fork] == v["line"][fork]:
            fork += 1
        self.keep_line(list(self.line), fork, self.review)
        self.stop_search()
        self.hint = None
        self.clear_premove()
        if self.reviewer is not None:
            self.reviewer.stop()
        self.reviewer = None
        self.line = list(v["line"])
        self.texts, self.turns = gamelog.notation_and_turns(self.rules, self.line)
        self.review = v.get("review")
        old = self.game
        self.game = gamelog.replay(self.rules, self.line, fork)
        old.close()
        self.pos = fork
        self.stats_ok = False
        self.moves_top = None
        self.cancel()
        self.reindex()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        self.save_session()
        self.message = ("Switched to the other line, which parts from the old one at move %d.  "
                        "Tab swaps back; %d line(s) kept." % (fork + 1, len(self.variations)))

    def drop_review(self):
        if getattr(self, "reviewer", None) is not None:
            self.reviewer.stop()
        self.reviewer = None
        self.review = None

    def start_review(self):
        if self.advisor is None:
            self.message = "Review needs the AI file (ai.npz or latest.pt next to play.py)."
        elif not self.line:
            self.message = "Nothing to review yet."
        elif self.reviewer is None:
            self.review = None
            self.reviewer = reviewlib.Reviewer(self.eval_advisor or self.advisor, self.rules, self.line, self.review_sims)

    def poll_review(self):
        rv = self.reviewer
        if rv is None or not rv.done:
            return
        self.reviewer = None
        if rv.error:
            self.message = "Review failed: " + rv.error
        elif rv.result is not None and self.line[:len(rv.actions)] == rv.actions:
            self.review = rv.result                 # (the game may have moved on meanwhile: still valid)

    def next_mistake(self, step=1):
        """Jump to the position before the next (step=1) / previous (-1) marked move."""
        if self.review is None:
            self.message = "Press Review first (it runs by itself when a game ends)."
            return
        marks = self.review["marks"]
        order = range(self.pos + 1, len(marks)) if step > 0 else range(self.pos - 1, -1, -1)
        for i in order:
            if marks[i]:
                self.jump(i)
                return
        self.message = "No more marked moves that way."

    def play_network_move(self):
        """At a reviewed position: play the network's choice instead of the move that was played there.
        The old line is kept as a variation (Tab swaps back to it), as with any new move."""
        if self.review is None:
            self.message = "Press Review first (it runs by itself when a game ends)."
            return
        if not self.browsing or self.pos >= len(self.review["best"]):
            self.message = "Step back to a position before a reviewed move first (M jumps to a marked one)."
            return
        best = self.review["best"][self.pos]
        if best == self.line[self.pos]:
            self.message = "That is already the network's move."
            return
        self.play(best)
        if not self.message:
            self.message = "Played the network's move; the old line is kept (Tab swaps back to it)."

    def copy_moves(self):
        """Copy the whole move list (not just up to the position shown) to the clipboard."""
        if not self.line:
            self.message = "No moves to copy yet."
            return
        try:
            clipboard_put(gamelog.move_list_text(self.turns, self.texts))
        except RuntimeError as err:
            self.message = "Could not copy: %s" % err
            return
        self.message = "Copied %d moves to the clipboard." % len(self.line)

    def copy_json(self):
        """Copy the whole game as the JSON that Save writes (see gamelog.py) to the clipboard."""
        if not self.line:
            self.message = "No moves to copy yet."
            return
        try:
            clipboard_put(gamelog.game_json(self.rules, self.line, self.game_meta()))
        except RuntimeError as err:
            self.message = "Could not copy: %s" % err
            return
        self.message = "Copied the game (JSON, %d moves) to the clipboard." % len(self.line)

    def mark_text(self, i):
        """'Move 12 (Rg7-g8, black) is a blunder: lost 0.41.' for the marked move i."""
        mark = self.review_mark(i)
        name = {"??": "a blunder", "?": "a mistake", "?!": "an inaccuracy"}.get(mark, "fine")
        return "Move %d (%s, %s) is %s: lost %.2f." % (
            i + 1, self.texts[i], full_name(self.turns[i]), name, self.review["loss"][i])

    def graph_x(self, i):
        g = self.layout['m_graph']
        n = max(len(self.review["values"]) - 1, 1)
        return g.x + (g.w - 1) * i / n

    def graph_mark_at(self, pos, reach=4):
        """Index of the marked move whose dot is within ``reach`` pixels (sideways) of ``pos``, or None.
        The graph is narrow and a game long, so the nearest dot wins (a bigger loss on a tie)."""
        if self.review is None or not self.layout['m_graph'].collidepoint(pos):
            return None
        best = None
        for i, mark in enumerate(self.review["marks"]):
            if not mark or i >= len(self.texts):
                continue
            key = (abs(self.graph_x(i + 1) - pos[0]), -self.review["loss"][i])
            if key[0] <= reach and (best is None or key < best[0]):
                best = (key, i)
        return None if best is None else best[1]

    def review_here(self):
        """True when the review has the network's view of the position on the board (every position
        before a move of the reviewed line, the opening position included)."""
        rev = self.review
        return (rev is not None and self.browsing and self.pos < len(rev.get("top", []))
                and self.pos < len(rev["best"]))

    def review_alts(self):
        """At a position just before a marked move: the panel's lines for the top three moves there
        (share of the search's visits or the network's probability, and the value after each for the
        side to move) and for the move that was played, or None."""
        rev, n = self.review, self.pos
        if not self.review_here():
            return None
        game, sims = self.game, rev.get("sims", 0)
        rows = []
        for rank, (action, share, value) in enumerate(rev["top"][n][:3]):
            text = "%d. %s  %d%%" % (rank + 1, gamelog.move_text(game, action), round(100 * share))
            if value is not None:
                text += "  %+.2f" % value
            if action == self.line[n]:
                text += "  (played)"
            rows.append(text)
        mover_value = rev["values"][n + 1] if self.turns[n] == "w" else -rev["values"][n + 1]
        source = "search, %d sims; %% = visits" % sims if sims else "network; % = policy"
        return {"header": "Top moves here (%s):" % source, "rows": rows,
                "played": "Played: %s  %+.2f" % (self.texts[n], mover_value)}

    def review_mark(self, i):
        """The review's mark for move ``i`` of the line ('' if none or not reviewed yet)."""
        if self.review is None or not (0 <= i < len(self.review["marks"])):
            return ""
        return self.review["marks"][i]

    def cancel_keep_message(self):
        message = self.message
        self.cancel()
        self.message = message

    def record_finished(self):
        if not self.stats_ok or self.recorded:
            return
        game, human = self.game, self.human_colour()
        result = "drawn" if game.winner is None else "won" if game.winner == human else "lost"
        self.stats.add(human, result, game.ply, self.hint_used, self.advisor.version)
        self.stats_summary = self.stats.summary()
        self.recorded = True
        if self.stats.error:
            self.message = self.stats.error

    # -- panel controls --------------------------------------------------
    def ai_changed(self):
        """The AI setup changed: a game already under way no longer counts as 'vs the AI'."""
        if self.game.ply > 0 and not self.game.done:
            self.stats_ok = False
        elif self.game.ply == 0:
            self.stats_ok = self.stats_eligible()
        self.refresh_agents()
        self.orient()
        self.cancel()
        self.clear_premove()
        self.hint = None
        self.stop_search()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms

    def toggle_ai(self):
        self.ai_on = not self.ai_on
        if self.ai_on and not self.ai_sides:
            self.ai_sides = {'w'}
        self.ai_changed()

    def set_sides(self, sides):
        if self.ai_on and set(sides) != self.ai_sides:
            self.ai_sides = set(sides)
            self.ai_changed()

    def toggle_eval(self):
        if self.advisor is None:
            self.message = "The eval bar needs the AI file (ai.npz or latest.pt next to play.py)."
        else:
            self.show_eval = not self.show_eval
            self.bar_pos = None
            self.message = ""

    def pos_key(self):
        return (id(self.game), self.game.ply)

    # -- search (mcts.py) --------------------------------------------------
    def stop_search(self):
        if self.searcher is not None:
            self.searcher.stop()
        if self.ana_searcher is not None:
            self.ana_searcher.stop()
        self.search_job = self.ana_job = None

    def analysis_searcher(self):
        """The search that feeds the eval bar and the hints: always the main AI's network, whatever level
        plays.  (The look-ahead switch, S / Game > Difficulty, only decides whether the *computer* searches.)"""
        if self.searcher is None or self.eval_advisor is None:
            return None
        if self.advisor is self.eval_advisor:
            return self.searcher                                # the computer is the main AI: one search serves both
        if self.ana_searcher is None or self.ana_searcher.advisor is not self.eval_advisor:
            self.ana_searcher = mcts.Searcher(self.eval_advisor, sims=self.sims_for(self.think_ms),
                                              think_ms=self.think_ms)
        return self.ana_searcher

    def analysis_job_here(self):
        job = self.search_job if self.advisor is self.eval_advisor else self.ana_job
        if job is not None and job.key == self.pos_key() and job.error is None:
            return job
        return None

    def ensure_analysis(self, ms=None):
        """Make sure a background analysis search of the current position is running or finished."""
        searcher = self.analysis_searcher()
        if searcher is None or self.game.done:
            return None
        job = self.analysis_job_here()
        if job is None:
            job = searcher.start(self.game, self.pos_key(), ms=ms)
            if searcher is self.searcher:
                self.search_job = job
            else:
                self.ana_job = job
        return job

    def toggle_search(self):
        if self.searcher is None:
            self.message = "Search needs the AI file (ai.npz or latest.pt next to play.py)."
            return
        self.use_search = not self.use_search
        self.stop_search()
        self.message = ""
        self.ai_due = pygame.time.get_ticks() + self.delay_ms

    def search_for_here(self):
        """The search job for the current position, if there is one."""
        job = self.search_job
        if job is not None and job.key == self.pos_key() and job.error is None:
            return job
        return None

    def ensure_search(self, ms=None):
        """Make sure a background search of the current position is running or finished."""
        if not self.use_search or self.searcher is None or self.game.done:
            return None
        job = self.search_for_here()
        if job is None:
            self.search_job = job = self.searcher.start(self.game, self.pos_key(), ms=ms)
        return job

    def searching(self):
        job = self.search_for_here()
        return job is not None and not job.finished

    def white_value_now(self):
        """(value from white's side, simulations behind it): the search once it has enough
        simulations to mean something, the bare network before that."""
        job = self.analysis_job_here()
        if job is not None and (job.solved or job.sims >= 30):
            v = job.white_value()
            if v is not None:
                return v, job.sims
        return (self.eval_advisor or self.advisor).white_value(self.game), 0

    def current_hint(self):
        if self.hint is not None and self.hint["key"] != self.pos_key():
            self.hint = None                      # the position moved on
        if self.hint is not None:
            job = self.analysis_job_here()
            if job is not None and (job.solved or job.sims >= 50):
                top = job.top(3)
                if top:                           # swap the policy's guess for the search's
                    self.hint["top"] = [(a, share) for a, share, _ in top]
        return self.hint

    def toggle_hint(self):
        if self.current_hint() is not None:
            self.hint = None
            return
        if self.advisor is None:
            self.message = "Hints need the AI file (ai.npz or latest.pt next to play.py)."
        elif self.game.done:
            self.message = ""
        elif not self.human_to_move():
            self.message = "Hints are for your own moves; it is the computer's turn."
        else:
            self.hint = {"key": self.pos_key(), "top": (self.eval_advisor or self.advisor).analyse(self.game).top(3)}
            self.hint_used = True
            self.ensure_analysis(1800)            # the hint sharpens as the search deepens
            self.message = ""

    def click_reset(self):
        if self.stats is None:
            return
        now = pygame.time.get_ticks()
        if now < self.confirm_reset_until:
            self.stats.reset()
            self.stats_summary = self.stats.summary()
            self.confirm_reset_until = 0
        else:
            self.confirm_reset_until = now + 3000

    def click_panel(self, pos):
        """A click right of the board: the network name, or (in review mode) the move-list column."""
        L = self.layout
        if L['net'].collidepoint(pos):
            self.open_picker()
        elif L['m_prev'].collidepoint(pos):
            self.next_mistake(-1)
        elif L['m_next'].collidepoint(pos):
            self.next_mistake(1)
        elif L['m_list'].collidepoint(pos):
            row = self.moves_first_row() + (pos[1] - L['m_list'].y) // ROW_H
            if 0 <= row <= len(self.line):
                self.jump(row)                     # row 0 = the start, row k = the position after move k
        elif L['m_graph'].collidepoint(pos) and self.review is not None:
            i = self.graph_mark_at(pos)
            if i is not None:                      # a marked move: the position before it, with its details
                self.jump(i)
                self.message = self.mark_text(i)
                return
            n = len(self.review["values"]) - 1
            frac = (pos[0] - L['m_graph'].x) / max(L['m_graph'].w - 1, 1)
            self.jump(int(round(frac * n)))

    # -- input -----------------------------------------------------------
    def handle_key(self, event):
        """Dispatch one key event. Modal UI gets first refusal, then global shortcuts, then menu keys."""
        key, mod = event.key, event.mod
        ctrl = bool(mod & pygame.KMOD_CTRL)
        help_key = key == pygame.K_F1 or (key == pygame.K_SLASH and bool(mod & pygame.KMOD_SHIFT))
        if self.menu_open is not None:              # any key closes an open menu; Esc does nothing more
            self.close_menus()
            if key == pygame.K_ESCAPE:
                return

        # The guide is a true modal overlay: no hidden move/navigation actions while it is open.
        if self.help_open:
            if key in (pygame.K_ESCAPE, pygame.K_F1) or help_key:
                self.help_open = False
            return
        if help_key:
            self.help_open = True
            return
        if self.picker is not None:
            self.picker_key(key)
            return
        if key == pygame.K_ESCAPE:
            self.escape()
            return
        if ctrl and key == pygame.K_s and self.pending is None:
            self.save()
            return
        if ctrl and key == pygame.K_o and self.pending is None:
            self.load()
            return
        # Keep the long-standing restart shortcut global, including while a move menu is open.
        if key == pygame.K_r and not ctrl:
            self.restart()
            return

        if self.pending is not None:
            self.menu_key(key)
            return

        if ctrl and key == pygame.K_LEFT:
            self.next_mistake(-1)
        elif ctrl and key == pygame.K_RIGHT:
            self.next_mistake(1)
        elif ctrl and key == pygame.K_c:
            self.copy_json() if mod & pygame.KMOD_SHIFT else self.copy_moves()
        elif ctrl and key == pygame.K_v:
            self.paste()
        elif ctrl and key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.zoom(1.1)
        elif ctrl and key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom(1 / 1.1)
        elif ctrl and key in (pygame.K_0, pygame.K_KP0):
            self.fit_screen()
        elif ctrl and key in (pygame.K_1, pygame.K_KP1):
            self.apply_scale(1.0)
        elif key == pygame.K_TAB:
            self.swap_variation(-1 if mod & pygame.KMOD_SHIFT else 1)
        elif key == pygame.K_LEFT:
            self.step_back()
        elif key == pygame.K_RIGHT:
            self.step_forward()
        elif key == pygame.K_HOME:
            self.jump(0)
        elif key == pygame.K_END:
            self.jump(len(self.line))
        elif key in (pygame.K_z, pygame.K_BACKSPACE):
            self.takeback()
        elif key == pygame.K_f:
            self.flip()
        elif key == pygame.K_c and not ctrl:
            self.toggle_coords()
        elif key == pygame.K_n and not ctrl:
            self.open_picker()
        elif key == pygame.K_u and not ctrl:
            self.toggle_sound()
        elif key == pygame.K_m:
            self.next_mistake(-1 if mod & pygame.KMOD_SHIFT else 1)
        elif key == pygame.K_v:
            self.start_review()
        elif key == pygame.K_p and not ctrl:
            self.play_network_move()
        elif ctrl:
            return
        elif key == pygame.K_h:
            self.toggle_hint()
        elif key == pygame.K_e:
            self.toggle_eval()
        elif key == pygame.K_a:
            self.toggle_ai()
        elif key == pygame.K_s:
            self.toggle_search()

    def human_to_move(self):
        return not self.game.done and self.agents[self.game.turn] is None

    def handle_click(self, pos, force_menu=False, raw=None):
        """A left click at canvas pixel ``pos`` (``raw``: the same click in window pixels, which is where the
        menu bar is; default: the same numbers).  ``force_menu`` (Shift held): ask even about a move that
        has only one variant, with a preview of it, before playing."""
        if self.help_open:
            self.help_open = False
            return
        if self.picker is not None:               # so is the network picker
            self.click_picker(pos)
            return
        if self.pending is not None:              # the move menu is modal
            self.click_menu(pos)
            return
        if self.menu_click(pos if raw is None else raw):   # the menu bar and its drop-downs
            return
        if pos[1] >= WIN_H:                       # the tall layout's control strip
            if self.strip['net'].collidepoint(pos):
                self.open_picker()
            return
        if pos[0] >= BOARD_PX:
            self.click_panel(pos)
            return
        if not self.human_to_move():
            self.premove_click(pos)               # the computer's turn: queue a move for afterwards
            return
        x, y = pos
        if not (0 <= x < BOARD_PX and BOARD_Y <= y < BOARD_Y + BOARD_PX):
            return
        sq = self.xy_sq(x, y)
        game = self.game
        piece = game.board[sq]
        if piece is not None and piece.team == game.turn:
            if sq == self.selected:       # click again to deselect
                self.cancel()
            elif sq in self.by_src:
                self.selected = sq
                self.message = ""
            else:                         # select it anyway so clicks can explain why
                self.selected = sq
                self.message = (game.lock_reason(piece) if piece.locked else
                                "This piece has no legal moves right now.  Click any "
                                "square to see why that move is illegal.")
            return
        if self.selected is None:
            self.message = "Click one of your pieces first (blue outline = it can move)."
            return
        if sq in {a.dst for a in self.by_src.get(self.selected, [])}:
            self.candidates = [a for a in self.by_src[self.selected] if a.dst == sq]
            self.history = []
            self.resolve(force_menu)
        else:
            self.message = (game.explain_illegal(self.selected, sq)
                            or "That move is not available.")

    def resolve(self, force_menu=False):
        """Play the move, or open a menu for the next undecided choice.  A move with a single variant
        is played at once (nothing to decide); ``force_menu`` shows it first, to confirm."""
        choice = next_choice(self.candidates)
        self.menu_sel = None
        if choice is not None:
            self.pending = choice
        elif force_menu and len(self.candidates) == 1:
            self.pending = ('confirm', [True])
        else:
            self.play(self.candidates[0])

    def choose(self, index):
        """Answer the open menu with its ``index``-th option."""
        dim, values = self.pending
        self.menu_sel = None
        if dim == 'confirm':
            action = self.candidates[0]
            self.pending = None
            self.play(action)
            return
        value = values[index]
        self.history.append(self.candidates)
        self.candidates = [a for a in self.candidates if getattr(a, dim) == value]
        self.pending = None
        self.resolve()

    def click_menu(self, pos):
        dim, values = self.pending
        panel, rects, x_rect, back_rect = self.menu_geometry(len(values), bool(self.history))
        if x_rect.collidepoint(pos):
            self.close_menu()
        elif back_rect is not None and back_rect.collidepoint(pos):
            self.menu_back()
        else:
            for i, rect in enumerate(rects):
                if rect.collidepoint(pos):
                    self.choose(i)
                    return
            if not panel.collidepoint(pos):   # a click away from the menu closes it
                self.close_menu()

    def update_ai(self):
        game = self.game
        if game.done:
            return
        agent = self.agents[game.turn]
        if agent is None:                         # a human's turn: think ahead for the eval bar / hint
            if self.premove is not None and not self.browsing and self.pending is None:
                self.run_premove()
                return
            if self.show_eval or self.current_hint() is not None:
                self.ensure_analysis(1800)
            return
        if self.browsing:                         # looking at an earlier position: the computer waits
            return
        if self.use_search and agent is self.default_ai:
            job = self.ensure_search()            # the network, with look-ahead
            if job is None or not job.finished or pygame.time.get_ticks() < self.ai_due:
                return
            if job.error is not None:
                self.message = "search failed (%s); using the network alone" % job.error
                self.search_job = None
                self.use_search = False
                return
            action = job.best()
            if action is None or action not in game.actions:
                action = agent(game)
        else:
            if pygame.time.get_ticks() < self.ai_due:
                return
            action = agent(game)
        if action not in game.actions:
            self.message = "agent returned an illegal action; playing a random one"
            action = self.rng.choice(game.legal_actions())
        self.play(action)

    # -- the menu bar -----------------------------------------------------------
    def close_menus(self):
        self.menu_open = None
        self.menu_sub = None

    def quit(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))

    def toggle_help(self):
        self.help_open = not self.help_open

    def set_moves_mode(self, mode):
        self.moves_mode = mode

    def set_ai_sides(self, sides):
        """Game > Computer plays: ``None`` = nobody (two humans), else the set of teams the AI plays."""
        if sides is None:
            if self.ai_on:
                self.toggle_ai()
        elif not self.ai_on or set(sides) != self.ai_sides:
            self.ai_on = True
            self.ai_sides = set(sides)
            self.ai_changed()

    # -- remembered settings (settings.json next to the win/loss record) ---------------------------
    def settings_snapshot(self):
        """The choices worth keeping between sessions."""
        tier = next((t["name"] for t in self.tiers if self.tier_current(t)), None)
        net = None
        if tier is None and self.advisor is not None and self.home_advisor is not None:
            path = getattr(self.advisor, "path", None)
            if path and os.path.realpath(path) != os.path.realpath(getattr(self.home_advisor, "path", "") or ""):
                net = path                                  # a network picked with N
        return {"tier": tier, "net": net, "sound": self.sound_on, "coords": self.coords,
                "eval": bool(self.show_eval), "search": bool(self.use_search), "think_ms": int(self.think_ms),
                "moves_mode": self.moves_mode, "ai_on": bool(self.ai_on), "ai_sides": sorted(self.ai_sides),
                "scale": round(float(self.scale), 3)}

    def save_settings(self, force=False):
        """Write the settings when they changed (checked about once a second; ``force`` at exit)."""
        path = self.settings_path
        if not path:
            return
        now = time.time()
        if not force and now < self.settings_next:
            return
        self.settings_next = now + 1.0
        snap = self.settings_snapshot()
        if snap == self.settings_saved:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2)
            os.replace(tmp, path)
            self.settings_saved = snap
        except OSError:
            self.settings_path = None                       # read-only home or the like: give up quietly

    def load_settings(self, given):
        """Apply the saved settings, except for the ones a command-line option has already decided (``given``)."""
        path = self.settings_path
        try:
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            assert isinstance(saved, dict)
        except Exception:
            return
        if "net" not in given and "tier" not in given:
            tier = next((t for t in self.tiers if t["name"] == saved.get("tier")), None)
            if tier is not None:
                self.choose_tier(tier)
            elif saved.get("net") and os.path.isfile(saved["net"]):
                try:
                    self.set_advisor(numpy_net.Advisor(saved["net"], self.advisor.max_plies if self.advisor else 300))
                except Exception:
                    pass
        if "sound" not in given and isinstance(saved.get("sound"), bool):
            self.sound_on = saved["sound"]
        if "coords" not in given and isinstance(saved.get("coords"), bool):
            self.coords = saved["coords"]
        if "eval" not in given and saved.get("eval") and self.advisor is not None:
            self.show_eval = True
        if "search" not in given and isinstance(saved.get("search"), bool) and self.searcher is not None:
            self.use_search = saved["search"]
        if "think" not in given and isinstance(saved.get("think_ms"), int) and self.searcher is not None:
            self.set_think(max(100, saved["think_ms"]))
        if saved.get("moves_mode") in ("auto", "on", "off"):
            self.moves_mode = saved["moves_mode"]
        if "ai" not in given and isinstance(saved.get("ai_on"), bool):
            sides = {x for x in saved.get("ai_sides", []) if x in ("b", "w")}
            self.ai_on, self.ai_sides = saved["ai_on"] and bool(sides), (sides or self.ai_sides)
            self.ai_changed()
        if "scale" not in given and isinstance(saved.get("scale"), (int, float)) and 0.4 <= saved["scale"] <= 4:
            try:
                dw, dh = pygame.display.get_desktop_sizes()[0]
            except Exception:
                dw, dh = 10 ** 6, 10 ** 6
            cw, ch = self.content_size()
            self.apply_scale(min(saved["scale"], (dh - 90 - MENU_H) / float(ch), (dw - 30) / float(cw)))
        self.settings_saved = self.settings_snapshot()

    def choose_tier(self, tier):
        """Game > Difficulty: load that level's network (and switch look-ahead search the way it was calibrated)."""
        try:
            advisor = numpy_net.Advisor(tier["path"], self.advisor.max_plies if self.advisor is not None else 300)
        except Exception as err:
            self.message = "Could not load %s: %s" % (os.path.basename(tier["path"]), err)
            return
        self.set_advisor(advisor, tier=True)
        self.use_search = bool(tier["search"]) and self.searcher is not None
        self.message = "Difficulty: %s." % tier["name"]

    def tier_current(self, tier):
        return (self.advisor is not None and
                os.path.realpath(getattr(self.advisor, "path", "") or "") == os.path.realpath(tier["path"]))

    def menu_items(self, index):
        """The entries of drop-down ``index``, built fresh (their labels and ticks follow the game's state)."""
        it = menu_item
        title = MENU_TITLES[index]
        have_ai = self.advisor is not None
        if title == "File":
            return [it("New game", self.restart, "R"), None,
                    it("Save game", self.save, "Ctrl+S", enabled=bool(self.line)),
                    it("Load game", self.load, "Ctrl+O"), None,
                    it("Copy moves", self.copy_moves, "Ctrl+C", enabled=bool(self.line)),
                    it("Copy game (JSON)", self.copy_json, "Ctrl+Shift+C", enabled=bool(self.line)),
                    it("Paste game", self.paste, "Ctrl+V"), None,
                    it("Quit", self.quit)]
        if title == "Game":
            ai_on, sides = self.ai_on, self.ai_sides
            items = [it("Undo", self.takeback, "Z", enabled=self.pos > 0),
                     it("Flip board", self.flip, "F", checked=self.flipped), None,
                     it("Computer plays", sub=[
                         it("Nobody (two humans)", lambda: self.set_ai_sides(None), checked=not ai_on),
                         it("Black", lambda: self.set_ai_sides({'b'}), checked=ai_on and sides == {'b'}),
                         it("White", lambda: self.set_ai_sides({'w'}), checked=ai_on and sides == {'w'}),
                         it("Both", lambda: self.set_ai_sides({'b', 'w'}), checked=ai_on and sides == {'b', 'w'})]),
                     it("Network...", self.open_picker, "N")]
            if self.tiers:
                items.append(it("Difficulty", sub=[
                    it("%s%s" % (t["name"], "  (loses %d%%)" % round(100 * t["loss_rate"])
                                 if isinstance(t.get("loss_rate"), (int, float)) and t["loss_rate"] > 0.5 else ""),
                       lambda t=t: self.choose_tier(t), checked=self.tier_current(t)) for t in self.tiers]))
            items.append(it("Think time", enabled=self.think_enabled(), sub=[
                it(think_label(ms), lambda ms=ms: self.set_think(ms), checked=self.think_ms == ms)
                for ms in THINK_STEPS]))
            items.append(None)
            confirming = pygame.time.get_ticks() < self.confirm_reset_until
            items.append(it("Click again to erase the record" if confirming else "Reset win / loss record",
                            self.click_reset, enabled=self.stats is not None and bool(self.stats.games), keep=True))
            return items
        if title == "Analysis":
            has_marks = self.review is not None and any(self.review["marks"])
            review_label = ("Reviewing... %d%%" % int(100 * self.reviewer.progress) if self.reviewer is not None
                            else "Review game")
            return [it("Computer uses look-ahead search (MCTS)", self.toggle_search, "S", checked=self.use_search,
                       enabled=self.searcher is not None),
                    it("Eval bar", self.toggle_eval, "E", checked=self.show_eval, enabled=have_ai),
                    it("Hint", self.toggle_hint, "H", checked=self.current_hint() is not None, enabled=have_ai),
                    None,
                    it(review_label, self.start_review, "V",
                       enabled=have_ai and bool(self.line) and self.reviewer is None),
                    it("Next marked move", lambda: self.next_mistake(1), "M", enabled=has_marks),
                    it("Previous marked move", lambda: self.next_mistake(-1), "Shift+M", enabled=has_marks),
                    it("Play the network's move", self.play_network_move, "P", enabled=have_ai)]
        if title == "View":
            return [it("Coordinates", self.toggle_coords, "C", checked=self.coords),
                    it("Sound", self.toggle_sound, "U", checked=self.sound_on), None,
                    it("Move list", sub=[
                        it("Automatic (review mode)", lambda: self.set_moves_mode('auto'), checked=self.moves_mode == 'auto'),
                        it("Always show", lambda: self.set_moves_mode('on'), checked=self.moves_mode == 'on'),
                        it("Never show", lambda: self.set_moves_mode('off'), checked=self.moves_mode == 'off')]),
                    None,
                    it("Zoom in", lambda: self.zoom(1.1), "Ctrl++"),
                    it("Zoom out", lambda: self.zoom(1 / 1.1), "Ctrl+-"),
                    it("Fit to screen", self.fit_screen, "Ctrl+0"),
                    it("Actual size", lambda: self.apply_scale(1.0), "Ctrl+1")]
        return [it("Quick reference", self.toggle_help, "F1")]

    def menu_title_rects(self):
        rects, x = [], 6
        for title in MENU_TITLES:
            w = self.font_btn.size(title)[0] + 26
            rects.append(pygame.Rect(x, 0, w, MENU_H))
            x += w
        return rects

    def dropdown_rects(self, items, x, y):
        """(panel, [(item or None, row rect)]) of a drop-down whose top-left corner is (x, y)."""
        label_w = max([self.font_btn.size(i["label"])[0] for i in items if i] or [60])
        key_w = max([self.font_small.size(i["key"])[0] for i in items if i and i["key"]] or [0])
        has_sub = any(i and i["sub"] for i in items)
        width = 24 + label_w + (26 + key_w if key_w else 0) + (20 if has_sub else 0) + 12
        rows, cy = [], y + 4
        for item in items:
            h = 9 if item is None else 26
            rows.append((item, pygame.Rect(x + 2, cy, width - 4, h)))
            cy += h
        return pygame.Rect(x, y, width, cy - y + 4), rows

    def menu_layout(self):
        """(panel, rows, sub_panel, sub_rows) of the open drop-down and of its open sub-menu (or None, None)."""
        titles = self.menu_title_rects()
        items = self.menu_items(self.menu_open)
        panel, rows = self.dropdown_rects(items, titles[self.menu_open].x, MENU_H)
        sub_panel = sub_rows = None
        if self.menu_sub is not None:
            for item, rect in rows:
                if item is not None and item["sub"] and item["label"] == self.menu_sub:
                    sub_panel, sub_rows = self.dropdown_rects(item["sub"], panel.right - 2, rect.y - 4)
                    if sub_panel.right > self.window.get_width():       # no room on the right: open leftwards
                        sub_panel, sub_rows = self.dropdown_rects(item["sub"], panel.x - sub_panel.w + 2, rect.y - 4)
                    break
        return panel, rows, sub_panel, sub_rows

    def menu_click(self, pos):
        """A click on the menu bar or in a drop-down.  True when it was used up there (a click anywhere else
        while a menu is open only closes the menu)."""
        for i, rect in enumerate(self.menu_title_rects()):
            if rect.collidepoint(pos):
                if self.menu_open == i:
                    self.close_menus()
                else:
                    self.menu_open, self.menu_sub = i, None
                return True
        if self.menu_open is None:
            return pos[1] < MENU_H
        panel, rows, sub_panel, sub_rows = self.menu_layout()
        for item, rect in list(sub_rows or []) + rows:           # the sub-menu sits on top of the main one
            if item is None or not rect.collidepoint(pos):
                continue
            if not item["enabled"]:
                return True
            if item["sub"]:
                self.menu_sub = None if self.menu_sub == item["label"] else item["label"]
                return True
            if not item["keep"]:
                self.close_menus()
            item["run"]()
            return True
        if panel.collidepoint(pos) or (sub_panel is not None and sub_panel.collidepoint(pos)):
            return True                                          # on a separator / the border
        self.close_menus()
        return True

    def menu_hover(self):
        """While a menu is open: moving over another title switches menus, moving over an entry with a
        sub-menu opens it (called once per frame)."""
        if self.menu_open is None or self.pending is not None:
            return
        pos = self.raw_mouse()
        for i, rect in enumerate(self.menu_title_rects()):
            if rect.collidepoint(pos) and i != self.menu_open:
                self.menu_open, self.menu_sub = i, None
                return
        panel, rows, sub_panel, sub_rows = self.menu_layout()
        if sub_panel is not None and sub_panel.collidepoint(pos):
            return
        for item, rect in rows:
            if item is not None and rect.collidepoint(pos):
                self.menu_sub = item["label"] if item["sub"] and item["enabled"] else None
                return

    def draw_menubar(self, surf):
        """The menu bar, on the window itself (window pixels, full width)."""
        width = surf.get_width()
        pygame.draw.rect(surf, (28, 28, 32), (0, 0, width, MENU_H))
        pygame.draw.line(surf, (14, 14, 18), (0, MENU_H - 1), (width, MENU_H - 1))
        mouse = self.raw_mouse()
        for i, rect in enumerate(self.menu_title_rects()):
            if self.menu_open == i:
                pygame.draw.rect(surf, BUTTON_ON, rect)
            elif rect.collidepoint(mouse) and self.pending is None:
                pygame.draw.rect(surf, BUTTON_HOVER, rect)
            label = self.font_btn.render(MENU_TITLES[i], True, TEXT)
            surf.blit(label, label.get_rect(center=rect.center))

    def draw_dropdown(self, surf):
        if self.menu_open is None or self.help_open or self.picker is not None or self.pending is not None:
            return
        panel, rows, sub_panel, sub_rows = self.menu_layout()
        mouse = self.raw_mouse()
        for pan, rws in ((panel, rows), (sub_panel, sub_rows)):
            if pan is None:
                continue
            pygame.draw.rect(surf, (44, 44, 50), pan, border_radius=6)
            pygame.draw.rect(surf, (96, 96, 108), pan, 1, border_radius=6)
            for item, rect in rws:
                if item is None:
                    pygame.draw.line(surf, (84, 84, 94), (rect.x + 8, rect.centery), (rect.right - 8, rect.centery))
                    continue
                if item["enabled"] and (rect.collidepoint(mouse) or (item["sub"] and item["label"] == self.menu_sub)):
                    pygame.draw.rect(surf, BUTTON_ON, rect, border_radius=4)
                colour = TEXT if item["enabled"] else MUTED
                if item["checked"]:
                    x, y = rect.x + 6, rect.centery
                    pygame.draw.lines(surf, colour, False, [(x, y), (x + 4, y + 4), (x + 11, y - 5)], 2)
                surf.blit(self.font_btn.render(item["label"], True, colour), (rect.x + 22, rect.y + 5))
                if item["key"]:
                    key = self.font_small.render(item["key"], True, MUTED)
                    surf.blit(key, (rect.right - 10 - key.get_width(), rect.y + 7))
                if item["sub"]:
                    cx, cy = rect.right - 12, rect.centery
                    pygame.draw.polygon(surf, colour, [(cx - 3, cy - 5), (cx - 3, cy + 5), (cx + 3, cy)])

    # -- drawing ---------------------------------------------------------
    def sq_xy(self, r, c):
        """Top-left pixel of square (r, c), honouring the flip."""
        if self.flipped:
            return ((SIZE - 1 - c) * SQ, BOARD_Y + (SIZE - 1 - r) * SQ)
        return (c * SQ, BOARD_Y + r * SQ)

    def board_overlay(self):
        """A transparent surface covering the canvas down to the bottom of the board (draw on it with the
        square coordinates of sq_xy / sq_centre, then blit it at (0, 0))."""
        return pygame.Surface((BOARD_PX, BOARD_Y + BOARD_PX), pygame.SRCALPHA)

    def xy_sq(self, x, y):
        r, c = (y - BOARD_Y) // SQ, x // SQ
        return (SIZE - 1 - r, SIZE - 1 - c) if self.flipped else (r, c)

    def sq_centre(self, sq):
        x, y = self.sq_xy(*sq)
        return (x + SQ // 2, y + SQ // 2)

    def sprite(self, team, ptype, size):
        key = (team, ptype, size)
        if key not in self._sprites:
            img = pygame.image.load(image_path(team, ptype)).convert_alpha()
            self._sprites[key] = pygame.transform.smoothscale(img, (size, size))
        return self._sprites[key]

    def text(self, string, pos, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(string, True, color), pos)

    def draw(self):
        self.sync_window()
        self.screen.fill(STATUS_BG)
        self.draw_board()
        self.draw_coords()
        self.draw_hint_arrows()
        self.draw_review_arrows()
        self.draw_premove()
        self.draw_eval_bar()
        self.draw_panel()
        if self.tall:
            self.draw_strip()
        self.draw_moves()
        self.draw_status()
        if self.pending is not None:
            self.draw_menu()
            self.draw_menu_preview()
        if self.picker is not None:
            self.draw_picker()
        self.draw_tooltip()
        if self.help_open:
            self.draw_help()

    def draw_help(self):
        """Draw the modal quick reference above the game without disturbing its state."""
        veil = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 205))
        self.screen.blit(veil, (0, 0))

        panel = pygame.Rect(60, 48, self.win_w - 120, self.win_h - 96)
        pygame.draw.rect(self.screen, (31, 32, 38), panel, border_radius=14)
        pygame.draw.rect(self.screen, (105, 110, 125), panel, 2, border_radius=14)
        self.text("Reverse Chess — quick reference", (panel.x + 26, panel.y + 18), self.font_big)
        self.text("F1 or ? closes this guide too. The game stays paused behind it.",
                  (panel.x + 28, panel.y + 54), self.font_small, MUTED)

        gutter = 36
        col_w = (panel.w - 3 * gutter) // 2
        left_x = panel.x + gutter
        right_x = left_x + col_w + gutter
        body_y = panel.y + 98

        left = [
            ("PLAYING", [
                "Click a blue-outlined piece, then a dotted destination.",
                "Click an unavailable square to see why that move is illegal.",
                "Shift-click a destination to confirm even a one-option move.",
                "Move menus: 1–9 or Up/Down + Enter; Backspace goes back. Esc closes the menu.",
                "During the computer's turn, click your piece then a destination to queue a premove. Right-click cancels.",
            ]),
            ("HISTORY & REVIEW", [
                "Left/Right: step through the line. Home/End: start/end. Click a move row or graph point to jump.",
                "Z or Backspace: take back your last move (and the AI reply). A new move keeps the old line as a variation.",
                "Tab / Shift+Tab: switch between saved variations. F flips the board.",
                "V starts position review; M / Next ? go to the next marked move, Shift+M goes back. P plays the network's move.",
            ]),
        ]
        right = [
            ("AI & ANALYSIS", [
                "A toggles the AI. Game > Computer plays chooses which side it plays; Game > Difficulty picks a level (when tiers.py has made some).",
                "H toggles move hints; E toggles the evaluation bar; S toggles MCTS look-ahead.",
                "Game > Think time sets the search time and simulation budget. The AI needs a loaded network for analysis.",
                "Click the network name or press N to choose another .npz / .pt file.",
            ]),
            ("SAVE, SHARE & DISPLAY", [
                "Ctrl+S saves; Ctrl+O loads. Repeated Load cycles through saved games. Drop a .json file onto the window to open it.",
                "Ctrl+C copies the move list; Ctrl+Shift+C copies game JSON; Ctrl+V pastes either format.",
                "Ctrl++ / Ctrl+- resize; Ctrl+0 fits the screen; Ctrl+1 sets 100%. C toggles coordinates; U toggles sounds. View > Move list: the move column shows by itself in review mode.",
                "The side panel shows home pieces and off-board pieces that can be returned as gifts. Grey locks are permanent; orange locks are temporary.",
            ]),
        ]

        def section(x, y, items):
            for heading, entries in items:
                self.text(heading, (x, y), self.font_status, GOOD)
                y += 31
                for entry in entries:
                    lines = wrap(entry, self.font_small, col_w - 22)
                    self.text("•", (x + 1, y), self.font_small, TEXT)
                    for line in lines:
                        self.text(line, (x + 18, y), self.font_small, TEXT)
                        y += 21
                    y += 5
                y += 13
            return y

        section(left_x, body_y, left)
        section(right_x, body_y, right)
        self.text("Press Esc to close this guide. Nothing is changed until you interact with the board again.",
                  (panel.x + 28, panel.bottom - 29), self.font_small, MUTED)

    def draw_coords(self):
        """Rank numbers down the left edge and file letters along the bottom, whichever way the board is
        turned: each label sits in its square's corner, in the other square colour."""
        if not self.coords:
            return
        for i in range(SIZE):
            r, c = self.xy_sq(2, BOARD_Y + i * SQ + 2)        # screen row i, left-hand column
            x, y = self.sq_xy(r, c)
            label = self.font_small.render(str(SIZE - r), True, DARK if (r + c) % 2 == 0 else LIGHT)
            self.screen.blit(label, (x + 4, y + 3))
            r, c = self.xy_sq(i * SQ + 2, BOARD_Y + (SIZE - 1) * SQ + 2)  # screen column i, bottom row
            x, y = self.sq_xy(r, c)
            label = self.font_small.render("abcdefgh"[c], True, DARK if (r + c) % 2 == 0 else LIGHT)
            self.screen.blit(label, (x + SQ - label.get_width() - 5, y + SQ - label.get_height() - 3))

    def draw_premove(self):
        """The queued premove as a blue arrow; while only its piece is picked, that square is ringed."""
        if self.premove is None and self.pre_src is None:
            return
        overlay = self.board_overlay()
        if self.premove is not None:
            src, dst = self.premove
            draw_arrow(overlay, self.sq_centre(src), self.sq_centre(dst), PREMOVE_COLOUR, 12)
        else:
            x, y = self.sq_xy(*self.pre_src)
            pygame.draw.rect(overlay, PREMOVE_COLOUR, (x, y, SQ, SQ), 6)
        self.screen.blit(overlay, (0, 0))

    def draw_menu_preview(self):
        """While a move menu is open: the highlighted option's gift, ghosted on the board where it lands
        (the menu is beside the board), so the choice can be judged before it is made."""
        action = self.menu_preview_action()
        if action is None or action.gift is None:
            return
        overlay = self.board_overlay()
        self.draw_gift_ghost(overlay, action, HINT_COLOURS[0])
        self.screen.blit(overlay, (0, 0))

    def draw_picker(self):
        pk = self.picker
        veil = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 150))
        self.screen.blit(veil, (0, BOARD_Y))
        panel, rows, x_rect = self.picker_rects()
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=12)
        self.text("Choose the AI's network", (panel.x + 20, panel.y + 14), self.font_big)
        self.draw_button(x_rect, "X")
        mouse = self.mouse()
        for i, rect in enumerate(rows):
            index = pk["top"] + i
            item = pk["items"][index]
            current = item["real"] == pk["cur"]
            hot = rect.collidepoint(mouse) or index == pk["sel"]
            pygame.draw.rect(self.screen, BUTTON_ON if current else BUTTON_HOVER if hot else BUTTON, rect, border_radius=6)
            if hot and current:
                pygame.draw.rect(self.screen, TEXT, rect, 1, border_radius=6)
            name = ("v%d   " % item["version"] if item["version"] is not None else "") + item["rel"]
            date = time.strftime("%Y-%m-%d", time.localtime(item["mtime"])) if item["mtime"] else ""
            self.screen.set_clip(pygame.Rect(rect.x, rect.y, rect.w - 110, rect.h))
            self.text(name + ("   (in use)" if current else ""), (rect.x + 10, rect.y + 6), self.font_btn)
            self.screen.set_clip(None)
            self.text(date, (rect.right - 100, rect.y + 7), self.font_small, MUTED)
        n = len(pk["items"])
        shown = len(rows)
        self.text("%d-%d of %d   click or Enter loads it, Esc closes, wheel / arrows scroll"
                  % (pk["top"] + 1, pk["top"] + shown, n), (panel.x + 20, panel.bottom - 24), self.font_small, MUTED)

    def draw_tooltip(self):
        """The text a hovered graph mark / move-list chip asked for while the frame was drawn."""
        tip, self.tooltip = self.tooltip, None
        if tip is None or self.pending is not None:
            return
        text, colour = tip
        surf = self.font_small.render(text, True, colour)
        x, y = self.mouse()
        rect = surf.get_rect().inflate(14, 8)
        rect.topright = (x - 10, y + 14)            # left of the pointer: the graph is at the window's right edge
        rect.clamp_ip(pygame.Rect(0, MENU_H, self.win_w, self.win_h - MENU_H))
        pygame.draw.rect(self.screen, (16, 16, 20), rect, border_radius=5)
        pygame.draw.rect(self.screen, (120, 120, 128), rect, 1, border_radius=5)
        self.screen.blit(surf, (rect.x + 7, rect.y + 4))

    def draw_board(self):
        game = self.game
        last = game.last_move
        dests = {a.dst for a in self.by_src.get(self.selected, [])}
        movable = self.human_to_move()
        for r in range(SIZE):
            for c in range(SIZE):
                x, y = self.sq_xy(r, c)
                rect = pygame.Rect(x, y, SQ, SQ)
                colour = LIGHT if (r + c) % 2 == 0 else DARK
                if last is not None and (r, c) in (last.src, last.dst):
                    colour = LAST
                if (r, c) == self.selected:
                    colour = SELECT
                pygame.draw.rect(self.screen, colour, rect)
                piece = game.board[r, c]
                if piece is not None:
                    self.screen.blit(self.sprite(piece.team, piece.ptype, SQ - 12), (x + 6, y + 6))
                    if piece.promoted:   # a pawn wearing another piece's type
                        centre = (x + SQ - 18, y + 18)
                        pygame.draw.circle(self.screen, (230, 190, 40), centre, 12)
                        label = self.font_small.render("P", True, (20, 20, 20))
                        self.screen.blit(label, label.get_rect(center=centre))
                    if piece.locked:     # can never move again
                        draw_padlock(self.screen, x + 6, y + SQ - 26, LOCKED)
                    elif (r, c) in self.blocked:   # cannot move right now
                        draw_padlock(self.screen, x + 6, y + SQ - 26, BLOCKED)
                if movable and (r, c) in self.by_src:
                    pygame.draw.rect(self.screen, MOVABLE, rect, 4)
                if (r, c) in dests:
                    pygame.draw.circle(self.screen, DOT, rect.center, 11)

    def draw_hint_arrows(self):
        hint = self.current_hint()
        if hint is None:
            return
        overlay = self.board_overlay()
        seen = set()
        for rank, (action, _) in enumerate(hint["top"]):
            if (action.src, action.dst) in seen:
                continue
            seen.add((action.src, action.dst))
            draw_arrow(overlay, self.sq_centre(action.src), self.sq_centre(action.dst),
                       HINT_COLOURS[rank], HINT_WIDTHS[rank])
        if hint["top"]:
            self.draw_gift_ghost(overlay, hint["top"][0][0], HINT_COLOURS[0])
        self.screen.blit(overlay, (0, 0))

    def draw_review_arrows(self):
        """At a position right before a marked move: the network's choice (green) and the move played (red)."""
        if not self.review_here() or self.pending is not None:
            return
        best, played = self.review["best"][self.pos], self.line[self.pos]
        overlay = self.board_overlay()
        marked = bool(self.review_mark(self.pos))
        if played == best and not marked:
            draw_arrow(overlay, self.sq_centre(best.src), self.sq_centre(best.dst), REVIEW_BEST, 13)
            self.draw_gift_ghost(overlay, best, REVIEW_BEST)
            self.screen.blit(overlay, (0, 0))
            return
        played_colour = REVIEW_PLAYED if marked else (170, 170, 185, 150)   # not a mistake: just "what was played"
        for action, colour, width in ((played, played_colour, 9), (best, REVIEW_BEST, 13)):
            draw_arrow(overlay, self.sq_centre(action.src), self.sq_centre(action.dst), colour, width)
        shared = (played.gift is not None and best.gift is not None
                  and self.gift_square(played) == self.gift_square(best))
        self.draw_gift_ghost(overlay, played, played_colour, small=shared)   # small: don't hide it under the green one
        self.draw_gift_ghost(overlay, best, REVIEW_BEST)
        self.screen.blit(overlay, (0, 0))

    def gift_square(self, action):
        """Where the gift of ``action`` lands: the vacated source square, or the pawn home row for
        reverse en passant."""
        if action.en_passant:
            return (pawn_home_row(other_team(self.game.turn)), action.src[1])
        return action.src

    def draw_gift_ghost(self, overlay, action, colour, small=False):
        """Show the gift of ``action`` as a translucent piece on the square it would land on, ringed
        in ``colour`` (``small``: a quarter-size piece in the corner instead)."""
        if action.gift is None:
            return
        recv = other_team(self.game.turn)
        x, y = self.sq_xy(*self.gift_square(action))
        ptype = action.promote or action.gift
        size = (SQ - 12) // 2 if small else SQ - 12
        ghost = self.sprite(recv, ptype, size).copy()
        ghost.set_alpha(150)
        overlay.blit(ghost, (x + SQ - size - 4, y + SQ - size - 4) if small else (x + 6, y + 6))
        pygame.draw.rect(overlay, colour, (x + 2, y + 2, SQ - 4, SQ - 4) if small else (x, y, SQ, SQ), 4 if not small else 3)
        if action.promote:                    # a gifted pawn wearing another piece's type
            centre = (x + SQ - 18, y + 18)
            pygame.draw.circle(overlay, (230, 190, 40), centre, 12)
            label = self.font_small.render("P", True, (20, 20, 20))
            overlay.blit(label, label.get_rect(center=centre))

    def eval_target(self):
        """Where the bar should sit: 1 = all white, 0 = all black."""
        game = self.game
        if game.done:
            return 0.5 if game.winner is None else 1.0 if game.winner == 'w' else 0.0
        return (max(-1.0, min(1.0, self.white_value_now()[0])) + 1.0) / 2.0

    def draw_eval_bar(self):
        pygame.draw.rect(self.screen, PANEL_BG, (BOARD_PX, BOARD_Y, BAR_W, BOARD_PX))
        if not self.show_eval or self.advisor is None:
            return
        target = self.eval_target()
        self.bar_pos = target if self.bar_pos is None else self.bar_pos + (target - self.bar_pos) * 0.25
        bar = pygame.Rect(BOARD_PX + 7, BOARD_Y + 8, BAR_W - 14, BOARD_PX - 16)
        if self.flipped:                  # white's side is at the top now: it fills down from the top
            split = bar.y + int(round(bar.height * self.bar_pos))
            pygame.draw.rect(self.screen, (240, 240, 240), (bar.x, bar.y, bar.w, split - bar.y))
            pygame.draw.rect(self.screen, (14, 14, 17), (bar.x, split, bar.w, bar.bottom - split))
        else:
            split = bar.bottom - int(round(bar.height * self.bar_pos))   # white fills up from the bottom
            pygame.draw.rect(self.screen, (14, 14, 17), (bar.x, bar.y, bar.w, split - bar.y))
            pygame.draw.rect(self.screen, (240, 240, 240), (bar.x, split, bar.w, bar.bottom - split))
        for frac in (0.25, 0.5, 0.75):
            y = bar.y + int(bar.height * frac)
            pygame.draw.line(self.screen, (150, 150, 155) if frac != 0.5 else (230, 110, 40),
                             (bar.x, y), (bar.right - 1, y), 1)
        pygame.draw.rect(self.screen, (120, 120, 125), bar, 1)

    def draw_button(self, rect, label, on=False, enabled=True):
        if not enabled:
            bg, fg = BUTTON_OFF, MUTED
        elif on:
            bg, fg = BUTTON_ON, TEXT
        elif rect.collidepoint(self.mouse()):
            bg, fg = BUTTON_HOVER, TEXT
        else:
            bg, fg = BUTTON, TEXT
        pygame.draw.rect(self.screen, bg, rect, border_radius=6)
        surf = self.font_btn.render(label, True, fg)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw_panel(self):
        game = self.game
        L = self.layout
        x0 = L['x0']
        pygame.draw.rect(self.screen, PANEL_BG, (PANEL_X, BOARD_Y, PANEL_W, self.win_h - BOARD_Y))
        y = BOARD_Y + 10
        self.text("Reverse Chess", (x0, y), self.font)
        ply = self.font_small.render("ply %d" % game.ply, True, MUTED)
        self.screen.blit(ply, (x0 + L['w'] - ply.get_width(), y + 6))
        y = BOARD_Y + 90
        for team in TEAMS:
            y = self.draw_tray(team, x0, y) + 16
        self.draw_controls()
        legend = ["blue outline: can move", "grey lock: locked for good",
                  "orange lock: blocked right now", "gold P: promoted piece",
                  "menu bar: everything else, F1: help"]
        for i, line in enumerate(legend):
            self.text(line, (x0, self.win_h - 6 - (len(legend) - i) * 20), self.font_small, MUTED)

    def draw_tray(self, team, x, y, per_row=3, step=72):
        """One side's tray: pieces at home, and the pieces that are not on the board (``per_row`` of them
        per row, ``step`` pixels apart).  Returns the y under it."""
        game = self.game
        on_board = game.on_board(team)
        home = sum(1 for p in on_board if p.at_home())
        self.text("%s: %d/%d home" % (full_name(team).capitalize(), home, len(on_board)), (x, y))
        y += 26
        self.text("not on the board:", (x, y), self.font_small, MUTED)
        y += 20
        off = Counter(p.ptype for p in game.pieces[team] if not p.ontheboard)
        for i, ptype in enumerate(('p', 'r', 'n', 'b', 'q')):
            cx, cy = x + (i % per_row) * step, y + (i // per_row) * 38
            self.screen.blit(self.sprite(team, ptype, 32), (cx, cy))
            count = off.get(ptype, 0)
            self.text("x%d" % count, (cx + 34, cy + 8), self.font, TEXT if count else MUTED)
        rows = (5 + per_row - 1) // per_row
        return y + rows * 38

    def draw_strip(self):
        """The tall layout's controls (the canvas rows under the wide picture): everything the side panel
        shows, laid out across the width under the board instead of down beside it."""
        S = self.strip
        top = S['net'].y - 6                         # = WIN_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, top, PANEL_X, STRIP_H))
        pygame.draw.line(self.screen, (22, 22, 26), (0, top), (PANEL_X, top), 2)
        self.text("Reverse Chess", (14, top + 12), self.font)
        ply = self.font_small.render("ply %d" % self.game.ply, True, MUTED)
        self.screen.blit(ply, (14 + self.font.size("Reverse Chess")[0] + 12, top + 18))
        for team, x in zip(TEAMS, (14, 424)):
            self.draw_tray(team, x, top + 56, per_row=5, step=78)
        pygame.draw.line(self.screen, (60, 60, 66), (14, top + 142), (PANEL_X - 14, top + 142), 1)
        self.draw_controls(S)
        self.text("blue outline: can move   grey lock: locked for good   orange lock: blocked right now   "
                  "gold P: promoted   F1: help", (14, top + STRIP_H - 24), self.font_small, MUTED)

    def draw_controls(self, L=None):
        """The side panel below the piece trays: the AI's name, the eval line, hints, the record."""
        L = L or self.layout
        x0, w = L['x0'], L['w']
        hx, hw = L.get('hint_x', x0), L.get('hint_w', w)
        if self.advisor is not None:
            label = "AI: " + self.advisor.name
        else:
            label = self.ai_note or "No AI file found: the computer moves at random."
        net = L['net']                                   # click: pick another network (N)
        pygame.draw.rect(self.screen, BUTTON_HOVER if net.collidepoint(self.mouse()) else BUTTON, net, border_radius=6)
        lines = wrap(label, self.font_small, net.w - 34)[:2]
        top = net.y + (net.h - 17 * len(lines)) // 2
        for i, line in enumerate(lines):
            self.text(line, (net.x + 9, top + i * 17), self.font_small, TEXT)
        pygame.draw.polygon(self.screen, MUTED, [(net.right - 22, net.centery - 3), (net.right - 10, net.centery - 3),
                                                 (net.right - 16, net.centery + 4)])
        hint = self.current_hint()
        if self.show_eval and self.advisor is not None:
            sims = 0
            if self.game.done:
                vw = 0.0 if self.game.winner is None else 1.0 if self.game.winner == 'w' else -1.0
            else:
                vw, sims = self.white_value_now()
            who = "white ahead" if vw > 0.05 else "black ahead" if vw < -0.05 else "about even"
            how = "search %d" % sims if sims else "net"
            if self.game.done:
                how = "final"
            elif sims and self.analysis_job_here() is not None and self.analysis_job_here().solved:
                how = "forced win"
            self.text("Eval %+.2f  (%s, %s)" % (vw, who, how), (hx, L['eval_text_y']),
                      self.font_small, TEXT)
        alts = self.review_alts() if hint is None else None
        if alts is not None:
            y = L['hint_y']
            self.text(alts["header"], (hx, y), self.font_small, MUTED)
            y += 18
            for rank, line in enumerate(alts["rows"]):
                self.text(line, (hx, y), self.font_small, HINT_TEXT[rank])
                y += 17
            self.text(alts["played"], (hx, y + 3), self.font_small, REVIEW_PLAYED[:3])
        if hint is not None:
            y = L['hint_y']
            for rank, (action, p) in enumerate(hint["top"]):
                line = "%d. %d%%  %s" % (rank + 1, round(100 * p), describe(self.game, action))
                for text in wrap(line, self.font_small, hw)[:2]:
                    self.text(text, (hx, y), self.font_small, HINT_TEXT[rank])
                    y += 17
                y += 3
        self.draw_stats(L)

    def draw_stats(self, L=None):
        L = L or self.layout
        x0, w = L.get('stats_x', L['x0']), L['w']
        y = L['stats_y']
        self.text("You vs the AI", (x0, y), self.font_status)
        if self.stats is None:
            self.text("record switched off", (x0, y + 24), self.font_small, MUTED)
            return
        s = self.stats_summary
        a = s["all"]
        self.text("%d won   %d lost   %d drawn" % (a["won"], a["lost"], a["drawn"]),
                  (x0, y + 24), self.font)
        b, wh = s['b'], s['w']
        self.text("black %d-%d-%d   white %d-%d-%d" % (
            b["won"], b["lost"], b["drawn"], wh["won"], wh["lost"], wh["drawn"]),
            (x0, y + 50), self.font_small, MUTED)
        notes = []
        if s["hinted"]:
            notes.append("%d with hints" % s["hinted"])
        if s["abandoned"]:
            notes.append("%d abandoned" % s["abandoned"])
        if not self.stats_ok and not self.game.done:
            notes.append("this game is not counted")
        if notes:
            self.text(", ".join(notes), (x0, y + 70), self.font_small, MUTED)

    def moves_first_row(self):
        """Index of the first move-list row on screen.  Row 0 is the starting position, row k the position
        after move k (so clicking a row jumps to that many moves played)."""
        rows = self.layout['m_list'].h // ROW_H
        last_first = max(0, len(self.line) + 1 - rows)
        if self.moves_top is not None:
            return max(0, min(self.moves_top, last_first))
        cur = self.pos
        return cur if cur < last_first else last_first

    def scroll_moves(self, dy):
        self.moves_top = self.moves_first_row() - 3 * dy

    def draw_moves(self):
        """The move-list column: only while the window is expanded (review mode, a browsed game, or
        View > Move list > Always show)."""
        if not self.expanded:
            return
        L = self.layout
        x0 = MOVES_X + 14
        pygame.draw.rect(self.screen, PANEL_BG, (MOVES_X, BOARD_Y, MOVES_W, self.win_h - BOARD_Y))
        pygame.draw.line(self.screen, (22, 22, 26), (MOVES_X, BOARD_Y), (MOVES_X, self.win_h), 2)
        self.text("Moves", (x0, BOARD_Y + 8), self.font_big)
        has_marks = self.review is not None and any(self.review["marks"])
        self.draw_button(L['m_prev'], "Prev ?", enabled=has_marks)
        self.draw_button(L['m_next'], "Next ?", enabled=has_marks)
        if self.reviewer is not None:
            self.text("reviewing... %d%%" % int(100 * self.reviewer.progress), (x0, BOARD_Y + 80),
                      self.font_small, TEXT)
        elif self.review is None:
            self.text("Analysis > Review game (V)", (x0, BOARD_Y + 80), self.font_small, MUTED)
        self.text("<- -> step  Home/End", (x0, BOARD_Y + 98), self.font_small, MUTED)
        self.text("Tab: other line (%d)" % len(self.variations) if self.variations
                  else "Z undo  F flip  M next ?", (x0, BOARD_Y + 114), self.font_small,
                  TEXT if self.variations else MUTED)
        box = L['m_list']
        pygame.draw.rect(self.screen, (30, 30, 34), box, border_radius=4)
        first = self.moves_first_row()
        rows = box.h // ROW_H
        if first == 0:                                    # row 0: the position before any move
            if self.pos == 0:
                pygame.draw.rect(self.screen, (58, 86, 128), (box.x + 2, box.y, box.w - 4, ROW_H), border_radius=3)
            self.text("start position" if self.line else "start position (no moves yet)", (box.x + 10, box.y + 3),
                      self.font_small, TEXT if self.pos == 0 else MUTED)
        for i in range(max(first - 1, 0), min(len(self.line), first + rows - 1)):
            y = box.y + (i + 1 - first) * ROW_H
            if i == self.pos - 1:
                pygame.draw.rect(self.screen, (58, 86, 128), (box.x + 2, y, box.w - 4, ROW_H), border_radius=3)
            num = self.font_small.render(str(i + 1), True, MUTED)
            self.screen.blit(num, (box.x + 30 - num.get_width(), y + 3))
            dot_c = (235, 235, 235) if self.turns[i] == 'w' else (12, 12, 14)
            pygame.draw.circle(self.screen, dot_c, (box.x + 44, y + ROW_H // 2), 5)
            pygame.draw.circle(self.screen, (140, 140, 145), (box.x + 44, y + ROW_H // 2), 5, 1)
            mark = self.review_mark(i)
            colour = MARK_COLOURS.get(mark, TEXT if i < self.pos else MUTED)
            show_bar = self.review is not None and i + 1 < len(self.review["values"])
            self.screen.set_clip(pygame.Rect(box.x + 56, y, box.w - 56 - (EVAL_COL_W + 8 if show_bar else 2), ROW_H))
            self.text(self.texts[i] + mark, (box.x + 56, y + 3), self.font_small, colour)
            self.screen.set_clip(None)
            if show_bar:
                self.draw_eval_chip(pygame.Rect(box.right - EVAL_COL_W - 6, y + 5, EVAL_COL_W, ROW_H - 10),
                                    self.review["values"][i + 1], i)
        self.draw_graph()


    def draw_eval_chip(self, rect, white_value, i):
        """A tiny eval bar for the move list: the white part is white's share, centre line = even."""
        frac = (max(-1.0, min(1.0, white_value)) + 1.0) / 2.0
        pygame.draw.rect(self.screen, (14, 14, 17), rect)
        pygame.draw.rect(self.screen, (225, 225, 225), (rect.x, rect.y, int(round(rect.w * frac)), rect.h))
        pygame.draw.line(self.screen, (230, 110, 40), (rect.centerx, rect.y), (rect.centerx, rect.bottom - 1), 1)
        pygame.draw.rect(self.screen, (110, 110, 116), rect, 1)
        if rect.inflate(0, 8).collidepoint(self.mouse()):
            who = "white ahead" if white_value > 0.05 else "black ahead" if white_value < -0.05 else "about even"
            self.tooltip = ("After move %d: %+.2f (%s)" % (i + 1, white_value, who), TEXT)

    def draw_graph(self):
        g = self.layout['m_graph']
        pygame.draw.rect(self.screen, (28, 28, 32), g)
        pygame.draw.rect(self.screen, (70, 70, 76), g, 1)
        if self.review is None:
            label = ("reviewing..." if self.reviewer is not None else
                     "Review rates every move")
            self.text(label, (g.x + 10, g.centery - 8), self.font_small, MUTED)
            return
        values = self.review["values"]
        n = max(len(values) - 1, 1)
        mid, amp = g.centery, g.h / 2 - 5
        xs = lambda i: g.x + (g.w - 1) * i / n
        ys = lambda v: mid - max(-1.0, min(1.0, v)) * amp
        pygame.draw.line(self.screen, (90, 90, 96), (g.x, mid), (g.right - 1, mid), 1)
        if len(values) >= 2:
            pygame.draw.lines(self.screen, (120, 190, 235), False,
                              [(xs(i), ys(v)) for i, v in enumerate(values)], 2)
        hover = self.graph_mark_at(self.mouse())
        for i, mark in enumerate(self.review["marks"]):
            if mark:
                centre = (int(xs(i + 1)), int(ys(values[i + 1])))
                pygame.draw.circle(self.screen, MARK_COLOURS[mark], centre, 5 if i == hover else 3)
                if i == hover:
                    pygame.draw.circle(self.screen, (240, 240, 240), centre, 6, 1)
        if hover is not None:
            self.tooltip = (self.mark_text(hover), MARK_COLOURS[self.review["marks"][hover]])
        cx = int(xs(min(self.pos, len(values) - 1)))
        pygame.draw.line(self.screen, (240, 240, 240), (cx, g.y + 1), (cx, g.bottom - 2), 1)
        self.text("network's view, white up", (g.x, g.bottom + 3), self.font_small, MUTED)
        summary = reviewlib.summarize(self.turns, self.review["loss"], self.review["marks"])
        for k, team in enumerate(("w", "b")):
            self.text(reviewlib.summary_text(team, summary[team]), (g.x, g.bottom + 20 + 17 * k),
                      self.font_small, TEXT)

    def draw_status(self):
        game = self.game
        colour = TEXT
        if self.message:
            line, colour = self.message, BAD
        elif self.browsing:
            mark = self.review_mark(self.pos)
            if mark:
                name = {"??": "a blunder", "?": "a mistake", "?!": "an inaccuracy"}[mark]
                line = ("Next, move %d is %s (lost %.2f).  Green: network wanted %s.  Red: played %s.  "
                        "P plays the green move."
                        % (self.pos + 1, name, self.review["loss"][self.pos],
                           describe(game, self.review["best"][self.pos]),
                           describe(game, self.line[self.pos])))
                colour = MARK_COLOURS[mark]
            elif self.review_here():
                best = self.review["best"][self.pos]
                line = ("Before move %d of %d: the network's choice is %s (green)%s.  P plays it; a new move "
                        "from here starts another line (the old one is kept: Tab)."
                        % (self.pos + 1, len(self.line), describe(game, best),
                           "" if best == self.line[self.pos] else ", the game went %s (grey)"
                           % describe(game, self.line[self.pos])))
            else:
                line = ("Looking at the position after move %d of %d.  Right / End: forward.  A new move "
                        "from here starts another line (the old one is kept: Tab)." % (self.pos, len(self.line)))
        elif game.done:
            who = "Draw" if game.winner is None else "%s wins" % full_name(game.winner).capitalize()
            line = "%s: %s.  R: new game, Ctrl+S: save it." % (who, game.end_reason)
            colour = GOOD
        else:
            name = full_name(game.turn).capitalize()
            kind = "you" if self.agents[game.turn] is None else "computer"
            last = game.last_move
            extra = (" - extra turn (reverse en passant)"
                     if last is not None and last.extra_turn and last.team == game.turn else "")
            if kind == "computer" and self.searching():
                extra += " - thinking (%d sims)" % self.search_for_here().sims
            if self.premove is not None:
                extra += " - premove %s-%s queued" % (sq_name(self.premove[0]), sq_name(self.premove[1]))
            elif self.pre_src is not None:
                extra += " - premove: click where %s goes" % sq_name(self.pre_src)
            line = "%s to move (%s)%s" % (name, kind, extra)
        lines = wrap(line, self.font_status, PANEL_X - 28)
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1].rstrip(" .,;:") + " ..."
        for i, text in enumerate(lines):
            self.text(text, (14, BOARD_Y + BOARD_PX + 8 + i * 26), self.font_status, colour)

    def draw_menu(self):
        dim, values = self.pending
        game = self.game
        mover = game.board[self.selected]
        recv = other_team(game.turn)        # the team that receives a gift
        panel, rects, x_rect, back_rect = self.menu_geometry(len(values), bool(self.history))
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=12)
        titles = {
            'demote': "Un-promote this piece?",
            'gift': ("Give a piece back?" if None in values
                     else "Which piece do you give back?"),
            'en_passant': "Where does the gifted pawn go?",
            'promote': "Promote the gifted pawn?",
            'confirm': "Play this move?",
        }
        if panel.y >= WIN_H:                        # the tall layout's strip: title and hint on one line
            self.text(titles[dim], (panel.x + 20, panel.y + 10), self.font_big)
            self.text("click, or keys 1-9, Enter", (panel.x + 20 + self.font_big.size(titles[dim])[0] + 18, panel.y + 20),
                      self.font_small, MUTED)
        elif panel.w >= 440:
            self.text(titles[dim], (panel.x + 20, panel.y + 18), self.font_big)
            self.text("click, or keys 1-9 / Up Down + Enter;  Backspace back", (panel.x + 20, panel.y + 54),
                      self.font_small, MUTED)
        else:                                       # the narrow menu of the compact window
            for j, line in enumerate(wrap(titles[dim], self.font, panel.w - 76)[:2]):
                self.text(line, (panel.x + 20, panel.y + 10 + 26 * j), self.font)
            self.text("click, or keys 1-9 / arrows + Enter", (panel.x + 20, panel.y + 66), self.font_small, MUTED)
        self.draw_button(x_rect, "X")
        if back_rect is not None:
            self.draw_button(back_rect, "< Back")
        mouse = self.mouse()
        for i, (value, rect) in enumerate(zip(values, rects)):
            pygame.draw.rect(self.screen, (84, 84, 96) if rect.collidepoint(mouse) or i == self.menu_sel
                             else (58, 58, 66), rect, border_radius=8)
            self.text(str(i + 1), (rect.x + 3, rect.y + 2), self.font_small, MUTED)
            team, ptype, label = mover.team, mover.ptype, ""
            if dim == 'confirm':
                label = describe(game, self.candidates[0])
            elif dim == 'demote':
                label = ("Un-promote: step off as a pawn" if value
                         else "Keep going as a %s" % NAMES[mover.ptype])
                ptype = 'p' if value else mover.ptype
            elif dim == 'gift':
                label = ("No gift" if value is None
                         else "Give back a %s" % NAMES[value])
                team, ptype = recv, value
            elif dim == 'en_passant':
                label = ("Reverse en passant: pawn on its home row, you move again"
                         if value else "Ordinary: pawn on the square you left")
                team, ptype = recv, 'p'
            elif dim == 'promote':
                label = ("Leave it a pawn" if value is None
                         else "Promote it to a %s" % NAMES[value])
                team, ptype = recv, value or 'p'
            size = min(56, rect.h - 8)
            if not (dim == 'gift' and value is None):
                self.screen.blit(self.sprite(team, ptype, size), (rect.x + 8, rect.y + (rect.h - size) // 2))
            avail = rect.w - 34 - size
            if self.font.size(label)[0] <= avail:
                self.text(label, (rect.x + 20 + size, rect.centery - 10), self.font)
            else:
                lines = wrap(label, self.font_small, avail)[:3]
                top = rect.y + (rect.h - 17 * len(lines)) // 2
                for j, line in enumerate(lines):
                    self.text(line, (rect.x + 20 + size, top + 17 * j), self.font_small)


def menu_rects_tall(n, with_back=False):
    """The move menu of the tall layout: it covers the control strip under the board (so the board and the
    gift preview stay in view), with the options in one or two columns and the Back button in the header."""
    panel = pygame.Rect(8, WIN_H + 6, PANEL_X - 16, STRIP_H - 12)
    head, gap = 50, 8
    cols = 1 if n <= 3 else 2 if n <= 8 else 3
    rows = (n + cols - 1) // cols
    height = max(30, min(60, (panel.h - head - 12 - (rows - 1) * gap) // rows))
    width = (panel.w - 40 - (cols - 1) * 12) // cols
    rects = [pygame.Rect(panel.x + 20 + (i % cols) * (width + 12), panel.y + head + (i // cols) * (height + gap),
                         width, height) for i in range(n)]
    x_rect = pygame.Rect(panel.right - 46, panel.y + 8, 34, 34)
    back_rect = pygame.Rect(panel.right - 46 - 112, panel.y + 8, 100, 34) if with_back else None
    return panel, rects, x_rect, back_rect


def menu_rects(n, with_back=False, win_w=COMPACT_W, win_h=WIN_H):
    """(background, one rect per option, X button, Back button or None) for a menu of n options.  It sits
    to the right of the board (over the side panel), so the board stays in view: the gift preview and the
    pieces being decided about are never covered.  In the compact window that strip is narrow, so the menu
    is too (its text wraps)."""
    width = min(480, win_w - BOARD_PX - 44)
    height, gap, foot = 70, 10, 56
    head = 60 if width >= 400 else 88
    total = head + n * height + (n - 1) * gap + (foot if with_back else 0)
    left = BOARD_PX + (win_w - BOARD_PX - width) // 2
    top = (win_h - total) // 2
    panel = pygame.Rect(left - 20, top - 10, width + 40, total + 30)
    rects = [pygame.Rect(left, top + head + i * (height + gap), width, height)
             for i in range(n)]
    x_rect = pygame.Rect(panel.right - 46, panel.y + 12, 34, 34)
    back_rect = (pygame.Rect(left, top + head + n * height + (n - 1) * gap + 12, 130, 38)
                 if with_back else None)
    return panel, rects, x_rect, back_rect


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def selftest(advisor, ai_note):
    """Headless check of a (packaged) build: 0 = AI loaded and a search produced a move.

    Windowed Windows builds have no stdout, so the report also goes to $RC_SELFTEST_LOG if set."""
    import engine
    lines = []

    def say(msg):
        lines.append(msg)
        if sys.stdout is not None:
            print(msg)
    code = 0
    try:
        say("engine: " + ("native (Rust)" if engine.NATIVE else "pure Python (slow)"))
        if advisor is None:
            say("FAIL: no AI loaded. " + (ai_note or "ai.npz not found next to the program"))
            code = 1
        else:
            try:
                import rustnet
                say("native net: " + ("yes, SIMD=%s" % rustnet.simd_active() if rustnet.AVAILABLE else "no"))
            except Exception as err:
                say("native net: unavailable (%s)" % err)
            job = mcts.Searcher(advisor, sims=40, think_ms=20000).search(Game(Rules()), rng=random.Random(0))
            say("search ok: " + type(job).__name__)
            say("selftest passed")
    except BaseException:
        import traceback
        say("FAIL:\n" + traceback.format_exc())
        code = 1
    log = os.environ.get("RC_SELFTEST_LOG")
    if log:
        with open(log, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai", default=None, metavar="FILE",
                    help="network file: ai.npz, or a train.py checkpoint such as latest.pt "
                         "(default: the newest of ai.npz / ai.pt / latest.pt next to play.py)")
    ap.add_argument("--ai-color", choices=("b", "w", "both"), default="w",
                    help="which side the AI plays (default w: you are black)")
    ap.add_argument("--no-ai", action="store_true", help="start with the AI switched off")
    ap.add_argument("--eval", action="store_true", help="start with the eval bar on")
    ap.add_argument("--black", default=None, help="human | ai | random | module:function")
    ap.add_argument("--white", default=None, help="human | ai | random | module:function")
    ap.add_argument("--delay", type=int, default=400, help="ms before a computer moves")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=None, help="draw after this many plies")
    ap.add_argument("--ai-max-plies", type=int, default=300,
                    help="the move cap the AI was trained with (train.py --max-plies); only scales "
                         "one of its inputs, it does not end games")
    ap.add_argument("--no-search", action="store_true",
                    help="start with the look-ahead search (MCTS) off; the AI plays straight "
                         "from the network's policy")
    ap.add_argument("--sims", type=int, default=600,
                    help="most simulations the search runs per move (default 600)")
    ap.add_argument("--think-ms", type=int, default=2500,
                    help="most milliseconds the search thinks per move (default 2500)")
    ap.add_argument("--strict-win", action="store_true",
                    help="experimental: every piece of both teams must be on the board")
    ap.add_argument("--review-sims", type=int, default=reviewlib.DEFAULT_SIMS, metavar="N",
                    help="search simulations per position in the game review (default %d; 0 = "
                         "the bare network, a second or two for a whole game)" % reviewlib.DEFAULT_SIMS)
    ap.add_argument("--no-resume", action="store_true",
                    help="do not reopen the unfinished game of the last session, and do not keep this "
                         "one on disk after every move")
    ap.add_argument("--scale", type=float, default=None, metavar="F",
                    help="window size as a multiple of 1114 x 914, 1334 x 914 in review mode (default: whatever fits the screen, at most 1; "
                         "Ctrl +/- change it, and the window can be dragged to any size)")
    ap.add_argument("--no-sound", action="store_true", help="start with the move sounds off (U toggles)")
    ap.add_argument("--no-coords", action="store_true", help="start without rank / file labels on the board (C toggles)")
    ap.add_argument("--net-dir", action="append", default=[], metavar="DIR",
                    help="another folder of network files for the picker (N); may be repeated.  The picker also "
                         "lists *.npz / *.pt next to play.py, in runs/*/ and in nets/")
    ap.add_argument("--stats", default=None, metavar="FILE", help="where to keep the win/loss record")
    ap.add_argument("--settings", default=None, metavar="FILE",
                    help="where to remember your settings (difficulty, sound, ...; default: next to the record)")
    ap.add_argument("--no-settings", action="store_true", help="do not remember or restore settings")
    ap.add_argument("--no-stats", action="store_true", help="do not keep a win/loss record")
    ap.add_argument("--flip", action="store_true",
                    help="draw the board from black's side and keep it that way (by default the board is "
                         "turned so that your colour is at the bottom)")
    ap.add_argument("--fixed-board", action="store_true",
                    help="never turn the board by itself: white at the bottom unless F / --flip says otherwise")
    ap.add_argument("--load", default=None, metavar="FILE", help="open a saved game (Ctrl+S in the window saves one)")
    ap.add_argument("--autosave", action="store_true", help="save every finished game to your user data folder")
    ap.add_argument("--selftest", action="store_true",
                    help="load the AI and engine, run a short search, print a report and exit "
                         "(no window; used to check a packaged build)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    advisor, ai_note = None, ""
    path = args.ai or None
    if path is None:
        path = numpy_net.find_ai_file(app_dir(), os.getcwd())
    if path:
        try:
            advisor = numpy_net.Advisor(path, args.ai_max_plies)
        except Exception as err:             # missing PyTorch for a .pt, a damaged file, ...
            if args.ai:
                raise SystemExit("cannot load the AI file %s: %s" % (path, err))
            ai_note = "Could not load %s (%s); the computer moves at random." % (os.path.basename(path), err)

    if args.selftest:
        raise SystemExit(selftest(advisor, ai_note))

    custom = {"b": None, "w": None}
    sides = set()
    if args.black is not None or args.white is not None:
        for team, spec in (("b", args.black), ("w", args.white)):
            if spec in (None, "human"):
                continue
            sides.add(team)
            if spec != "ai":
                custom[team] = load_agent(spec, rng)
        ai_on = bool(sides)
    else:
        sides = {"b", "w"} if args.ai_color == "both" else {args.ai_color}
        ai_on = not args.no_ai

    stats = None if args.no_stats else Stats(args.stats)
    rules = Rules(win_requires_all_pieces_on_board=args.strict_win,
                  max_plies=args.max_plies)
    searcher = mcts.Searcher(advisor, sims=args.sims, think_ms=args.think_ms) if advisor else None
    app = App(rules, args.delay, rng, advisor, ai_note, custom, ai_on, sides, stats, args.eval,
              searcher, flipped=args.flip, autosave=args.autosave,
              review_sims=args.review_sims, session=not args.no_resume, scale=args.scale,
              sound=not args.no_sound, coords=not args.no_coords, net_dirs=args.net_dir,
              auto_orient=not (args.flip or args.fixed_board))
    if args.no_search:
        app.use_search = False
    if not args.no_settings:
        app.settings_path = args.settings or os.path.join(os.path.dirname(stats_default_path()), "settings.json")
        argv = " ".join(sys.argv[1:])
        given = set()
        for key, flags in (("net", ("--ai ",)), ("sound", ("--no-sound",)), ("coords", ("--no-coords",)),
                           ("eval", ("--eval",)), ("search", ("--no-search", "--sims")), ("think", ("--think-ms",)),
                           ("ai", ("--ai-color", "--no-ai", "--black", "--white")), ("scale", ("--scale",))):
            if any(f in argv + " " for f in flags):
                given.add(key)
        app.load_settings(given)
    if args.load:
        app.load(args.load)
    elif not args.no_resume:
        app.resume()

    clock = pygame.time.Clock()
    running = True
    while running:
        resize = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                resize = event.size                        # (a drag sends many: only the newest matters)
            elif event.type == pygame.KEYDOWN:
                app.handle_key(event)
            elif event.type == pygame.DROPFILE and app.pending is None and not app.help_open:
                app.drop_file(event.file)
            elif event.type == pygame.DROPTEXT and app.pending is None and not app.help_open:
                app.paste_text(event.text)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # (Shift held: a move with one variant still shows its menu, to confirm)
                app.handle_click(app.to_logical(event.pos), bool(pygame.key.get_mods() & pygame.KMOD_SHIFT), raw=event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if not app.help_open:
                    app.right_click()
            elif event.type == pygame.MOUSEWHEEL and not app.help_open:
                if app.picker is not None:
                    app.picker_scroll(event.y)
                elif app.menu_open is None and app.expanded and app.layout['m_list'].collidepoint(app.mouse()):
                    app.scroll_moves(event.y)
        if resize is not None:
            app.on_resize(resize)
        app.save_settings()
        if not app.help_open:
            app.poll_review()
            app.update_ai()
        app.menu_hover()
        app.draw()
        app.present()
        clock.tick(30)
    app.save_settings(force=True)
    pygame.quit()


if __name__ == "__main__":
    main()
