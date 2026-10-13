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
* AI on/off (A) and which colour(s) the AI plays: buttons in the panel.
* Your record against the AI (won / lost / drawn, by colour) is kept in a small file in your
  user data folder (see stats.py); a game counts only if the opponent was the AI from the first
  move to the last.  Games where you used hints are counted too, and noted.
* Keys: R = new game, Esc = back out of the menu / deselect, H, E, A as above.

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
import importlib
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pygame

import mcts
import numpy_net
from chessassets import TEAMS, PTYPES, image_path
from engine import SIZE, Game, IllegalActionError, Rules, full_name, other_team
from stats import Stats

# --------------------------------------------------------------------------
# Layout and colours
# --------------------------------------------------------------------------
SQ = 100                         # pixels per square
BOARD_PX = SQ * SIZE
BAR_W = 34                       # the eval bar, between the board and the panel
PANEL_X = BOARD_PX + BAR_W
PANEL_W = 280
STATUS_H = 92
WIN_W = PANEL_X + PANEL_W
WIN_H = BOARD_PX + STATUS_H

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

NAMES = {'p': 'pawn', 'k': 'king', 'r': 'rook', 'b': 'bishop', 'q': 'queen', 'n': 'knight'}

# A move may need these decisions, asked in this order.  Each is a field of
# engine.Action; a decision is only asked when the legal actions disagree on it.
DIMENSIONS = ('demote', 'gift', 'en_passant', 'promote')


def make_layout():
    """Rectangles of the panel's controls (fixed; only their labels change)."""
    x0 = PANEL_X + 14
    w = PANEL_W - 28
    half = (w - 8) // 2
    third = (w - 16) // 3   # also used by the AI / Eval / Search row
    R = pygame.Rect
    return {
        'x0': x0, 'w': w,
        'ai_label_y': 344,
        'ai': R(x0, 382, third, 30),
        'eval': R(x0 + third + 8, 382, third, 30),
        'search': R(x0 + 2 * (third + 8), 382, third, 30),
        'side_label_y': 418,
        'side_b': R(x0, 434, third, 28),
        'side_w': R(x0 + third + 8, 434, third, 28),
        'side_both': R(x0 + 2 * (third + 8), 434, third, 28),
        'hint': R(x0, 470, w, 30),
        'eval_text_y': 506,
        'hint_y': 524,
        'stats_y': 642,
        'reset': R(x0, 734, w, 24),
    }


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


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

class App:
    def __init__(self, rules, delay_ms, rng, advisor, ai_note, custom, ai_on, ai_sides,
                 stats, show_eval, searcher=None):
        pygame.init()
        pygame.display.set_caption("Reverse Chess")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
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
        self.ai_note = ai_note        # why there is no advisor, shown in the panel
        self.custom = custom          # {'b': fn|None, 'w': fn|None}: agents given on the command line
        self.ai_on = ai_on
        self.ai_sides = set(ai_sides)
        self.stats = stats
        self.show_eval = show_eval and advisor is not None
        self.searcher = searcher if advisor is not None else None   # mcts.Searcher or None
        self.use_search = self.searcher is not None
        self.search_job = None
        np_rng = np.random.default_rng(rng.randrange(2 ** 32))
        if advisor is not None:
            self.default_ai = lambda game: advisor.choose(game, np_rng)
        else:
            self.default_ai = lambda game: rng.choice(game.legal_actions())
        self.stats_summary = stats.summary() if stats is not None else None
        self.confirm_reset_until = 0
        self.refresh_agents()
        self.new_game()
        if stats is not None and stats.error:
            self.message = stats.error

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
        self.stop_search()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        self.reindex()

    def restart(self):
        """Start a new game; one abandoned part-way is noted in the record."""
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
        self.history = []
        self.message = ""

    def close_menu(self):
        """Close the move menu without playing; the piece stays selected."""
        self.candidates = []
        self.pending = None
        self.history = []
        self.message = ""

    def menu_back(self):
        """Undo the last answer in the menu (only offered when there is one)."""
        self.candidates = self.history.pop()
        self.pending = next_choice(self.candidates)

    def escape(self):
        if self.pending is not None:
            self.close_menu()
        else:
            self.cancel()

    def play(self, action):
        try:
            self.game.step(action)
        except IllegalActionError as err:
            self.message = str(err)
        else:
            self.message = ""
        self.cancel_keep_message()
        self.reindex()
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        if self.game.done:
            self.record_finished()

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
        self.cancel()
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
        self.search_job = None

    def toggle_search(self):
        if self.searcher is None:
            self.message = "Search needs the AI file (ai.npz or latest.pt next to play.py)."
            return
        self.use_search = not self.use_search
        self.stop_search()
        self.hint = None
        self.bar_pos = None
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
        job = self.search_for_here() if self.use_search else None
        if job is not None and (job.solved or job.sims >= 30):
            v = job.white_value()
            if v is not None:
                return v, job.sims
        return self.advisor.white_value(self.game), 0

    def current_hint(self):
        if self.hint is not None and self.hint["key"] != self.pos_key():
            self.hint = None                      # the position moved on
        if self.hint is not None and self.use_search:
            job = self.search_for_here()
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
            self.hint = {"key": self.pos_key(), "top": self.advisor.analyse(self.game).top(3)}
            self.hint_used = True
            self.ensure_search(1800)              # the hint sharpens as the search deepens
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
        L = self.layout
        if L['ai'].collidepoint(pos):
            self.toggle_ai()
        elif L['eval'].collidepoint(pos):
            self.toggle_eval()
        elif L['search'].collidepoint(pos):
            self.toggle_search()
        elif L['side_b'].collidepoint(pos):
            self.set_sides({'b'})
        elif L['side_w'].collidepoint(pos):
            self.set_sides({'w'})
        elif L['side_both'].collidepoint(pos):
            self.set_sides({'b', 'w'})
        elif L['hint'].collidepoint(pos):
            self.toggle_hint()
        elif L['reset'].collidepoint(pos):
            self.click_reset()

    # -- input -----------------------------------------------------------
    def human_to_move(self):
        return not self.game.done and self.agents[self.game.turn] is None

    def handle_click(self, pos):
        if self.pending is not None:              # the menu is modal
            self.click_menu(pos)
            return
        if pos[0] >= BOARD_PX:
            self.click_panel(pos)
            return
        if not self.human_to_move():
            return
        x, y = pos
        if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
            return
        sq = (y // SQ, x // SQ)
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
            self.resolve()
        else:
            self.message = (game.explain_illegal(self.selected, sq)
                            or "That move is not available.")

    def resolve(self):
        """Play the move, or open a menu for the next undecided choice."""
        choice = next_choice(self.candidates)
        if choice is None:
            self.play(self.candidates[0])
        else:
            self.pending = choice

    def click_menu(self, pos):
        dim, values = self.pending
        panel, rects, x_rect, back_rect = menu_rects(len(values), bool(self.history))
        if x_rect.collidepoint(pos):
            self.close_menu()
        elif back_rect is not None and back_rect.collidepoint(pos):
            self.menu_back()
        else:
            for value, rect in zip(values, rects):
                if rect.collidepoint(pos):
                    self.history.append(self.candidates)
                    self.candidates = [a for a in self.candidates if getattr(a, dim) == value]
                    self.pending = None
                    self.resolve()
                    return
            if not panel.collidepoint(pos):   # a click away from the menu closes it
                self.close_menu()

    def update_ai(self):
        game = self.game
        if game.done:
            return
        agent = self.agents[game.turn]
        if agent is None:                         # a human's turn: think ahead for the eval bar / hint
            if self.use_search and (self.show_eval or self.current_hint() is not None):
                self.ensure_search(1800)
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

    # -- drawing ---------------------------------------------------------
    def sprite(self, team, ptype, size):
        key = (team, ptype, size)
        if key not in self._sprites:
            img = pygame.image.load(image_path(team, ptype)).convert_alpha()
            self._sprites[key] = pygame.transform.smoothscale(img, (size, size))
        return self._sprites[key]

    def text(self, string, pos, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(string, True, color), pos)

    def draw(self):
        self.screen.fill(STATUS_BG)
        self.draw_board()
        self.draw_hint_arrows()
        self.draw_eval_bar()
        self.draw_panel()
        self.draw_status()
        if self.pending is not None:
            self.draw_menu()

    def draw_board(self):
        game = self.game
        last = game.last_move
        dests = {a.dst for a in self.by_src.get(self.selected, [])}
        movable = self.human_to_move()
        for r in range(SIZE):
            for c in range(SIZE):
                rect = pygame.Rect(c * SQ, r * SQ, SQ, SQ)
                colour = LIGHT if (r + c) % 2 == 0 else DARK
                if last is not None and (r, c) in (last.src, last.dst):
                    colour = LAST
                if (r, c) == self.selected:
                    colour = SELECT
                pygame.draw.rect(self.screen, colour, rect)
                piece = game.board[r, c]
                if piece is not None:
                    self.screen.blit(self.sprite(piece.team, piece.ptype, SQ - 12),
                                     (c * SQ + 6, r * SQ + 6))
                    if piece.promoted:   # a pawn wearing another piece's type
                        centre = (c * SQ + SQ - 18, r * SQ + 18)
                        pygame.draw.circle(self.screen, (230, 190, 40), centre, 12)
                        label = self.font_small.render("P", True, (20, 20, 20))
                        self.screen.blit(label, label.get_rect(center=centre))
                    if piece.locked:     # can never move again
                        draw_padlock(self.screen, c * SQ + 6, r * SQ + SQ - 26, LOCKED)
                    elif (r, c) in self.blocked:   # cannot move right now
                        draw_padlock(self.screen, c * SQ + 6, r * SQ + SQ - 26, BLOCKED)
                if movable and (r, c) in self.by_src:
                    pygame.draw.rect(self.screen, MOVABLE, rect, 4)
                if (r, c) in dests:
                    pygame.draw.circle(self.screen, DOT, rect.center, 11)

    def draw_hint_arrows(self):
        hint = self.current_hint()
        if hint is None:
            return
        overlay = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        seen = set()
        for rank, (action, _) in enumerate(hint["top"]):
            if (action.src, action.dst) in seen:
                continue
            seen.add((action.src, action.dst))
            centre = lambda sq: (sq[1] * SQ + SQ // 2, sq[0] * SQ + SQ // 2)
            draw_arrow(overlay, centre(action.src), centre(action.dst),
                       HINT_COLOURS[rank], HINT_WIDTHS[rank])
        self.screen.blit(overlay, (0, 0))

    def eval_target(self):
        """Where the bar should sit: 1 = all white, 0 = all black."""
        game = self.game
        if game.done:
            return 0.5 if game.winner is None else 1.0 if game.winner == 'w' else 0.0
        return (max(-1.0, min(1.0, self.white_value_now()[0])) + 1.0) / 2.0

    def draw_eval_bar(self):
        pygame.draw.rect(self.screen, PANEL_BG, (BOARD_PX, 0, BAR_W, BOARD_PX))
        if not self.show_eval or self.advisor is None:
            return
        target = self.eval_target()
        self.bar_pos = target if self.bar_pos is None else self.bar_pos + (target - self.bar_pos) * 0.25
        bar = pygame.Rect(BOARD_PX + 7, 8, BAR_W - 14, BOARD_PX - 16)
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
        elif rect.collidepoint(pygame.mouse.get_pos()):
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
        pygame.draw.rect(self.screen, PANEL_BG, (PANEL_X, 0, PANEL_W, WIN_H))
        y = 12
        self.text("Reverse Chess", (x0, y), self.font_big)
        y += 34
        self.text("ply %d" % game.ply, (x0, y), self.font, MUTED)
        y += 34
        for team in TEAMS:
            on_board = game.on_board(team)
            home = sum(1 for p in on_board if p.at_home())
            self.text("%s: %d/%d home" % (full_name(team).capitalize(), home, len(on_board)),
                      (x0, y))
            y += 26
            self.text("not on the board:", (x0, y), self.font_small, MUTED)
            y += 20
            off = Counter(p.ptype for p in game.pieces[team] if not p.ontheboard)
            for i, ptype in enumerate(('p', 'r', 'n', 'b', 'q')):
                cx, cy = x0 + (i % 3) * 72, y + (i // 3) * 38
                self.screen.blit(self.sprite(team, ptype, 32), (cx, cy))
                count = off.get(ptype, 0)
                self.text("x%d" % count, (cx + 34, cy + 8), self.font,
                          TEXT if count else MUTED)
            y += 2 * 38 + 16
        self.draw_controls()
        legend = ["blue outline: can move", "grey lock: locked for good",
                  "orange lock: blocked right now", "gold P: promoted piece",
                  "H hint  E eval  A AI  S MCTS  R new  Esc back"]
        for i, line in enumerate(legend):
            self.text(line, (x0, WIN_H - 8 - (len(legend) - i) * 21), self.font_small, MUTED)

    def draw_controls(self):
        L = self.layout
        x0, w = L['x0'], L['w']
        if self.advisor is not None:
            label = "AI: " + self.advisor.name
        else:
            label = self.ai_note or "No AI file found: the computer moves at random."
        for i, line in enumerate(wrap(label, self.font_small, w)[:2]):
            self.text(line, (x0, L['ai_label_y'] + i * 17), self.font_small, MUTED)
        self.draw_button(L['ai'], "AI ON" if self.ai_on else "AI OFF", on=self.ai_on)
        self.draw_button(L['eval'], "Eval ON" if self.show_eval else "Eval OFF",
                         on=self.show_eval, enabled=self.advisor is not None)
        self.draw_button(L['search'], "MCTS ON" if self.use_search else "MCTS OFF",
                         on=self.use_search, enabled=self.searcher is not None)
        self.text("AI plays:", (x0, L['side_label_y']), self.font_small, MUTED)
        for name, text, sides in (('side_b', "Black", {'b'}), ('side_w', "White", {'w'}),
                                  ('side_both', "Both", {'b', 'w'})):
            self.draw_button(L[name], text, on=self.ai_on and self.ai_sides == sides,
                             enabled=self.ai_on)
        hint = self.current_hint()
        self.draw_button(L['hint'], "Hide hint" if hint else "Hint",
                         on=hint is not None, enabled=self.advisor is not None)
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
            elif sims and self.search_for_here() is not None and self.search_for_here().solved:
                how = "forced win"
            self.text("Eval %+.2f  (%s, %s)" % (vw, who, how), (x0, L['eval_text_y']),
                      self.font_small, TEXT)
        if hint is not None:
            y = L['hint_y']
            for rank, (action, p) in enumerate(hint["top"]):
                line = "%d. %d%%  %s" % (rank + 1, round(100 * p), describe(self.game, action))
                for text in wrap(line, self.font_small, w)[:2]:
                    self.text(text, (x0, y), self.font_small, HINT_TEXT[rank])
                    y += 17
                y += 3
        self.draw_stats()

    def draw_stats(self):
        L = self.layout
        x0, w = L['x0'], L['w']
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
        self.text("as black %d-%d-%d    as white %d-%d-%d" % (
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
        confirming = pygame.time.get_ticks() < self.confirm_reset_until
        self.draw_button(L['reset'], "Click again to erase" if confirming else "Reset record",
                         on=confirming, enabled=bool(self.stats.games))

    def draw_status(self):
        game = self.game
        colour = TEXT
        if self.message:
            line, colour = self.message, BAD
        elif game.done:
            who = "Draw" if game.winner is None else "%s wins" % full_name(game.winner).capitalize()
            line = "%s: %s.  Press R for a new game." % (who, game.end_reason)
            colour = GOOD
        else:
            name = full_name(game.turn).capitalize()
            kind = "you" if self.agents[game.turn] is None else "computer"
            last = game.last_move
            extra = (" - extra turn (reverse en passant)"
                     if last is not None and last.extra_turn and last.team == game.turn else "")
            if kind == "computer" and self.searching():
                extra += " - thinking (%d sims)" % self.search_for_here().sims
            line = "%s to move (%s)%s" % (name, kind, extra)
        lines = wrap(line, self.font_status, PANEL_X - 28)
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1].rstrip(" .,;:") + " ..."
        for i, text in enumerate(lines):
            self.text(text, (14, BOARD_PX + 8 + i * 26), self.font_status, colour)

    def draw_menu(self):
        dim, values = self.pending
        game = self.game
        mover = game.board[self.selected]
        recv = other_team(game.turn)        # the team that receives a gift
        veil = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 140))
        self.screen.blit(veil, (0, 0))
        panel, rects, x_rect, back_rect = menu_rects(len(values), bool(self.history))
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=12)
        titles = {
            'demote': "Un-promote this piece?",
            'gift': ("Give a piece back?" if None in values
                     else "Which piece do you give back?"),
            'en_passant': "Where does the gifted pawn go?",
            'promote': "Promote the gifted pawn?",
        }
        self.text(titles[dim], (panel.x + 20, panel.y + 18), self.font_big)
        self.draw_button(x_rect, "X")
        if back_rect is not None:
            self.draw_button(back_rect, "< Back")
        mouse = pygame.mouse.get_pos()
        for value, rect in zip(values, rects):
            pygame.draw.rect(self.screen, (84, 84, 96) if rect.collidepoint(mouse)
                             else (58, 58, 66), rect, border_radius=8)
            team, ptype, label = mover.team, mover.ptype, ""
            if dim == 'demote':
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
            if not (dim == 'gift' and value is None):
                self.screen.blit(self.sprite(team, ptype, 56), (rect.x + 8, rect.y + 7))
            self.text(label, (rect.x + 76, rect.y + 24), self.font)


def menu_rects(n, with_back=False):
    """(background, one rect per option, X button, Back button or None) for a centred menu of n options."""
    width, height, gap, head, foot = 600, 70, 10, 60, 56
    total = head + n * height + (n - 1) * gap + (foot if with_back else 0)
    left = (BOARD_PX - width) // 2
    top = (BOARD_PX - total) // 2
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
    ap.add_argument("--stats", default=None, metavar="FILE", help="where to keep the win/loss record")
    ap.add_argument("--no-stats", action="store_true", help="do not keep a win/loss record")
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
              searcher)
    if args.no_search:
        app.use_search = False

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    app.escape()
                elif event.key == pygame.K_r:
                    app.restart()
                elif app.pending is None:
                    if event.key == pygame.K_h:
                        app.toggle_hint()
                    elif event.key == pygame.K_e:
                        app.toggle_eval()
                    elif event.key == pygame.K_a:
                        app.toggle_ai()
                    elif event.key == pygame.K_s:
                        app.toggle_search()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                app.handle_click(event.pos)
        app.update_ai()
        app.draw()
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
