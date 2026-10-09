#!/usr/bin/env python
"""Pygame front-end for Reverse Chess, built on the headless engine.

    python play.py                          # you (black) vs a random computer (white)
    python play.py --white human            # hot-seat, two humans
    python play.py --black random           # you play white against a random black
    python play.py --black random --white random --delay 100   # watch a game
    python play.py --white mymodule:policy  # plug in your own agent (see below)

How to play
-----------
* Click one of your pieces: its legal destinations get a green dot.
* Click a destination.  If the move needs a decision (un-promote? which piece
  to give back? reverse en passant? promote the gifted pawn to what?) a menu
  appears in the window.  Esc cancels back to the board.
* The panel on the right shows which pieces are still off the board for each
  team (those are the pieces the *other* side can give back) and how many of
  a team's on-board pieces are home.
* Keys: R = new game, Esc = cancel selection/menu.

Custom agents
-------------
``--black`` / ``--white`` accept ``human``, ``random`` or ``module:function``.
The function is called as ``function(game)`` and must return one of
``game.legal_actions()``.  A trained network fits here directly.

Only pygame and numpy are needed; there is no tkinter and no pop-up window.
"""
import argparse
import importlib
import random
from collections import Counter, defaultdict

import pygame

from chessassets import TEAMS, PTYPES, image_path
from engine import SIZE, Game, IllegalActionError, Rules, full_name, other_team

# --------------------------------------------------------------------------
# Layout and colours
# --------------------------------------------------------------------------
SQ = 100                         # pixels per square
BOARD_PX = SQ * SIZE
PANEL_W = 240
STATUS_H = 44
WIN_W = BOARD_PX + PANEL_W
WIN_H = BOARD_PX + STATUS_H

LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
SELECT = (230, 220, 60)
LAST = (205, 210, 106)
DOT = (40, 120, 220)
PANEL_BG = (38, 38, 42)
STATUS_BG = (24, 24, 28)
TEXT = (235, 235, 235)
MUTED = (150, 150, 155)
BAD = (240, 120, 100)
GOOD = (130, 220, 130)

NAMES = {'p': 'pawn', 'k': 'king', 'r': 'rook', 'b': 'bishop', 'q': 'queen', 'n': 'knight'}

# A move may need these decisions, asked in this order.  Each is a field of
# engine.Action; a decision is only asked when the legal actions disagree on it.
DIMENSIONS = ('demote', 'gift', 'en_passant', 'promote')


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
# Agents
# --------------------------------------------------------------------------

def load_agent(spec, rng):
    """None means human; otherwise a function game -> Action."""
    if spec == 'human':
        return None
    if spec == 'random':
        return lambda game: rng.choice(game.legal_actions())
    module, _, name = spec.partition(':')
    if not name:
        raise SystemExit("agent must be 'human', 'random' or 'module:function', got %r" % spec)
    return getattr(importlib.import_module(module), name)


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

class App:
    def __init__(self, agents, rules, delay_ms, rng):
        pygame.init()
        pygame.display.set_caption("Reverse Chess")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(None, 28)
        self.font_big = pygame.font.SysFont(None, 38)
        self.font_small = pygame.font.SysFont(None, 21)
        self._sprites = {}
        self.agents = agents          # {'b': fn|None, 'w': fn|None}
        self.rules = rules
        self.delay_ms = delay_ms
        self.rng = rng
        self.new_game()

    # -- state -----------------------------------------------------------
    def new_game(self):
        self.game = Game(self.rules)
        self.selected = None          # square of the piece being moved
        self.candidates = []          # actions for the chosen src -> dst
        self.pending = None           # (dimension, values) while a menu is open
        self.message = ""
        self.ai_due = pygame.time.get_ticks() + self.delay_ms
        self.reindex()

    def reindex(self):
        """Group the side to move's legal actions by source square."""
        self.by_src = defaultdict(list)
        for action in self.game.legal_actions():
            self.by_src[action.src].append(action)

    def cancel(self):
        self.selected = None
        self.candidates = []
        self.pending = None
        self.message = ""

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

    def cancel_keep_message(self):
        message = self.message
        self.cancel()
        self.message = message

    # -- input -----------------------------------------------------------
    def human_to_move(self):
        return not self.game.done and self.agents[self.game.turn] is None

    def handle_click(self, pos):
        if not self.human_to_move():
            return
        if self.pending is not None:
            self.click_menu(pos)
            return
        x, y = pos
        if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
            return
        sq = (y // SQ, x // SQ)
        piece = self.game.board[sq]
        if piece is not None and piece.team == self.game.turn:
            if sq in self.by_src:
                self.selected = sq
                self.message = ""
            else:
                self.selected = None
                self.message = "That piece has no legal moves."
        elif self.selected is not None and sq in {a.dst for a in self.by_src.get(self.selected, [])}:
            self.candidates = [a for a in self.by_src[self.selected] if a.dst == sq]
            self.resolve()
        else:
            self.selected = None

    def resolve(self):
        """Play the move, or open a menu for the next undecided choice."""
        choice = next_choice(self.candidates)
        if choice is None:
            self.play(self.candidates[0])
        else:
            self.pending = choice

    def click_menu(self, pos):
        dim, values = self.pending
        _, rects = menu_rects(len(values))
        for value, rect in zip(values, rects):
            if rect.collidepoint(pos):
                self.candidates = [a for a in self.candidates if getattr(a, dim) == value]
                self.pending = None
                self.resolve()
                return
        self.cancel()                 # clicked outside every option

    def update_ai(self):
        game = self.game
        if game.done or self.agents[game.turn] is None:
            return
        if pygame.time.get_ticks() < self.ai_due:
            return
        action = self.agents[game.turn](game)
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
        self.draw_panel()
        self.draw_status()
        if self.pending is not None:
            self.draw_menu()

    def draw_board(self):
        game = self.game
        last = game.last_move
        dests = {a.dst for a in self.by_src.get(self.selected, [])}
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
                        pygame.draw.circle(self.screen, (70, 70, 80),
                                           (c * SQ + 14, r * SQ + SQ - 14), 7)
                if (r, c) in dests:
                    pygame.draw.circle(self.screen, DOT, rect.center, 11)

    def draw_panel(self):
        game = self.game
        x0 = BOARD_PX + 14
        pygame.draw.rect(self.screen, PANEL_BG, (BOARD_PX, 0, PANEL_W, WIN_H))
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
        legend = ["P badge: promoted piece", "grey dot: locked in place",
                  "R: new game   Esc: cancel"]
        for i, line in enumerate(legend):
            self.text(line, (x0, BOARD_PX - 70 + i * 22), self.font_small, MUTED)

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
            line = "%s to move (%s)%s" % (name, kind, extra)
        self.text(line, (14, BOARD_PX + 12))

    def draw_menu(self):
        dim, values = self.pending
        game = self.game
        mover = game.board[self.selected]
        recv = other_team(game.turn)        # the team that receives a gift
        veil = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 140))
        self.screen.blit(veil, (0, 0))
        panel, rects = menu_rects(len(values))
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=12)
        titles = {
            'demote': "Un-promote this piece?",
            'gift': ("Give a piece back?" if None in values
                     else "Which piece do you give back?"),
            'en_passant': "Where does the gifted pawn go?",
            'promote': "Promote the gifted pawn?",
        }
        self.text(titles[dim], (panel.x + 20, panel.y + 18), self.font_big)
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


def menu_rects(n):
    """(background rect, one rect per option) for a centred menu of n options."""
    width, height, gap, head = 600, 70, 10, 60
    total = head + n * height + (n - 1) * gap
    left = (BOARD_PX - width) // 2
    top = (BOARD_PX - total) // 2
    panel = pygame.Rect(left - 20, top - 10, width + 40, total + 30)
    rects = [pygame.Rect(left, top + head + i * (height + gap), width, height)
             for i in range(n)]
    return panel, rects


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--black", default="human", help="human | random | module:function")
    ap.add_argument("--white", default="random", help="human | random | module:function")
    ap.add_argument("--delay", type=int, default=400, help="ms before a computer moves")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=None, help="draw after this many plies")
    ap.add_argument("--strict-win", action="store_true",
                    help="experimental: every piece of both teams must be on the board")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    agents = {"b": load_agent(args.black, rng), "w": load_agent(args.white, rng)}
    rules = Rules(win_requires_all_pieces_on_board=args.strict_win,
                  max_plies=args.max_plies)
    app = App(agents, rules, args.delay, rng)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    app.cancel()
                elif event.key == pygame.K_r:
                    app.new_game()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                app.handle_click(event.pos)
        app.update_ai()
        app.draw()
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
