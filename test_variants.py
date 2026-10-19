#!/usr/bin/env python
"""Headless checks of the undo button / Ctrl+Z shortcut and of the two variants:

    * gift limit   Rules.max_gifts   each side may give back at most N pieces in the whole game
    * custom start Rules.start       any pieces on the board at move 0 (+ Rules.start_turn)

    python test_variants.py
"""
import os
import random
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
_tmp = tempfile.mkdtemp()
os.environ["XDG_DATA_HOME"] = _tmp                       # saved games / autosave go to a scratch folder

import pygame  # noqa: E402

import gamelog  # noqa: E402
import play  # noqa: E402
from engine import Game, Rules, start_problem  # noqa: E402

KINGS = (("b", "k", 3, 5), ("w", "k", 4, 3))
SETUP = KINGS + (("w", "q", 5, 5), ("b", "r", 2, 2), ("w", "p", 4, 4), ("b", "p", 3, 1))   # black to move: playable


def key(k, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode="")


def make_app(rules=None, ai_on=False, session=False):
    return play.App(rules or Rules(max_plies=300), 0, random.Random(1), None, "", {"b": None, "w": None},
                    ai_on, {"w"}, None, False, None, session=session)


def playout(rules, seed, check=None):
    rng = random.Random(seed)
    game = Game(rules)
    while not game.done:
        acts = game.legal_actions()
        if check:
            check(game, acts)
        game.step(rng.choice(acts))
    game.validate()
    return game


# --------------------------------------------------------------------------- engine
def test_cap_is_neutral_when_huge():
    for seed in range(5):
        rng = random.Random(seed)
        a, b = Game(Rules(max_plies=200), native=False), Game(Rules(max_plies=200, max_gifts=999))
        while not a.done:
            assert set(a.legal_actions()) == set(b.legal_actions()), seed
            act = rng.choice(a.legal_actions())
            a.step(act)
            b.step(act)
        assert b.done and (a.winner, a.end_reason) == (b.winner, b.end_reason)
    print("ok: a gift limit nobody reaches changes nothing")


def test_cap_enforced():
    for cap in (0, 1, 2, 3):
        def check(game, acts, cap=cap):
            for team in "bw":
                assert game.gifts_given(team) <= cap
            if game.gifts_left(game.turn) == 0:
                assert not any(a.gift for a in acts), "gift offered with none left"
        used = 0
        for seed in range(8):
            g = playout(Rules(max_plies=250, max_gifts=cap), seed, check)
            used += g.gifts_given("b") + g.gifts_given("w")
        assert cap == 0 or used > 0, "the limit is so tight that nobody ever gifts"
    g = Game(Rules(max_gifts=0))
    assert g.gifts_left("b") == 0 and all(a.gift is None for a in g.legal_actions())
    assert Game().gifts_left("b") is None
    print("ok: gift limit enforced for every side, every move")


def test_custom_start():
    g = Game(Rules(start=SETUP))
    assert g.turn == "b" and len(g.on_board("b")) == 3 and len(g.on_board("w")) == 3
    g.validate()
    assert Game(Rules(start=SETUP, start_turn="w")).turn == "w"
    assert start_problem(Rules(start=SETUP)) is None
    assert start_problem(Rules(start=KINGS)) is None
    bad = {
        "no king": (("b", "k", 3, 5),),
        "two on a square": KINGS + (("w", "q", 3, 5),),
        "pawn on the last row": KINGS + (("w", "p", 0, 3),),
        "nine pawns": KINGS + tuple(("w", "p", 3, c) for c in range(8)) + (("w", "p", 5, 0),),
        "bishops on one colour": KINGS + (("w", "b", 5, 0), ("w", "b", 5, 2)),
        "everyone already home": (("b", "k", 0, 4), ("w", "k", 7, 4)),
    }
    for name, start in bad.items():
        assert start_problem(Rules(start=start)), name
    print("ok: custom start (placement, side to move, bad setups refused)")


def test_variants_combine_and_replay():
    rules = Rules(max_plies=200, max_gifts=2, start=SETUP)
    for seed in range(6):
        g = playout(rules, seed)
        # every prefix replays to the same position (this is what undo does)
        for n in (0, len(g.move_log) // 2, len(g.move_log)):
            r = gamelog.replay(rules, g.move_log, n)
            assert r.ply == n and r.gifts_given("b") <= 2 and r.gifts_given("w") <= 2
    print("ok: both variants together, replay of every prefix")


def test_json_roundtrip():
    rules = Rules(max_plies=200, max_gifts=2, start=KINGS + (("b", "r", 2, 2), ("w", "p", 4, 4)), start_turn="w")
    assert start_problem(rules) is None
    g = Game(rules)
    assert g.turn == "w"
    for _ in range(4):
        g.step(random.Random(len(g.move_log)).choice(g.legal_actions()))
    text = gamelog.game_json(rules, g.move_log)
    rules2, actions, _ = gamelog.loads_game(text)
    assert rules2 == rules and actions == g.move_log
    assert hash(rules2) == hash(rules)
    old = gamelog.loads_game(gamelog.game_json(Rules(max_plies=50), []))[0]       # files without the new keys
    assert old.standard
    assert gamelog.parse_start(gamelog.format_start(SETUP)) == SETUP
    assert gamelog.parse_start("wKd4 bKf5") == (("w", "k", 4, 3), ("b", "k", 3, 5))
    print("ok: save / load / paste keep the variant, old files still load")


def test_fast_paths_refuse_variants():
    import fastgame
    for rules in (Rules(max_gifts=3), Rules(start=SETUP), Rules(start_turn="w")):
        try:
            fastgame._Config(rules)
        except ValueError:
            continue
        raise AssertionError("fastgame accepted %r" % (rules,))
    fastgame._Config(Rules())
    print("ok: the native step refuses what it cannot play (no silent wrong rules in training)")


# --------------------------------------------------------------------------- play.py
def test_undo():
    app = make_app()
    rng = random.Random(5)
    for _ in range(6):
        app.play(rng.choice(app.game.legal_actions()))
    assert app.pos == 6
    app.handle_key(key(pygame.K_z, pygame.KMOD_LCTRL))
    assert (app.pos, app.game.ply) == (5, 5), "Ctrl+Z"
    app.handle_key(key(pygame.K_z, pygame.KMOD_LMETA))
    assert (app.pos, app.game.ply) == (4, 4), "Cmd+Z"
    app.handle_key(key(pygame.K_z, pygame.KMOD_LCTRL | pygame.KMOD_LSHIFT))
    assert app.pos == 5, "Ctrl+Shift+Z redoes"
    app.handle_click(app.layout["undo"].center)
    assert app.pos == 4, "the Undo button"
    app.handle_key(key(pygame.K_z))
    assert app.pos == 3, "plain Z still works"
    for _ in range(5):
        app.handle_key(key(pygame.K_z, pygame.KMOD_LCTRL))
    assert app.pos == 0 and "Nothing" in app.message
    app.draw()
    # with the computer playing white, one Undo takes back both plies
    app = make_app(ai_on=True)
    app.agents["w"] = lambda g: g.legal_actions()[0]
    app.play(app.game.legal_actions()[0])           # black (the human)
    app.play(app.game.legal_actions()[0])           # white (the computer's reply)
    app.handle_key(key(pygame.K_z, pygame.KMOD_LCTRL))
    assert app.pos == 0, "one Undo takes back the move and the computer's reply"
    print("ok: undo button, Ctrl+Z, Cmd+Z, redo")


def test_undo_in_tall_layout():
    app = make_app()
    app.set_tall(True)
    app.play(app.game.legal_actions()[0])
    app.draw()
    app.handle_click(app.strip["undo"].center)
    assert app.pos == 0, "the strip's Undo button"
    print("ok: undo button in the tall layout")


def test_gift_limit_menu_and_state():
    app = make_app()
    app.play(app.game.legal_actions()[0])
    app.set_gift_limit(2)
    assert app.rules.max_gifts == 2 and app.game.rules.max_gifts == 2 and app.game.ply == 0
    assert "gift limit 2" in app.variant_label() and not app.stats_eligible()
    labels = [it["label"] for it in app.menu_items(play.MENU_TITLES.index("Game")) if it]
    assert any(l.startswith("Gift limit") for l in labels) and any(l.startswith("Start position") for l in labels)
    app.draw()
    app.set_gift_limit(None)
    assert app.rules.standard and app.variant_label() == ""
    print("ok: Game > Gift limit")


def test_setup_editor():
    app = make_app()
    app.handle_key(key(pygame.K_b))
    assert app.setup is not None
    L = app.setup_layout()

    def pick(k):
        app.handle_click(L[k].center)

    def sq(r, c):
        x, y = app.sq_xy(r, c)
        app.handle_click((x + 50, y + 50))

    sq(5, 5)
    assert "Pick a piece" in app.setup["error"]
    pick(("w", "q"))
    sq(5, 5)
    sq(6, 6)                                    # the only queen: it moves
    assert app.setup["pieces"].get((6, 6)) == ("w", "q") and (5, 5) not in app.setup["pieces"]
    pick(("w", "p"))
    sq(0, 3)
    assert "pawn" in app.setup["error"]
    sq(4, 4)
    sq(4, 4)                                    # same piece again removes it
    assert (4, 4) not in app.setup["pieces"]
    sq(4, 3)                                    # the white king's square, queen tool
    assert "king" in app.setup["error"]
    pick("erase")
    sq(4, 3)
    assert app.setup["pieces"].get((4, 3)) == ("w", "k"), "kings cannot be erased"
    pick(("b", "r"))
    sq(2, 2)
    pick(("b", "p"))
    sq(3, 1)
    pick(("w", "p"))
    sq(4, 4)
    sq(6, 6)                                    # replaces the queen with a pawn
    pick(("w", "q"))
    sq(5, 5)
    app.draw()
    placed = dict(app.setup["pieces"])
    app.handle_key(key(pygame.K_RETURN))
    assert app.setup is None, "start refused: %s" % app.rules
    assert app.rules.start is not None and "custom start" in app.variant_label()
    on_board = {p.pos: (p.team, p.ptype) for t in "bw" for p in app.game.on_board(t)}
    assert on_board == placed and app.game.turn == "b" and not app.game.done
    app.draw()
    # undo works from a custom start (replay uses the same rules)
    app.play(app.game.legal_actions()[0])
    app.handle_key(key(pygame.K_z, pygame.KMOD_LCTRL))
    assert app.pos == 0 and sum(len(app.game.on_board(t)) for t in "bw") == len(placed)
    # editing again starts from the current setup; Standard puts the two kings back
    app.handle_key(key(pygame.K_b))
    assert app.setup["pieces"] == placed
    app.handle_click(app.setup_layout()["standard"].center)
    app.handle_key(key(pygame.K_RETURN))
    assert app.rules.start is None and app.rules.standard
    # a setup that cannot be played is refused, with a reason
    app.handle_key(key(pygame.K_b))
    for s in ((5, 0), (5, 2)):                      # two white bishops on squares of one colour
        app.handle_click(app.setup_layout()[("w", "b")].center)
        x, y = app.sq_xy(*s)
        app.handle_click((x + 50, y + 50))
    app.handle_key(key(pygame.K_RETURN))
    assert app.setup is not None and "bishops" in app.setup["error"], app.setup
    assert app.rules.standard, "a refused setup changes nothing"
    app.handle_key(key(pygame.K_ESCAPE))
    assert app.setup is None
    print("ok: custom-start editor")


def test_setup_editor_eraser_and_menus():
    app = make_app()
    app.handle_key(key(pygame.K_b))
    app.handle_click(app.setup_layout()["erase"].center)
    assert app.setup["tool"] == "erase"
    app.handle_click(app.setup_layout()["erase"].center)
    assert app.setup["tool"] is None, "the eraser toggles off when picked again"
    # the menu bar works over the editor: View entries keep it open, game-changing ones end it
    titles = app.menu_title_rects()
    app.handle_click((0, 0), raw=titles[3].center)
    assert app.menu_open == 3 and app.setup is not None
    app.handle_click((0, 0), raw=app.menu_layout()[1][0][1].center)        # View > Coordinates
    assert app.setup is not None and app.menu_open is None
    app.handle_click((0, 0), raw=titles[0].center)
    app.handle_click((0, 0), raw=app.menu_layout()[1][0][1].center)        # File > New game
    assert app.setup is None and app.menu_open is None
    # right-click closes an open menu first, not the editor's square
    app.handle_key(key(pygame.K_b))
    app.handle_click((0, 0), raw=titles[3].center)
    app.right_click()
    assert app.menu_open is None and app.setup is not None
    print("ok: custom-start editor eraser toggle and menu bar")


def test_variants_combine_in_the_app():
    app = make_app()
    app.set_gift_limit(1)
    app.handle_key(key(pygame.K_b))
    L = app.setup_layout()
    app.handle_click(L[("w", "q")].center)
    x, y = app.sq_xy(5, 5)
    app.handle_click((x + 50, y + 50))
    app.handle_click(L[("b", "r")].center)
    x, y = app.sq_xy(2, 2)
    app.handle_click((x + 50, y + 50))
    app.handle_key(key(pygame.K_RETURN))
    assert app.setup is None, app.setup and app.setup["error"]
    assert app.rules.max_gifts == 1 and app.rules.start is not None
    rng = random.Random(2)
    for _ in range(40):
        if app.game.done:
            break
        app.play(rng.choice(app.game.legal_actions()))
    assert app.game.gifts_given("b") <= 1 and app.game.gifts_given("w") <= 1
    app.draw()
    print("ok: gift limit and custom start together, in the app")


def test_settings_and_resume():
    path = os.path.join(_tmp, "settings.json")
    app = make_app(session=True)
    app.settings_path = path
    app.set_gift_limit(3)
    app.handle_key(key(pygame.K_b))
    L = app.setup_layout()
    app.handle_click(L[("w", "q")].center)
    x, y = app.sq_xy(5, 5)
    app.handle_click((x + 50, y + 50))
    app.handle_key(key(pygame.K_RETURN))
    assert app.setup is None
    rng = random.Random(4)
    for _ in range(6):
        app.play(rng.choice(app.game.legal_actions()))
    line = list(app.line)
    app.save_settings(force=True)
    # a new session: the variant comes back from the settings, and the unfinished game resumes
    app2 = make_app(session=True)
    app2.settings_path = path
    app2.load_settings(set())
    assert app2.rules.max_gifts == 3 and app2.rules.start == app.rules.start
    assert app2.resume() and app2.line == line
    # a command-line variant wins over the saved one
    app3 = make_app(Rules(max_plies=300, max_gifts=7), session=True)
    app3.settings_path = path
    app3.load_settings({"variant"})
    assert app3.rules.max_gifts == 7
    print("ok: variant remembered between sessions, game resumes")


def test_promoted_pieces_in_custom_start():
    # a promoted piece is a pawn in disguise: it sits next to the real queen, and uses up a pawn
    start = KINGS + (("w", "q", 5, 1), ("w", "q", 7, 0, True), ("b", "n", 2, 4, True), ("b", "p", 3, 1))
    assert start_problem(Rules(start=start)) is None
    g = Game(Rules(start=start))
    g.validate()
    assert g.board[7, 0].promoted and g.board[7, 0].ptype == "q" and not g.board[5, 1].promoted
    assert not g.board[7, 0].locked
    # on the board, in the text form, and in JSON (tuples become lists) it is the same position
    assert gamelog.parse_start(gamelog.format_start(Rules(start=start).start)) == Rules(start=start).start
    assert Rules(start=[list(e) for e in start]) == Rules(start=start)
    assert Rules(start=KINGS + (("w", "q", 5, 1, False),)) == Rules(start=KINGS + (("w", "q", 5, 1),))
    # native and pure-python generators agree from such a position
    for seed in range(4):
        rng = random.Random(seed)
        a, b = Game(Rules(start=start, max_plies=120), native=False), Game(Rules(start=start, max_plies=120))
        while not a.done:
            assert set(a.legal_actions()) == set(b.legal_actions()), seed
            act = rng.choice(a.legal_actions())
            a.step(act)
            b.step(act)
        a.validate()
    # pawn pool: 8 pawns in all, promoted ones included; only Q/R/B/N can be promoted
    pawns = tuple(("w", "p", 2, c) for c in range(7))
    assert start_problem(Rules(start=KINGS + pawns + (("w", "r", 5, 0, True),))) is None
    assert "pawns" in start_problem(Rules(start=KINGS + pawns + (("w", "p", 5, 0), ("w", "r", 5, 1, True))))
    assert start_problem(Rules(start=KINGS + (("w", "k", 5, 0, True),)))
    assert start_problem(Rules(start=KINGS + (("w", "p", 5, 0, True),)))
    # a promoted piece is no bishop of its own: two same-coloured ones are fine if one is promoted
    assert start_problem(Rules(start=KINGS + (("w", "b", 5, 0), ("w", "b", 5, 2, True)))) is None
    assert start_problem(Rules(start=KINGS + (("w", "b", 5, 0), ("w", "b", 5, 2))))
    for bad in ("wQ*", "wK*d4", "wP*d4", "wQ**d4"):
        try:
            gamelog.parse_start(bad)
        except ValueError:
            continue
        raise AssertionError("parse_start accepted %r" % bad)
    print("ok: promoted pieces in a custom start (engine, notation, native agrees)")


def test_setup_editor_promoted():
    app = make_app()
    app.handle_key(key(pygame.K_b))
    L = app.setup_layout()

    def pick(k):
        app.handle_click(L[k].center)

    def sq(r, c):
        x, y = app.sq_xy(r, c)
        app.handle_click((x + 50, y + 50))

    pick(("w", "q"))
    sq(5, 1)
    pick("promo")
    assert app.setup["promo"]
    sq(7, 0)                                    # a second white queen, promoted this time
    assert app.setup["pieces"][(7, 0)] == ("w", "q", True) and app.setup["error"] == ""
    sq(7, 0)                                    # same promoted piece again removes it
    assert (7, 0) not in app.setup["pieces"]
    sq(7, 0)
    pick(("w", "k"))
    sq(6, 6)                                    # kings ignore the promoted switch
    assert app.setup["pieces"][(6, 6)] == ("w", "k") and (4, 3) not in app.setup["pieces"]
    pick(("w", "p"))
    for c in range(1, 8):                       # seven plain pawns + the promoted queen = the full eight
        sq(2, c)
    assert app.setup["error"] == "" and (2, 7) in app.setup["pieces"]
    sq(1, 0)
    assert "pawns" in app.setup["error"], app.setup["error"]
    pick(("w", "r"))
    sq(2, 1)                                    # promoted rook over a pawn: the pool stays at eight
    assert app.setup["error"] == "" and app.setup["pieces"][(2, 1)] == ("w", "r", True)
    app.draw()
    placed = dict(app.setup["pieces"])
    app.handle_key(key(pygame.K_RETURN))
    assert app.setup is None, "start refused: %s" % app.rules
    on_board = {p.pos: (p.team, p.ptype) + ((True,) if p.promoted else ()) for t in "bw" for p in app.game.on_board(t)}
    assert on_board == placed, (on_board, placed)
    app.draw()
    # editing again brings the promoted pieces back; Standard clears the switch with the rest
    app.handle_key(key(pygame.K_b))
    assert app.setup["pieces"] == placed
    app.handle_click(app.setup_layout()["standard"].center)
    assert not app.setup["promo"] and app.setup["pieces"] == play.DEFAULT_START
    app.handle_key(key(pygame.K_ESCAPE))
    # save / load keeps promoted pieces
    rules, actions, _ = gamelog.parse_game(gamelog.game_dict(app.rules, [], {}))
    assert rules == app.rules
    print("ok: custom-start editor places promoted pieces")


def test_ai_does_not_crash_on_variants():
    app = make_app(Rules(max_plies=300, max_gifts=1, start=SETUP), ai_on=True)
    app.use_search = True
    for _ in range(30):
        if app.game.done:
            break
        app.ai_due = 0
        app.update_ai()
        if app.agents[app.game.turn] is None:
            app.play(random.Random(len(app.line)).choice(app.game.legal_actions()))
    print("ok: the computer keeps playing legal moves in the variants (quality: not trained, see README)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all variant tests passed")
