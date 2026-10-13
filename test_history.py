#!/usr/bin/env python
"""Headless checks of the game-history features of play.py (move list, undo, flip, save/load, review).

    python test_history.py [--shots DIR]      # --shots also writes screenshots of the window
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
_tmp = tempfile.mkdtemp()
os.environ["XDG_DATA_HOME"] = _tmp                       # saved games go to a scratch folder

import pygame  # noqa: E402

import gamelog  # noqa: E402
import numpy_net  # noqa: E402
import play  # noqa: E402
from engine import Game, Rules  # noqa: E402


def make_app(advisor, ai_on=False):
    rng = random.Random(1)
    searcher = None
    return play.App(Rules(max_plies=300), 0, rng, advisor, "", {"b": None, "w": None}, ai_on,
                    {"w"}, None, False, searcher)


def play_random(app, n, rng):
    for _ in range(n):
        if app.game.done:
            break
        app.play(rng.choice(app.game.legal_actions()))


def wait_review(app, timeout=20):
    t = time.time()
    while (app.reviewer is not None) and time.time() - t < timeout:
        app.poll_review()
        time.sleep(0.01)
    app.poll_review()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=None)
    args = ap.parse_args()
    ai = numpy_net.find_ai_file(os.path.dirname(os.path.abspath(__file__)), os.getcwd())
    advisor = numpy_net.Advisor(ai) if ai else None
    rng = random.Random(7)

    # 1. line / notation / step / jump
    app = make_app(advisor)
    play_random(app, 14, rng)
    assert len(app.line) == app.pos == 14 == len(app.texts) == len(app.turns)
    fresh = gamelog.replay(app.rules, app.line)
    assert fresh.ascii() == app.game.ascii()
    app.jump(5)
    assert app.pos == 5 and app.browsing and app.game.ply == 5
    assert gamelog.replay(app.rules, app.line, 5).ascii() == app.game.ascii()
    app.step_forward()
    assert app.pos == 6
    app.jump(10 ** 6)
    assert app.pos == 14 and not app.browsing
    print("ok: line / jump")

    # 2. undo = browse back, a new move from there replaces the rest
    app.takeback()
    assert app.pos == 13 and app.browsing                     # hot-seat: one ply
    old_tail = list(app.line[13:])
    mine = [a for a in app.game.legal_actions() if a not in old_tail]
    assert mine, "need a different move to branch with"
    app.stats_ok = True
    app.play(mine[0])
    assert len(app.line) == 14 and app.line[13] == mine[0] and not app.browsing
    assert app.stats_ok is False                              # a branch is not a clean game
    print("ok: undo / branch")

    # 3. undo against the computer takes back the reply too
    app = make_app(advisor, ai_on=True)                       # AI plays white, human black
    app.play(app.game.legal_actions()[0])                     # human (black) moves
    app.ai_due = 0
    app.update_ai()                                           # white replies (policy, no search)
    app.update_ai()
    n = len(app.line)
    assert n >= 2, n
    humans_turn_at = [i for i in range(n) if app.agents[app.turns[i]] is None]
    app.takeback()
    assert app.pos == max(i for i in humans_turn_at if i < n) if n % 2 == 0 else True
    assert app.agents[app.game.turn] is None or app.pos == 0
    app.update_ai()                                           # browsing: the computer must wait
    assert len(app.line) == n
    print("ok: undo vs computer")

    # 4. save / load round trip, loaded game can be continued
    app = make_app(advisor)
    play_random(app, 25, rng)
    app.save()
    assert app.message.startswith("Saved "), app.message
    saved = gamelog.saved_games()
    assert len(saved) == 1 and saved[0].startswith(_tmp)
    line = list(app.line)
    app2 = make_app(advisor)
    app2.load()
    assert app2.line == line and app2.pos == len(line) and app2.stats_ok is False
    assert app2.texts == app.texts and app2.turns == app.turns
    assert app2.game.ascii() == app.game.ascii()
    if not app2.game.done:
        app2.play(app2.game.legal_actions()[0])
        assert len(app2.line) == len(line) + 1
    app2.load("/nonexistent/file.json")
    assert app2.message.startswith("Could not load")
    bad = os.path.join(_tmp, "bad.json")
    open(bad, "w").write('{"format": 1, "rules": {}, "actions": [{"src": [0, 0], "dst": [5, 5]}]}')
    app2.load(bad)
    assert app2.message.startswith("Could not load"), app2.message
    print("ok: save / load")

    # 5. review
    if advisor is not None:
        app3 = make_app(advisor)
        app3.load(saved[0])
        wait_review(app3)
        assert app3.review is not None, app3.message
        assert len(app3.review["marks"]) == len(line) and len(app3.review["values"]) == len(line) + 1
        flagged = [i for i, m in enumerate(app3.review["marks"]) if m]
        print("review marks:", [(i + 1, app3.review["marks"][i]) for i in flagged][:8])
        if flagged:
            app3.jump(0)
            app3.next_mistake()
            assert app3.pos == next(i for i in flagged if i > 0), (app3.pos, flagged)
            app3.draw()                                       # arrows + status text must not crash
            app3.next_mistake(-1) if app3.pos else None
        # a branch keeps the still-valid part of the review
        app3.jump(10)
        mine = [a for a in app3.game.legal_actions() if a != app3.line[10]]
        app3.play(mine[0])
        assert len(app3.review["marks"]) == 10 and len(app3.review["values"]) == 11
        print("ok: review")

    # 6. flip geometry round trip, clicks while flipped
    app = make_app(advisor)
    for flipped in (False, True):
        app.flipped = flipped
        for r in range(8):
            for c in range(8):
                x, y = app.sq_xy(r, c)
                assert app.xy_sq(x + 5, y + 5) == (r, c)
    app.flipped = True
    src = next(iter(app.by_src))
    x, y = app.sq_xy(*src)
    app.handle_click((x + 10, y + 10))
    assert app.selected == src
    print("ok: flip")

    # 7. every draw path runs, optionally write screenshots
    app = make_app(advisor)
    app.load(saved[0]); wait_review(app)
    for flipped in (False, True):
        app.flipped = flipped
        for n in (0, len(app.line) // 2, len(app.line)):
            app.jump(n)
            app.draw()
            if args.shots:
                os.makedirs(args.shots, exist_ok=True)
                pygame.image.save(app.screen, os.path.join(args.shots, "shot-%s-%d.png" % ("flip" if flipped else "norm", n)))
    # a brand-new game (empty move list) and the start of a long one must draw correctly
    fresh = make_app(advisor)
    fresh.draw()
    assert fresh.moves_first_row() == 0
    app.jump(0)
    assert app.moves_first_row() == 0
    app.draw()
    # mouse wheel / row click / graph click
    L = app.layout
    app.scroll_moves(-2); app.draw()
    app.handle_click((L['m_list'].x + 20, L['m_list'].y + 5))
    app.handle_click((L['m_graph'].centerx, L['m_graph'].centery))
    app.handle_click(L['m_undo'].center); app.handle_click(L['m_flip'].center)
    app.handle_click(L['m_save'].center); app.handle_click(L['m_load'].center)
    app.draw()
    print("ok: drawing / clicks")
    shutil.rmtree(_tmp, ignore_errors=True)
    print("all history tests passed")


if __name__ == "__main__":
    main()
