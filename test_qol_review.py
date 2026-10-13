#!/usr/bin/env python
"""Headless checks of the review / input QoL features of play.py: variation stack, search-backed review,
top-3 alternatives, paste / drop, session autosave and resume.

    python test_qol_review.py
"""
import os
import random
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
_tmp = tempfile.mkdtemp()
os.environ["XDG_DATA_HOME"] = _tmp                       # games and the autosave go to a scratch folder

import gamelog  # noqa: E402
import numpy_net  # noqa: E402
import play  # noqa: E402
import review as reviewlib  # noqa: E402
from engine import Rules  # noqa: E402


def make_app(advisor, **kw):
    return play.App(Rules(max_plies=300), 0, random.Random(1), advisor, "", {"b": None, "w": None},
                    False, {"w"}, None, False, None, **kw)


def play_random(app, n, rng):
    for _ in range(n):
        if app.game.done:
            break
        app.play(rng.choice(app.game.legal_actions()))


def wait_review(app, timeout=60):
    t = time.time()
    while app.reviewer is not None and time.time() - t < timeout:
        app.poll_review()
        time.sleep(0.01)
    app.poll_review()


def main():
    ai = numpy_net.find_ai_file(os.path.dirname(os.path.abspath(__file__)), os.getcwd())
    advisor = numpy_net.Advisor(ai) if ai else None
    rng = random.Random(11)

    # 1. variations: a different move keeps the old line; Tab swaps; the same move keeps the tail
    app = make_app(advisor)
    play_random(app, 12, rng)
    full = list(app.line)
    app.jump(6)
    same = full[6]
    app.play(same)                                            # the move already there: just steps on
    assert app.line == full and app.pos == 7 and not app.variations
    app.jump(6)
    other = [a for a in app.game.legal_actions() if a != same][0]
    app.play(other)
    assert len(app.line) == 7 and app.line[6] == other and len(app.variations) == 1
    assert app.variations[0]["line"] == full and app.variations[0]["branch"] == 6
    mine = list(app.line)
    app.swap_variation()
    assert app.line == full and app.pos == 6 and app.variations[0]["line"] == mine
    assert len(app.texts) == len(app.turns) == len(full)
    app.swap_variation()
    assert app.line == mine and app.variations[0]["line"] == full
    app.takeback()
    app.play(other)                                           # same move as the line's next one
    assert app.line == mine
    print("ok: variations")

    # 2. paste: JSON and move list round trip; bad text is refused with a message
    text = gamelog.game_json(app.rules, app.line)
    app.paste_text(text)
    assert app.line == mine and app.message.startswith("Pasted")
    app.paste_text(gamelog.move_list_text(app.turns, app.texts))
    assert app.line == mine
    app.paste_text("this is not a game")
    assert app.message.startswith("Could not paste") and app.line == mine
    print("ok: paste")

    # 3. session autosave / resume
    session = make_app(advisor, session=True)
    play_random(session, 9, rng)
    session.jump(4)
    session.play([a for a in session.game.legal_actions() if a != session.line[4]][0])
    saved_line, saved_vars = list(session.line), [v["line"] for v in session.variations]
    assert os.path.exists(gamelog.autosave_path())
    fresh = make_app(advisor, session=True)
    assert fresh.resume() and fresh.line == saved_line
    assert [v["line"] for v in fresh.variations] == saved_vars and fresh.stats_ok is False
    fresh.restart()
    assert not os.path.exists(gamelog.autosave_path())
    assert make_app(advisor, session=True).resume() is False
    print("ok: autosave / resume")

    if advisor is None:
        print("no AI file: skipped the review checks")
        return

    # 4. review: network-only and search-backed, top three, graph / chip helpers
    for sims in (0, 8):
        app = make_app(advisor, review_sims=sims)
        play_random(app, 16, random.Random(5))
        app.start_review()
        wait_review(app)
        rv = app.review
        assert rv is not None and len(rv["values"]) == len(app.line) + 1 == len(rv["top"]) + 1
        assert rv["sims"] == sims
        for i, mark in enumerate(rv["marks"]):
            assert len(rv["top"][i]) >= 1
            if mark and sims == 0:
                assert all(v is not None for _, _, v in rv["top"][i])      # valued where it matters
        marked = [i for i, m in enumerate(rv["marks"]) if m]
        if marked:
            app.jump(marked[0])
            alts = app.review_alts()
            assert alts is not None and 1 <= len(alts["rows"]) <= 3 and alts["played"].startswith("Played")
            assert "lost" in app.mark_text(marked[0])
            x = int(app.graph_x(marked[0] + 1))
            g = app.layout["m_graph"]
            assert app.graph_mark_at((x, g.centery)) is not None
        app.draw()                                                        # nothing raises while drawing
        print("ok: review (sims=%d, %d marked)" % (sims, len(marked)))

    # 5. branching keeps the review of the old line
    app.jump(3)
    app.play([a for a in app.game.legal_actions() if a != app.line[3]][0])
    assert app.variations and app.variations[0]["review"] is not None
    app.swap_variation()
    assert app.review is not None and len(app.review["values"]) == len(app.line) + 1
    print("ok: variation keeps its review")


if __name__ == "__main__":
    main()
