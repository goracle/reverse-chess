#!/usr/bin/env python
"""Headless checks of the play-feel QoL features of play.py: coordinates, menu keys / Shift-confirm / gift
preview, premove, think-time slider, sound, window scaling, network picker.

    python test_qol_play.py
"""
import os
import random
import shutil
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
_tmp = tempfile.mkdtemp()
os.environ["XDG_DATA_HOME"] = _tmp

import pygame  # noqa: E402

import mcts  # noqa: E402
import numpy_net  # noqa: E402
import play  # noqa: E402
from engine import Rules  # noqa: E402


def make_app(advisor=None, ai_on=False, sides=("w",), **kw):
    searcher = mcts.Searcher(advisor, sims=40, think_ms=500) if advisor is not None else None
    return play.App(Rules(max_plies=300), 0, random.Random(1), advisor, "", {"b": None, "w": None},
                    ai_on, set(sides), None, False, searcher, **kw)


def click_move(app, action, force=False):
    app.handle_click(app.sq_centre(action.src))
    app.handle_click(app.sq_centre(action.dst), force)


def find_menu_move(app, rng, tries=400):
    """Play random moves until some src -> dst has several variants; returns that candidate list."""
    for _ in range(tries):
        if app.game.done:
            return None
        for src, actions in app.by_src.items():
            for dst in {a.dst for a in actions}:
                group = [a for a in actions if a.dst == dst]
                if play.next_choice(group) is not None:
                    return group
        app.play(rng.choice(app.game.legal_actions()))
    return None


def main():
    ai = numpy_net.find_ai_file(os.path.dirname(os.path.abspath(__file__)), os.getcwd())
    advisor = numpy_net.Advisor(ai) if ai else None
    rng = random.Random(3)

    # 1. coordinates: drawing works both ways round, on and off
    app = make_app()
    for flipped in (False, True):
        app.flipped = flipped
        for coords in (True, False):
            app.coords = coords
            app.draw()
    app.toggle_coords()
    assert app.coords is True
    print("ok: coordinates")

    # 2. single-variant move plays at once; Shift asks for confirmation first
    app = make_app()
    action = next(a for a in app.game.legal_actions() if len(app.by_src[a.src]) and
                  len([b for b in app.by_src[a.src] if b.dst == a.dst]) == 1)
    click_move(app, action)
    assert app.line == [action] and app.pending is None
    app = make_app()
    click_move(app, action, force=True)
    assert app.pending is not None and app.pending[0] == "confirm" and not app.line
    app.draw()
    app.menu_key(pygame.K_1)
    assert app.line == [action] and app.pending is None
    app = make_app()
    click_move(app, action, force=True)
    app.escape()
    assert app.pending is None and not app.line
    print("ok: quick play / Shift confirm")

    # 3. menu keys, highlight, gift preview
    app = make_app()
    group = find_menu_move(app, rng)
    if group is None:
        print("no menu move found in the random game: skipped the menu checks")
    else:
        app.selected, app.candidates, app.history = group[0].src, group, []
        app.resolve()
        assert app.pending is not None
        dim, values = app.pending
        panel = play.menu_rects(len(values), False)[0]
        assert panel.left >= play.BOARD_PX and panel.right <= play.WIN_W and panel.top >= 0 and panel.bottom <= play.WIN_H
        app.draw()
        app.menu_key(pygame.K_DOWN)
        assert app.menu_sel == 0
        app.menu_key(pygame.K_UP)
        assert app.menu_sel == len(values) - 1
        if dim == "gift":
            i = next((i for i, v in enumerate(values) if v is not None), None)
            if i is not None:
                app.menu_sel = i
                preview = app.menu_preview_action()
                assert preview is not None and preview.gift == values[i]
                app.draw()
        before = len(app.line)
        app.menu_key(pygame.K_1)                      # answers the first question
        if app.pending is None:
            assert len(app.line) == before + 1
        else:
            assert app.history                        # a second question: Backspace goes back
            app.menu_key(pygame.K_BACKSPACE)
            assert not app.history
            app.menu_key(pygame.K_BACKSPACE)          # no history: closes the menu
            assert app.pending is None
        print("ok: menu keys / preview (%s)" % dim)

    # 4. premove
    app = make_app(ai_on=True, sides=("b",))          # you are white; black (the computer) moves first
    assert app.flipped is False
    black = app.game.on_board("w")[0].pos             # (the name is from when this was written the other way round)
    empty = next((r, c) for r in range(8) for c in range(8) if app.game.board[r, c] is None)
    app.handle_click(app.sq_centre(black))
    assert app.pre_src == black and app.selected is None
    app.handle_click(app.sq_centre(empty))
    assert app.premove == (black, empty)
    app.draw()
    app.right_click()
    assert app.premove is None and app.pre_src is None
    app.play(app.game.legal_actions()[0])             # stands in for the computer's reply
    reply = app.by_src[next(iter(app.by_src))][0]
    app.premove = (reply.src, reply.dst)
    n = len(app.line)
    app.update_ai()
    assert app.premove is None
    assert len(app.line) == n + 1 or app.pending is not None
    app.premove = (black, empty)                      # not legal for black now: dropped with a message
    app.cancel()
    app.update_ai()
    assert app.premove is None
    print("ok: premove")

    # 4b. the board turns so that your colour is at the bottom
    assert make_app(ai_on=True, sides=("w",)).flipped is True       # you are black
    assert make_app(ai_on=True, sides=("b",)).flipped is False      # you are white
    assert make_app(ai_on=False).flipped is False                   # two humans: left alone
    assert make_app(ai_on=True, sides=("w",), auto_orient=False).flipped is False
    app = make_app(ai_on=True, sides=("w",))
    app.set_sides({"b"})
    assert app.flipped is False
    app.flip()
    assert app.flipped is True
    print("ok: orientation")

    # 5. window scale: resizing keeps the mouse mapping; frames present
    app = make_app()
    app.apply_scale(0.5)
    app.draw()
    app.present()
    assert abs(app.scale - 0.5) < 0.01
    cx = (app.view.x + app.view.w // 2, app.view.y + app.view.h // 2)
    lx, ly = app.to_logical(cx)
    assert abs(lx - play.WIN_W // 2) <= 3 and abs(ly - play.WIN_H // 2) <= 3
    app.zoom(1.1)
    app.apply_scale(1.0)
    app.present()
    assert app.view.size == (play.WIN_W, play.WIN_H) and app.to_logical((10, 20)) == (10, 20)
    print("ok: window scale")

    # 6. sound toggles without raising (no audio device is fine)
    app = make_app()
    app.toggle_sound()
    app.toggle_sound()
    app.sound_on = True
    app.play(app.game.legal_actions()[0])
    print("ok: sound")

    if advisor is None:
        print("no AI file: skipped the think-time and network-picker checks")
        return

    # 6b. the review shows the network's view of every position, the opening one included
    app = make_app(advisor)
    play_rng = random.Random(9)
    for _ in range(10):
        app.play(play_rng.choice(app.game.legal_actions()))
    app.start_review()
    import time
    t = time.time()
    while app.reviewer is not None and time.time() - t < 120:
        app.poll_review()
        time.sleep(0.01)
    assert app.review is not None
    app.jump(0)
    assert app.review_here() and app.review_alts() is not None
    app.draw()
    app.jump(len(app.line))
    assert not app.review_here()
    print("ok: review of the first move")

    # 6c. the move list has a start row: clicking it shows the position before the first move
    app.jump(len(app.line))
    box = app.layout["m_list"]
    app.moves_top = 0
    app.draw()
    app.handle_click((box.x + 20, box.y + 5))
    assert app.pos == 0
    app.handle_click((box.x + 20, box.y + 5 + play.ROW_H))     # the first move's row: the position after it
    assert app.pos == 1
    app.moves_top = None
    app.draw()
    print("ok: start row")

    # 7. think-time slider
    app = make_app(advisor)
    track = app.think_track()
    app.set_think_from_x(track.x - 50)
    assert app.think_ms == play.THINK_STEPS[0] and app.searcher.think_ms == play.THINK_STEPS[0]
    app.set_think_from_x(track.right + 50)
    assert app.think_ms == play.THINK_STEPS[-1] and app.searcher.sims > 40
    app.click_panel((app.layout["m_think"].x + 20, app.layout["m_think"].centery))
    assert app.dragging == "think"
    app.draw()
    print("ok: think time")

    # 8. network picker (a copy of the AI file, a snapshot-named dummy, and a damaged file)
    nets = os.path.join(_tmp, "nets")
    os.makedirs(nets)
    if ai.endswith(".npz"):
        shutil.copy(ai, os.path.join(nets, "copy.npz"))
    for name in ("iter_00005.pt", "iter_00012.pt"):
        with open(os.path.join(nets, name), "wb") as fh:
            fh.write(b"not a checkpoint")
    items = play.find_nets([_tmp])
    versions = [it["version"] for it in items if "iter_" in it["rel"]]
    assert versions == sorted(versions, reverse=True) and 12 in versions
    app = make_app(advisor, net_dirs=[_tmp])
    app.open_picker()
    assert app.picker is not None
    app.draw()
    app.picker_scroll(-1)
    app.picker_key(pygame.K_DOWN)
    app.picker_key(pygame.K_END)
    app.picker_key(pygame.K_HOME)
    bad = next(i for i, it in enumerate(app.picker["items"]) if it["rel"].endswith("iter_00012.pt"))
    app.picker_choose(bad)
    assert app.picker is None and app.message.startswith("Could not load") and app.advisor is advisor
    good = [it for it in play.find_nets([_tmp]) if it["rel"].endswith("copy.npz")]
    if good:
        app.play(app.game.legal_actions()[0])
        app.open_picker()
        app.picker_choose(next(i for i, it in enumerate(app.picker["items"]) if it["rel"].endswith("copy.npz")))
        assert app.advisor is not advisor and app.searcher.advisor is app.advisor
        assert app.agents["w"] is app.default_ai or not app.ai_on
        assert app.stats_ok is False or app.stats is None
        app.draw()
    app.escape()
    print("ok: network picker")


if __name__ == "__main__":
    main()
