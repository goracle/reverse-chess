"""play.py's window: the game is fitted and centred in whatever size the window has, and nothing is asked of the
window manager while it is resized.  Run with a display (or under Xvfb):  python test_window.py"""
import os, random, sys
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame, play
from engine import Rules


def check(cond, msg):
    print(("ok: " if cond else "FAIL: ") + msg)
    if not cond:
        sys.exit(1)


app = play.App(Rules(), 0, random.Random(1), None, "", {'b': None, 'w': None}, False, set(), None, False, sound=False)
for _ in range(3):
    app.draw(); app.present()
for ww, ah in ((1086, 884), (1920, 1000), (700, 900), (2400, 700)):
    app.compute_regions(ww, ah)
    src, (dx, dy), sc = app.regions[0]
    w, h = round(src.w * sc), round(src.h * sc)
    check(w <= ww + 1 and h <= ah + 1, "fits %dx%d" % (ww, ah))
    check(abs(dx - (ww - w) / 2) <= 1 and abs(dy - play.MENU_H - (ah - h) / 2) <= 1, "centred in %dx%d" % (ww, ah))
before = app.requested.copy()
app.on_resize((1500, 900)); app.draw(); app.present()
check(app.size_wanted is None, "a resize makes no request of the window manager")
x, y = app.sq_centre((3, 4))[:2]
src, (dx, dy), sc = app.regions[0]
back = app.to_logical((dx + (x - src.x) * sc, dy + (y - src.y) * sc))
check(abs(back[0] - x) <= 2 and abs(back[1] - y) <= 2, "pointer maps back after centring")
print("all window tests passed")
