# PyInstaller spec: builds a self-contained "ReverseChess" folder (no Python needed to run it).
#   pyinstaller --noconfirm reverse_chess.spec
# Expects, next to this file: ai.npz, and the native library built by ./build_native.sh
# (librc_engine.so / librc_engine.dylib / rc_engine.dll).  The native library is optional:
# without it the game falls back to the pure-Python engine (slower search).
import glob
import os
import sys

here = os.path.dirname(os.path.abspath(SPEC))

datas = [(os.path.join(here, "chessimages"), "chessimages")]
ai = os.path.join(here, "ai.npz")
if not os.path.exists(ai):
    raise SystemExit("ai.npz not found next to reverse_chess.spec")
datas.append((ai, "."))
if os.path.isfile(os.path.join(here, "nets", "tiers.json")):      # the Game > Difficulty levels made by tiers.py
    datas.append((os.path.join(here, "nets"), "nets"))

binaries = [(p, ".") for pat in ("librc_engine.so", "librc_engine.dylib", "rc_engine.dll")
            for p in glob.glob(os.path.join(here, pat))]
if not binaries:
    print("WARNING: no native engine library found; the build will use the slow pure-Python engine")

a = Analysis(
    [os.path.join(here, "play.py")],
    pathex=[here],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    excludes=["torch", "tkinter", "pygad", "pyautogui", "matplotlib", "scipy", "pandas"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ReverseChess",
          console=(os.environ.get("RC_CONSOLE") == "1"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="ReverseChess", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="ReverseChess.app", icon=None,
                 bundle_identifier="com.goracle.reversechess")
