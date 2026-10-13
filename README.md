# Reverse Chess

> "Play chess in reverse! It's worse!"

Reverse Chess is a mind-bending variant where the entire game is played backward. The goal isn't to checkmate the opponent, but to return all of your pieces to their original starting squares.

Instead of capturing, you "un-capture" by **gifting** pieces back to your opponent. Instead of promoting, you **un-promote**. All the logic you know is flipped, forcing you to think in a completely new way (or in the same way but backwards!).

## Key Features

* **A Fully Playable Game:** Play locally against a friend in "hot-seat" mode.
* **AI Opponent:** Challenge an AI powered by **randomness**.  It moves at random!
* **GUI:** Built with `pygame` for the main game board and `tkinter` for the pop-up piece-selection menus.
* **Unique Mechanics:** Explore strange new strategies involving reverse en-passant, pawn un-promotion, and careful gifting to avoid blocking your own pieces.

## Getting Started

### Prerequisites

* Python 3.x
* Git

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/goracle/reverse-chess.git](https://github.com/goracle/reverse-chess.git)
    cd reverse-chess
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # On Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `pygad` library is only needed if you plan to run in `GENETICS = True` mode ).*

4.  **Run the game!**
    ```bash
    python3 reverse_chess.py
    ```

## How to Play

The rules are simple...ish.

* **The Goal:** The first player to get all of their pieces back to their original starting positions (Rank 1 & 2 for White, Rank 7 & 8 for Black) wins the game.  You can also automatically win if your opponent accidentally permanently blocks their own pieces from returning home.
* **Movement:** All pieces move according to their standard chess rules, but with the goal of moving *backward* toward their home rank.
* **"Gifting" (Un-Capturing):** You cannot land on a friendly or enemy piece. Instead, you can "gift" a piece from your opponent's "captured" pile back onto the board.
* **Pawns:**
    * Pawns move one square *backward* (e.g., White pawns move from rank 3 to 2).
    * They "un-capture" diagonally backward, which requires you to *simultaneously* gift the "captured" piece back to an adjacent square.
    * If a promoted piece (e.g., a Queen) reaches its original pawn rank, it can be "un-promoted" back into a pawn.

* Blind Alley Rules - The game is programmed to avoid (some) common blind alleys, which are immediate impossible situations you could place your opponent in.  It will not let you do that, usually.  For instance, you cannot give your opponent a column of pawns such that they automatically lose.

* Strategy - You are trying to force your opponent to make a mistake and trap themselves (leading to a loss) or to clutter the board.

## Game history, undo, save/load and review

`play.py` has a move list (third column) with the usual conveniences:

* **Step / jump:** Left / Right, Home / End, or click a move (the top row, "start position", is the position
  before the first move).  Mouse wheel scrolls the list.
* **Undo:** `Z` / Backspace takes back your move (and the computer's reply).  Right / End bring it back;
  playing a different move starts another line and the old one is kept as a variation (see below).
  A game with a take-back is not counted in your record.
* **Flip board:** with one human side the board is turned by itself so that your colour is at the bottom
  (you always move up); `F` flips it by hand.  `--fixed-board` stops the automatic turn, `--flip` also
  starts from black's side and keeps it.
* **Save / load:** `Ctrl+S` / `Ctrl+O` (or the buttons); `--load FILE`; `--autosave` saves every finished game.
  Files are small JSON in your user data folder (`ReverseChess/games/`).
* **Review:** `V`.  The network rates every position and marks inaccuracies (`?!`), mistakes (`?`) and
  blunders (`??`) in the move list, and graphs the game.  `M` jumps to the position before the next marked
  move and draws the network's preferred move (green) against the one played (red) -- so the "puzzles" come
  from your own games, nothing to author.  `python review.py [game.json]` prints the same in the terminal.
  Browsing any position of a reviewed line (the opening position included) shows the network's choice
  there in green and the move played in grey (red if it was marked), with its top three in the panel.
  The gift a move gives is drawn as a translucent piece on the square it lands on (hints too).
  `Shift+M` / `Ctrl+Left` jump to the previous marked move, `Ctrl+Right` to the next; `P` plays the
  network's move instead of the one played there (the old line is kept, see Variations).  Under the
  graph: each side's average loss per move and how many `?!` / `?` / `??` it made.
  With `--review-sims N` (default 30; `review.py --sims N` likewise) every position is searched, so a move
  is only marked when the *search* sees the value drop; `0` is the bare network.  At a marked position the
  panel lists the top three moves with their share (search visits, or network probability) and value.  Each
  move-list row has a small eval bar (white's share after the move; hover for the number); hover a dot on
  the graph for the loss, click it to jump to the position before that move.
* **Variations:** a different move from an earlier position keeps the line it cut off, with its review.
  `Tab` swaps to the most recent one (`Shift+Tab`: the oldest), showing where the lines part; the line you
  leave is kept, so `Tab` again comes back.  Up to 12 are kept per game.  Playing the move that was already
  next in the line just steps forward.
* **Copy:** `Ctrl+C` puts the whole move list on the clipboard (one move per line), `Ctrl+Shift+C` the game as
  JSON (what Save writes).  Uses pygame's clipboard, then `wl-copy` / `xclip` / `xsel` if that fails.
* **Paste / drop:** `Ctrl+V` opens a game from the clipboard (that JSON, or a move list as copied), and a `.json`
  game file dropped on the window opens it (`wl-paste` / `xclip` / `xsel` as fallbacks).
* **Resume:** the game in progress is written to `ReverseChess/autosave.json` after every move (side lines
  included) and reopened at the next start if it was not over.  `R` discards it; `--no-resume` turns it off.

## Play feel and window

* **Coordinates:** rank numbers and file letters on the board edge, following the flip.  `C` or the Coords
  button toggles them; `--no-coords` starts without.
* **Move menu from the keyboard:** `1`-`9` pick an option, `Up` / `Down` + `Enter` pick the highlighted one,
  `Backspace` goes back.  A move with a single variant plays at once; hold `Shift` while clicking the
  destination to get a "Play this move?" confirmation anyway.
* **Gift preview:** hovering (or highlighting) a menu option ghosts the piece it gives back on the square
  where it would land.  The menu opens to the right of the board, so the board stays fully in view.
* **Premove:** on the computer's turn click one of your pieces, then where it should go; it is played as soon
  as it is your turn if still legal (a move that needs a decision opens its menu).  Right-click or `Esc`
  cancels.
* **AI think time:** the slider in the move column sets how long the search may think (100 ms - 20 s); the
  simulation cap scales with it.  Needs MCTS on.
* **Sound:** short synthesised move / gift / game-end sounds; `U` or the Sound button toggles, `--no-sound`
  starts without.
* **Window size:** the window is resizable and defaults to what fits the screen; `--scale F`, `Ctrl +` /
  `Ctrl -`, `Ctrl+0` (fit the screen), `Ctrl+1` (100%).
* **Network picker:** `N` (or click the AI's name in the panel) lists the network files next to `play.py`, in
  `runs/*/` (including `iter_NNNNN.pt` snapshots) and `nets/`, plus any `--net-dir DIR`.  Choosing one makes
  it the AI (play, search, eval, hints); the current review is dropped and a game under way stops counting in
  your record.  `.pt` files need PyTorch; `export_weights.py` makes an `.npz` that does not.

## Playing against the AI

`play.py` is the pygame front end.  The computer opponent is the network trained by `train.py`:
copy its checkpoint (e.g. `runs/run4/latest.pt`) next to `play.py`, then

```bash
python play.py                  # you are black, the AI is white
python play.py --ai-color b     # you are white
python play.py --no-ai          # hot-seat; the AI can be switched on in the window
```

The AI looks ahead with a Monte-Carlo tree search (`mcts.py`) by default: every legal move is checked
exactly for an immediate win/loss, then the network's policy and value guide a PUCT search that runs
in the background while the window stays responsive.  `--sims N` / `--think-ms MS` set how hard it
thinks per move, `--no-search` (or the **MCTS** button / `S`) switches back to the plain network.

In the window: **AI on/off** and which colour it plays, an **eval bar** (the network's opinion of who
is ahead, white at the bottom), a **Hint** button (its three favourite moves), and a **record** of your
games against it.  The move menu has a **Back** button and an **X**.  Keys: `H` hint, `E` eval bar,
`A` AI on/off, `R` new game, `Esc` back out.

A `.pt` checkpoint needs PyTorch.  To play (or package the game) without it, convert once:

```bash
python export_weights.py runs/run4/latest.pt        # writes ai.npz (NumPy only, ~2 MB)
python check_numpy_net.py runs/run4/latest.pt       # optional: NumPy output == PyTorch output
```

The record is kept in your user data folder (see `stats.py`), not in the game folder.

## Contributing

Contributions, issues, and ideas for new (and worse) features are welcome!

## License

*GPL-3.0 license*

## Native engine (optional, much faster self-play)

`native/` is a dependency-free Rust port of the engine's move generation.
`./build_native.sh` builds it and puts `librc_engine.so` next to `engine.py`;
`engine.py` then uses it automatically (pure Python is the fallback).

* `python check_native.py` - plays random games on both backends in lockstep and
  fails on the first difference, then prints plies/second for each.
* `RC_ENGINE=py` forces pure Python, `RC_ENGINE=native` makes a missing library an error.

## Releases (Windows / macOS / Linux downloads)

Pushing a tag builds and publishes everything via GitHub Actions (`.github/workflows/release.yml`):

```bash
git tag v0.1.0 && git push origin v0.1.0
```

Each platform is built on its own runner (Rust engine + PyInstaller), checked against the pure-Python
reference engine, smoke-tested with `play.py --selftest`, then attached to the release.  `ai.npz` is
committed to the repo and bundled into every download.  To test without publishing, use
**Actions -> build-release -> Run workflow**; the zips appear as run artifacts.

To build locally: `sh build_native.sh && pip install pyinstaller && pyinstaller --noconfirm reverse_chess.spec`.
