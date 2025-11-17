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

## Contributing

Contributions, issues, and ideas for new (and worse) features are welcome!

## License

*GPL-3.0 license*