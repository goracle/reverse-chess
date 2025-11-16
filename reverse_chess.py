#!/usr/bin/python3
"""play chess in reverse"""
from pyautogui import prompt
import time
import sys
import numpy as np


# pop-up box
from tkinter import *
from tkinter import messagebox
import tkinter
Tk().wm_withdraw() #to hide the main window

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


# modules
from givebox import givebox
from chessassets import IMAGES, PTYPES, offboard_count, onboard_count
from tenants import tenant_matching # check this

COMPUTER_ONLY = True # computer plays itself
COMPUTER_ONLY = False

ONLY_HUMAN = True # human v. human (on the same computer, no network play); useful for debugging
ONLY_HUMAN = False

DRAW = False
DRAW = True

GENETICS = True
GENETICS = False
if GENETICS:
    import pygad

if COMPUTER_ONLY:
    DRAW = False
    DRAW = True

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


# derived from:
# https://levelup.gitconnected.com/chess-python-ca4532c7f5a4

# TODO:
# features:
# menu (random ai) # https://pygame-menu.readthedocs.io/en/4.3.6/
# custom starting config
# show promoted on the icon

# bugs:
# highlighting is busted
# does the computer demote?
# two times it says legal when the move is not legal, which could cause a problem:
## pawn un-taking a piece, which then results in no legal moves
## general movement which results in no legal moves.
## general gifts which result in no legal moves
### In general the game assumes all your moves which are legal do not cause your opponent to have no legal moves.  this just causes more draws, since the only time this mis-count causes a problem is if it results in no legal moves for you, which results in a draw.  The general solution to this problem is a proof by construction that the other team can get their pieces home from the proposed board configuration.  this seems to imply we would need to search the move tree deeper than may be feasible.  The point of the rule is to prevent games with small numbers of moves.  Given that draws resulting from this case seem to be small in number, it does not currently seem worth the trouble of implementing such a search.  Even a small depth search of the move tree would probably decrease the number of draws a lot, so this may be added at some point depending on player feedback.



global BOARD
global SNAP
global REVERSE_CHECK
REVERSE_CHECK = None
BOARD = [[None for i in range(8)] for j in range(8)]
BOARD = np.array(BOARD)
SNAP = None
WIDTH = 800
NUM_GENES = 100

WIN = pygame.display.set_mode((WIDTH, WIDTH))

pygame.display.set_caption('Reverse Chess:  "It\'s worse"')
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
GREEN = (50, 164, 49)


## Creates a chess piece class that shows what team a piece is on, what type of piece it is and whether or not it can be killed by another selected piece.
class Piece:
    def __init__(self, team, ptype, image, home, ontheboard=False):
        """init"""
        # params used to check for draws:
        self.ptype = ptype
        self.team = team
        self.locked = False
        self.promoted = False
        self.sq_color = None

        # internal parameters
        self.attribute_str = None
        self.ontheboard = ontheboard
        #self.pre_reverse_check = False
        #self.reverse_check = False
        self.image = image
        self.pos = None
        self.image = None
        self.x = None
        self.y = None
        self.update_image()
        self.home = home

    @PROFILE
    def attributes(self):
        """get a string describing the piece"""
        attributes = [self.team, self.ptype, self.promoted, self.locked, self.sq_color]
        encode_str = ''
        for k in attributes:
            encode_str += '@'+str(k)+'@'
        self.attribute_str = encode_str
        return self.attribute_str

    def square_color(self):
        """set the square color (bishop identity)"""
        assert self.pos is not None
        assert self.ontheboard
        self.sq_color = square_color(self.pos)
        self.attributes()
        return self.sq_color

    def ident(self):
        """print identifying information"""
        retstr = ('id:', self.team, self.ptype, self.pos, self.promoted, self.locked)
        print(retstr)
        return retstr

    def is_home(self):
        """check if piece is home"""
        row, col = self.pos
        homerows, homecols = self.home
        ret = row in homerows and col in homecols
        chk = not self.promoted
        ret = ret and chk
        if not ret:
            self.unlock()
        return ret

    def move_home(self):
        """sends the piece home (which should be unambiguous)"""
        assert self.ptype in ('k', 'r'), self.ptype
        assert self.ontheboard, "do not send pieces home from the void"
        row, col = self.home
        row = row[0]
        col = col[0]
        newpos = (row, col)
        self.update(newpos)

    def lock(self):
        """prevent the piece from moving"""
        assert self.ontheboard, "sanity check"
        assert not self.promoted, "sanity check"
        self.locked = True
        self.attributes()

    def unlock(self):
        """prevent the piece from moving"""
        assert self.ontheboard, "sanity check"
        self.locked = False
        self.attributes()

    def update_image(self, promoted=False):
        """update image to reflect ptype"""
        im1 = IMAGES[(self.team, self.ptype, promoted)]
        self.image = pygame.image.load(im1)

    def demote(self):
        """change promoted pawn back to pawn"""
        if self.promoted:
            self.ptype = 'p'
            self.update_image()
            self.promoted = False
            self.update_image()
            self.attributes()

    def promote(self, ptype_new):
        """change promoted pawn back to pawn"""
        if not self.promoted:
            assert not ptype_new == 'p', ptype_new
            assert ptype_new != self.ptype, ptype_new
            self.ptype = ptype_new
            self.update_image()
            self.promoted = True
            self.update_image(True)
            self.unlock()
            self.attributes()

    def pawn_is_home(self):
        """is the piece a pawn that can no longer move?"""
        ret = self.ptype == 'p'
        chk = self.is_home()
        ret = ret and chk
        if ret:
            self.lock()
        else:
            self.unlock()
        return ret

    @PROFILE
    def update(self, newpos, ignore_lock=False):
        """update the position of a piece on the board"""
        global BOARD
        if not ignore_lock:
            assert not self.locked, (self.ptype, self.pos, self.team)
        if self.ontheboard:
            assert not BOARD[newpos]
            oldpos = (self.pos[0], self.pos[1])
            self.pos = (newpos[0], newpos[1])
            self.update_xy(newpos, ignore_lock=ignore_lock)
            BOARD[newpos] = self
            if oldpos != newpos:
                BOARD[oldpos] = None
        else:
            self.place(newpos, ignore_lock=ignore_lock)
        # check if the piece (pawn) is now locked
        if not ignore_lock:
            self.pawn_is_home()

    @PROFILE
    def place(self, pos, ignore_lock=False):
        """place the piece onto the board"""
        global BOARD
        if not ignore_lock:
            assert not self.locked, (self.ptype, self.team)
        assert not self.ontheboard, self.ident()
        assert not BOARD[pos], (BOARD, pos)
        self.ontheboard = True
        self.pos = pos
        self.update(pos, ignore_lock=ignore_lock)
        self.draw()
        self.square_color()
        self.attributes()

    @PROFILE
    def remove(self):
        """remove piece from the board"""
        global BOARD
        assert self.ontheboard
        assert self == BOARD[self.pos], (self, pos, BOARD[pos])
        pos = self.pos
        BOARD[pos] = None
        self.ontheboard = False
        self.pos = None
        self.x = None
        self.y = None
        self.sq_color = None
        self.locked = False
        self.demote()

    if not DRAW:
        def draw(self):
            pass
    else:
        def draw(self):
            """draws the piece onto the board"""
            pos = (self.x, self.y)
            assert pos[0] is not None, pos
            WIN.blit(self.image, pos)

    @PROFILE
    def update_xy(self, pos, ignore_lock=False):
        """converts row, column -> x, y
        (row, col -> position)"""
        if not ignore_lock:
            assert not self.locked, (self.ptype, self.pos, self.team)
        width = WIDTH // 8 # width = 100
        assert width == 100, (width, "sanity check")
        row, col = pos
        self.y = int(row * width)
        self.x = int(col * width)




### DRAW


class Node: # chess squares
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = int(row * width)
        self.y = int(col * width)
        self.color = WHITE
        #self.occupied = None

    def draw(self):
        """draw the rectangle and piece"""
        self.draw_rect()
        self.draw_piece_here()

    if not DRAW:
        def draw_rect(self):
            pass
        def draw_piece_here(self):
            pass
    else:
        def draw_rect(self):
            """draw rectangle of a certain fill color, node.color"""
            pygame.draw.rect(WIN, self.color, (self.y, self.x, int(WIDTH / 8), int(WIDTH / 8)))

        def draw_piece_here(self):
            """draw the piece occupying this node"""
            global BOARD
            piece = BOARD[(self.row, self.col)]
            if piece is not None:
                #print("drawing at node:", self.row, self.col, self.x, self.y)
                #print(piece.ptype, piece.ontheboard, piece.pos)
                piece.draw()


@PROFILE
def make_grid(rows, width):
    """creates the visual grid; run once"""
    if not GENETICS:
        assert not make_grid.hasrun, "sanity check"
    grid = []
    gap = WIDTH // rows
    #print(gap)
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap)
            grid[i].append(node)
            if (i+j)%2 == 1:
                grid[i][j].color = GREY
    make_grid.hasrun = True
    return grid
make_grid.hasrun = False

if not DRAW:
    def draw_grid(rows, width):
        pass
else:
    def draw_grid(rows, width):
        """draws the outlines of the chess squares"""
        gap = width // 8
        for i in range(rows):
            pygame.draw.line(WIN, BLACK, (0, i * gap), (width, i * gap))
            for j in range(rows):
                pygame.draw.line(WIN, BLACK, (j * gap, 0), (j * gap, width))

@PROFILE
def remove_highlight(grid, skip_color_pos=None):
    """reverts the grid square colors"""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (i,j) == skip_color_pos:
                continue
            grid[i][j].color = square_color((i, j))
    return grid

def square_color(pos):
    """get the color (white or grey) of the square
    useful for checking bishop color"""
    row, col = pos
    return GREY if (row+col)%2 else WHITE

if not DRAW:
    def highlight(pos, grid, color):
        i, j = pos
        node = grid[i][j]
        node.color = color
else:
    def highlight(pos, grid, color):
        """reverts the grid square colors"""
        i, j = pos
        node = grid[i][j]
        node.color = color
        instant_draw(node)

if not DRAW:
    def instant_draw(node, rows=8, width=WIDTH):
        pass
else:
    def instant_draw(node, rows=8, width=WIDTH):
        """instantly re-draw the square"""
        draw_grid(rows, width) # is this call necessary?
        node.draw() # draw the rectangle, and piece
        pygame.display.update()

if not DRAW:
    def update_display(grid, rows, width):
        pass
else:
    def update_display(grid, rows, width):
        """update display"""
        # loop over nodes in grid
        for row in grid:
            for node in row:
                node.draw()
        draw_grid(rows, width)
        pygame.display.update()

### END DRAW

@PROFILE
def board_snapshot():
    """convert board to snapshot for 
    three/five fold repetition rule check"""
    global BOARD
    ret = [[0 for i in range(8)] for j in range(8)]
    assert len(BOARD) == 8, len(BOARD)
    for i, row in enumerate(BOARD):
        #assert len(row) == 8, row
        for j, piece in enumerate(row):
            if piece is None:
                ret[i][j] = 'None'
            else:
                ret[i][j] = piece.attribute_str
    ret = str(ret)
    return ret

            


### MOUSE LOGIC

def mouse():
    """get input from user via mouse;
    convert to which part of chess grid user is clicking on"""
    found = False
    leftright = None
    while not found or leftright not in ('l', 'r', None):
        pygame.time.delay(50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                xy = pygame.mouse.get_pos()
                row, col = xy_to_rowcol(xy, WIDTH)
                leftright = 'l' if event.button == 1 else leftright
                leftright = 'r' if (event.button == 3 or shift) else leftright
                found = True
                break
    return (row, col), leftright

@PROFILE
def xy_to_rowcol(xy, WIDTH):
    """converts screen position to row, column"""
    interval = WIDTH / 8
    x, y = xy 
    columns = x // interval
    rows = y // interval
    return int(rows), int(columns)


### END MOUSE LOGIC

def Pieces(team):
    """Create all the pieces for a team"""
    ret = []
    count = 0
    for ptype in PTYPES:
        if ptype == 'p':
            for _ in range(8):
                home = ((tuple([1]) if team == 'b' else tuple([6])),
                        tuple(range(8)))
                piece = Piece(team, ptype, IMAGES[(team, ptype, False)], home)
                ret.append(piece)
            #piece = ret[-1]
            #piece.place((3,3) if team == 'w' else (5,5))
        elif ptype in ['r', 'b', 'n']:
            for i in range(2):
                if ptype == 'r':
                    home = (tuple([0 if team == 'b' else 7]), (0,7))
                elif ptype == 'n':
                    home = (tuple([0 if team == 'b' else 7]), (1,6))
                else:
                    home = (tuple([0 if team == 'b' else 7]), (2,5))
                piece = Piece(team, ptype, IMAGES[(team, ptype, False)], home)
                ret.append(piece)
        elif ptype == 'k':
            count += 1
            home = (tuple([7 if team == 'w' else 0]), tuple([4]))
            piece = Piece(team, ptype, IMAGES[(team, ptype, False)], home)
            piece.place((3,5) if team == 'b' else (4,3))
            assert piece.ontheboard
            ret.append(piece)
        else:
            assert ptype == 'q', ptype
            home = (tuple([7 if team == 'w' else 0]), tuple([3]))
            piece = Piece(team, ptype, IMAGES[(team, ptype, False)], home)
            ret.append(piece)
    assert count == 1, count
    return ret

## Creates instances of chess pieces, so far we got: pawn, king, rook and bishop
## The first parameter defines what team its on and the second, what type of piece it is
global bpieces
global wpieces
bpieces = Pieces('b')
wpieces = Pieces('w')

@PROFILE
def bishop_wall_check(team, ignore_lock=False):
    """weak check to see if the bishop can get home
    """
    global BOARD
    global wpieces
    global bpieces
    pieces = bpieces if team == 'b' else wpieces
    left = [(6, 1), (6, 3)] if team == 'w' else [(1, 1), (1, 3)]
    right = [(6, 4), (6, 6)] if team == 'w' else [(1, 4), (1, 6)]
    lbishop = (7, 2) if team == 'w' else (0, 2)
    rbishop = (7, 5) if team == 'w' else (0, 5)
    legal = True
    lhome = False # assume bishop is not on the board
    rhome = False
    home_count = 0
    for piece in pieces:
        if not piece.ontheboard:
            continue
        if not piece.ptype == 'b':
            continue
        if piece.promoted:
            continue
        # found a bishop
        if piece.is_home():
            _, col = piece.pos
            if col == 2:
                lhome = True
            else:
                rhome = True
    problems = []
    if not lhome:
        count = 0
        for pos in left:
            piece = BOARD[pos]
            if piece is None:
                continue
            if piece.team != team:
                continue
            if not piece.is_home():
                continue
            assert piece.ptype == 'p' and not piece.promoted, piece.ident()
            if not ignore_lock:
                assert piece.locked, piece.ident()
            problems.append(pos)
            count += 1
        legal = count < 2
    if legal and not rhome:
        count = 0
        for pos in right:
            piece = BOARD[pos]
            if piece is None:
                continue
            if piece.team != team:
                continue
            if not piece.is_home():
                continue
            problems.append(pos)
            count += 1
            assert piece.ptype == 'p' and not piece.promoted, piece.ident()
            if not ignore_lock:
                assert piece.locked, piece.ident()
        legal = count < 2
    if not legal:
        print("rhome", rhome)
        print("lhome", lhome)
        print("problem positions:", problems, "for", team)
    return legal


@PROFILE
def king_check_pawn_is_home(team, newpos=None):
    """pawns on their own home row should not be able to
    check the king of the other team"""
    global wpieces
    global bpieces
    oteam = other_team(team)
    pieces = wpieces if oteam == 'w' else bpieces
    legal = True
    save = None
    for piece in pieces:
        if not legal:
            break
        if not piece.ontheboard or piece.ptype != 'p':
            continue
        pawn = piece
        pawn.is_home() # set locked, if it hasn't been set already
        attacking = newpos if newpos is not None else find_king(team).pos
        if check_attacking(pawn, attacking):
            legal = not pawn.locked
            if not legal:
                save = pawn
    if save is not None:
        pass
        #print("check pawn debug:")
        #save.ident()
    return legal

@PROFILE
def other_team(team):
    """return the other team"""
    assert team in ('w', 'b'), team
    return 'w' if team == 'b' else 'b'

@PROFILE
def allonboard(pieces):
    """checks to see if all pieces in list are on the board"""
    ret = True
    for piece in pieces:
        if not ret:
            break
        ret = ret and piece.ontheboard
    return ret

@PROFILE
def allhome(pieces):
    """check to see if all pieces home"""
    ret = None
    for piece in pieces:
        if not piece.ontheboard:
            ret = False
        else:
            ret = piece.is_home()
        if not ret:
            break
    assert ret is not None, pieces
    return ret
    

@PROFILE
def check_winning(team_pieces):
    """Check board to see if team is winning
    goal for team is to reset the board 
    to the starting chess position for team
    
    also, all the pieces must be on the board
    (you must give back all your opponent's pieces
    before you can end the game)
    """
    global wpieces
    global bpieces
    team = team_pieces[0].team
    ret = allonboard(bpieces) and allonboard(wpieces)
    ret = ret and allhome(team_pieces)
    # redundant check, for safety
    for piece in team_pieces:
        if not ret:
            break
        if piece.promoted:
            ret = False
            break
        row, col =  piece.pos
        ptype = piece.ptype
        # pawn check
        if pytpe == 'p':
            if team == 'b' and row != 6:
                ret = False
            if team == 'w' and row != 1:
                ret = False
            continue

        ## check rest of pieces
        # front/back row check
        if team == 'b':
            if row != 7:
                ret = False
                break
        elif team == 'w':
            if row != 0:
                ret = False
                break
        
        # final type check, via columns
        if ptype == 'r':
            if col not in [0, 7]:
                ret = False
                break
        elif ptype == 'n':
            if col not in [1, 6]:
                ret = False
                break
        elif ptype == 'b':
            if col not in [2, 5]:
                ret = False
                break
        elif ptype == 'q':
            if col != 3:
                ret = False
                break
        elif ptype == 'k':
            if col != 4:
                ret = False
                break
    return ret

@PROFILE
def pawn_spawn_home_check(pawn, spawn_pos, newpos=None):
    """pawn cannot spawn at on its home row attacking the opposite king
    this would give that king's team an instant win, and wouldn't make sense
    due to the king not being able to ever be checked by a pawn on its home row
    in regular chess
    """
    team = pawn.team
    oteam = other_team(team)
    king = find_king(oteam)
    row = spawn_pos[0]
    if team == 'b':
        chk = row != 1
    else:
        chk = row != 6
    if newpos == None:
        legal = not pawn_attacking(team, spawn_pos, king.pos)
    else:
        legal = not pawn_attacking(team, spawn_pos, newpos)
    return legal or chk

@PROFILE
def check_bishop_color(bishop, spawn_pos):
    """check the bishop's color"""
    assert bishop.ptype == 'b'
    team = bishop.team
    global wpieces
    global bpieces
    pieces = bpieces if team == 'b' else wpieces
    color = square_color(spawn_pos)
    legal = True
    for piece in pieces:
        if not legal:
            break
        if not piece.ontheboard:
            continue
        if piece.ptype != 'b' or piece.promoted:
            continue
        legal = piece.sq_color != color
    legal = legal or bishop.promoted
    return legal


@PROFILE
def check_legal_give(give_params, piece_to_move, newpos):
    """Check that the piece given back is
    a) placed properly and
    b) comes from the available set"""
    global BOARD

    pos = piece_to_move.pos
    spawn_pos = give_params.spawn_pos

    reason = ""
    # check that the space is unoccupied
    legal = BOARD[spawn_pos] is None or spawn_pos == pos
    # check that the piece to give back is not on the board
    legal = legal and not give_params.spawn_piece.ontheboard

    if give_params.spawn_piece.ptype == 'b':
        legal = legal and check_bishop_color(give_params.spawn_piece, spawn_pos)
        if not legal:
            reason = "bishop color dupe"

    # legal moves check; premove
    legal_count, only_legal = count_legal(give_params.spawn_piece.team) 
    oldpos = piece_to_move.pos
    promoted = give_params.spawn_piece.promoted
    piece_to_move.update(newpos, ignore_lock=True)
    give_params.spawn_piece.place(spawn_pos, ignore_lock=True)
    legal_count2, only_legal2 = count_legal(give_params.spawn_piece.team) 
    if only_legal and not only_legal2:
        legal = False
        reason = 'no legal moves for '+full(give_params.spawn_piece.team)+"."
    # bishop wall check
    if legal:
        legal = bishop_wall_check(give_params.spawn_piece.team, ignore_lock=True)
        if not legal:
            print("illegal gift: no walling off the other team's bishop squares.")
    give_params.spawn_piece.remove()
    give_params.spawn_piece.promoted = promoted
    piece_to_move.update(oldpos, ignore_lock=True)


    # can the pawns get home?
    if give_params.spawn_piece.ptype == 'p' and legal and give_params.promote_type is None:
        pawn = give_params.spawn_piece
        legal = pawn_spawn_home_check(pawn, spawn_pos, newpos=(
            newpos if piece_to_move.ptype=='k' else None))
        if not legal:
            reason = "locked pawn attacking king"
        if legal:
            legal = pawn_scan(pawn.team, spawn_pos)
            if not legal:
                reason = "pawn scan fail"
        # don't allow pawn spawn on the same row as their king
        if legal:
            row_spawn = spawn_pos[0]
            if pawn.team == 'b':
                legal = row_spawn != 0
            else:
                legal = row_spawn != 7
            if not legal:
                reason = "no spawning pawns on back row"
    if not legal and not GENETICS:
        print("illegal gift:", reason)
    return legal

@PROFILE
def pawn_scan(team, pos=None):
    """can the pawns of team get home?"""
    global wpieces
    global bpieces
    pieces = bpieces if team == 'b' else wpieces
    opieces = wpieces if team == 'b' else bpieces
    pawn_list = []
    locked_cols = []
    oteam = other_team(team)
    for piece in pieces:
        if piece.ontheboard and piece.ptype == 'p' and not piece.promoted:
            if piece.locked:
                locked_cols.append(piece.pos[1])
            else:
                pawn_list.append(piece)
    if not pawn_list:
        return True # no need to check yet
    count = 0
    opencols = set(range(8))-set(locked_cols)
    reachable = {}
    col_diffs = {}
    for idx, pawn in enumerate(pawn_list):
        reachable[idx] = set()
        row, col = pawn.pos
        col_diffs[idx] = {}
        for col1 in opencols:
            cold = abs(col1-col)
            home_row = pawn.home[0][0]
            rowd = abs(row-home_row)
            if rowd >= cold:
                reachable[idx].add(col1)
                col_diffs[idx][col1] = cold

    # add the pawn we are just now spawning
    if pos is not None:
        reachable[len(pawn_list)] = set()
        row, col = pos
        home_row = 1 if team == 'b' else 6
        col_diffs[len(pawn_list)] = {}
        for col1 in opencols:
            cold = abs(col1-col)
            rowd = abs(row-home_row)
            if rowd >= cold:
                reachable[len(pawn_list)].add(col1)
                col_diffs[len(pawn_list)][col1] = cold
    # and now we brute force the solution using recursion
    # we try solution columns in order of how far away they are
    # closer first
    #solution = tenant_matching(reachable, mins=col_diffs)
    solution = tenant_matching(col_diffs)
    if len(reachable) == 1 and reachable[list(reachable)[0]]:
        assert solution, (solution, reachable)

    # now we check to make sure there are enough pieces off the board
    # do to our sorting, we can read off this minimum from the solution
    if solution:
        min_off_board = 0
        for pawn in solution:
            min_off_board += col_diffs[pawn][solution[pawn]]
        if min_off_board > sum(offboard_count(opieces).values()):
            if not GENETICS:
                print("not enough pieces off board for pawns to get home")
                print(min_off_board, sum(offboard_count(opieces).values()),
                      offboard_count(opieces))
            solution = None

    if not solution:
        if not GENETICS:
            print("no solution, pawn scan.  pawn list:")
            for idx, pawn in enumerate(pawn_list):
                pawn.ident()
                print("idx, reachable", idx, reachable[idx])
                print("reachable", reachable, tenant_matching(col_diffs))
                print("col diffs", col_diffs)
            print("END no solution")

    legal = bool(solution)
    return legal


@PROFILE
def king_around(team, newpos):
    """check squares around newpos for king of other team
    we aren't allowed to move our king next to the other king in chess,
    so we aren't allowed to do so in reverse chess either.
    """
    global BOARD
    origin = (newpos[0]-1, newpos[1]-1)
    origin = np.asarray(origin)
    for col in range(3):
        for row in range(3):
            if col == 1 and row == 1: # skip the king's square
                continue
            around_pos = origin+np.array([row,col]) # square of other king
            around_pos = tuple(around_pos)
            piece = BOARD[around_pos]
            if piece is None:
                continue
            if piece.team == other_team(team) and piece.ptype == 'k':
                return False
    return True

@PROFILE
def discovered_check_check(piece, newpos, demote=False):
    """checking the other king is not allowed,
    as in reverse this would mean the other king
    is moving into check.
    """
    global BOARD

    # pre-move the piece to perform check
    assert piece.pos != newpos, (piece.pos, newpos)
    oldpos = piece.pos
    piece.update(newpos, ignore_lock=True)
    old_ptype = piece.ptype
    if demote:
        piece.demote()

    ret = False
    for row in BOARD:
        if ret:
            break
        for piece2 in row:
            if piece2 is None:
                continue
            if piece2.team != piece.team:
                continue
            if ret:
                break
            ret = attacking_king(piece2)
            discovery = piece2
    if ret:
        pass
        #print("illegal discovery:", discovery.ptype, discovery.pos)

    # undo pre-move
    piece.update(oldpos, ignore_lock=True)
    if demote and piece.ptype != old_ptype:
        piece.promote(old_ptype)

    return ret
        
@PROFILE
def attacking_king(piece, newpos=None):
    """Check if piece is attacking king of other team"""
    global BOARD
    assert piece is not None
    team = piece.team
    oteam = other_team(team)
    # find other king
    piece2 = find_king(oteam)
    assert id(piece) != id(piece2), "sanity check"
    assert piece.pos != piece2.pos, (piece.pos, piece.ident(), piece2.ident(), BOARD[piece.pos].ident())
    return check_attacking(piece, piece2.pos, newpos=newpos)


@PROFILE
def find_king(team):
    """find king on board of given team"""
    global BOARD
    if team in find_king.cache:
        return find_king.cache[team]
    else:
        for row in BOARD:
            for piece in row:
                if piece is None:
                    continue
                if piece.ptype == 'k' and piece.team == team:
                    find_king.cache[team] = piece
                    return piece
    assert None, "king not found on board"
find_king.cache = {}


@PROFILE
def rook_attacking(pos_qr, pos):
    """check if the rook is attacking pos"""
    row, col = pos
    row_qr, col_qr = pos_qr # position of attacking piece
    ret = False
    if row == row_qr or col == col_qr:
        if not check_impinging(pos_qr, pos):
            ret = True
    return ret

@PROFILE
def pawn_attacking(pawn_team, pawn_pos, pos):
    """Is pawn attacking pos?"""
    # which row is being attacked by the pawn?
    # black attacks down (row1+1)
    # white attacks up (row1-1)
    row, col = pos
    row1, col1 = pawn_pos
    row_attack = row1 + 1 if pawn_team == 'b' else row1 - 1
    return row_attack == row and col in (col1+1, col1-1)

@PROFILE
def bishop_attacking(bishop_pos, pos):
    """Check if bishop at bishop_pos is attacking pos"""
    row, col = pos
    row1, col1 = bishop_pos
    row_delta = abs(row-row1)
    col_delta = abs(col-col1)
    return row_delta == col_delta and not check_impinging(bishop_pos, pos)

@PROFILE
def knight_attacking(knight_pos, pos):
    """check if knight is attacking; similar to other attacking functions"""
    row, col = pos
    row1, col1 = knight_pos
    row_delta = abs(row-row1)
    col_delta = abs(col-col1)
    return (row_delta, col_delta) in [(1,2),(2,1)]

@PROFILE
def king_attacking(king_pos, pos):
    """needed?"""
    row, col = pos
    row1, col1 = king_pos
    row_delta = abs(row-row1)
    col_delta = abs(col-col1)
    return (row_delta, col_delta) in [(0, 1), (1, 0), (1, 1)]

@PROFILE
def check_attacking(piece, pos, newpos=None):
    """check if piece (optionally at proposed position newpos)
    is attacking pos"""
    ptype = piece.ptype
    piecepos = piece.pos if newpos is None else newpos
    if ptype == 'r':
        ret = rook_attacking(piecepos, pos)
    elif ptype in 'q':
        ret = rook_attacking(piecepos, pos)
        ret = ret or bishop_attacking(piecepos, pos)
    elif ptype == 'p':
        ret = pawn_attacking(piece.team, piecepos, pos)
    elif ptype == 'n':
        ret = knight_attacking(piecepos, pos)
    elif ptype == 'b':
        ret = bishop_attacking(piecepos, pos)
    elif ptype == 'k':
        ret = king_attacking(piecepos, pos)
    return ret

@PROFILE
def reverse_check_check(team, newpos, ptype):
    """check for reverse check.
    if the king is being attacked by more than one piece, then the move is illegal
    if the king is being attacked by only one piece,
    then that piece is in reverse check and must move on the next turn.
    """
    assert None, "turns out this isn't needed; discovered check does the job"
    global wpieces
    global bpieces
    global BOARD
    oteam = other_team(team)
    pieces = bpieces if oteam == 'b' else wpieces
    row, col = newpos
    attacking = []
    for piece in pieces:
        if len(attacking) > 1:
            break
        piece.pre_reverse_check = False
        if not piece.ontheboard:
            continue
        if check_attacking(piece, newpos) and ptype == 'k':
            attacking.append(piece)
    legal = len(attacking) <= 1
    if len(attacking) == 1:
        attacking[0].pre_reverse_check = True
        assert attacking[0].ptype != 'k', "king cannot be put in reverse check"
        # make sure we actually modified the global piece containers
        chk_count = 0
        if oteam == 'w':
            for piece in wpieces:
                if piece.pre_reverse_check:
                    pos = piece.pos
                    assert BOARD[pos].pre_reverse_check, BOARD[pos]
                    chk_count += 1
        else:
            for piece in bpieces:
                if piece.pre_reverse_check:
                    pos = piece.pos
                    assert BOARD[pos].pre_reverse_check, BOARD[pos]
                    chk_count += 1
        assert chk_count == 1, (chk_count, oteam)
    return legal


@PROFILE
def check_impinging(pos1, pos2):
    """check if any pieces are in between two positions"""
    global BOARD
    row1, col1 = pos1
    row2, col2 = pos2
    row_delta = row1-row2
    col_delta = col1-col2
    found = False
    range_delta = range(1, abs(row_delta)) if row_delta else range(1, abs(col_delta))
    assert pos1 != pos2, (pos1, pos2)
    for delta in range_delta:
        chk_row = row2 + delta*np.sign(row_delta)
        chk_col = col2 + delta*np.sign(col_delta)
        chk_pos = (chk_row, chk_col)
        if BOARD[chk_pos] is not None:
            found = True
            break
    return found


@PROFILE
def check_legal_move(ptype, pos, newpos, prin=False):
    """Check legality of movement of piece of type ptype
    from pos to newpos
    """
    global BOARD
    global wpieces
    global bpieces
    legal = True
    row, col = pos
    new_row, new_col = newpos
    row_delta = new_row-row
    col_delta = new_col-col
    forced_give = False
    reverse_castle = False
    reason = ''

    # check unoccupied, and that the move is non-zero
    if BOARD[newpos] or pos == newpos:
        #print("occupied by:", BOARD[newpos].team, BOARD[newpos].ptype)
        legal = False
        reason = 'occupied'
        # spawn_piece, spawn_pos = give_params
        #if give_params.spawn_piece is not None:
        #    legal = legal and BOARD[give_params.spawn_pos] is None

    # king movement
    elif ptype == 'k':
        if not col_delta:
            legal = abs(row_delta) == 1
        elif not row_delta:
            legal = abs(col_delta) == 1
        else:
            legal = abs(col_delta) == 1 and abs(row_delta) == 1


        if not legal:
            legal = reverse_castle_check('k', BOARD[pos].team, pos, newpos)
            legal = legal and not BOARD[pos].promoted
            reverse_castle = legal
        
        #if legal:
        #    legal = king_around(BOARD[pos].team, newpos) # redundant with attacking_king

    # rook movement
    elif ptype == 'r':
        legal = check_legal_rook(pos, newpos)
        if not legal:
            reason = 'bad rook movement'

    # knight movement
    elif ptype == 'n':
        pair = (abs(row_delta), abs(col_delta))
        if pair not in [(1,2), (2,1)]:
            legal = False
            reason = 'bad knight movement'

    # queen movement
    elif ptype == 'q':
        if not row_delta or not col_delta:
            legal = check_legal_rook(pos, newpos)
        else:
            legal = check_legal_bishop(pos, newpos)
        if not legal:
            reason = 'bad queen movement'

    # bishop movement
    elif ptype == 'b':
       legal = check_legal_bishop(pos, newpos) 
       if not legal:
            reason = 'bad bishop movement'

    # pawn movement
    elif ptype == 'p':
       legal, forced_give = check_legal_pawn(pos, newpos, BOARD[pos].team)
       if not legal:
           reason = 'bad pawn movement'

    # check for blocking pieces
    if legal:
        legal = not check_impinging(pos, newpos) or reverse_castle or ptype == 'n'
        if not legal:
            reason = 'piece(s) in the way'
        if not legal and ptype == 'r':
            legal = reverse_castle_check('r', BOARD[pos].team, pos, newpos)
            reverse_castle = legal
            if not legal:
                reason = 'pieces in the way, and not reverse castle'

    # king cannot move into check in chess,
    # so in reverse chess you cannot attack the other king
    # or cause an attack on the other king
    # check for discovered, regular check:
    if legal:
        # this check must happen last, since we do the check by moving piece=BOARD[pos]
        # and then see if any kings are under attack as a result
        # king is allowed to move into check though, that's a reverse check(mate)
        legal = not discovered_check_check(BOARD[pos], newpos, demote=ptype!=BOARD[pos].ptype)
        if not legal:
            reason = 'discovered check'

    # move should not put your king under attack by a pawn at home
    if legal:
        legal = king_check_pawn_is_home(BOARD[pos].team,
                                        newpos=newpos if BOARD[pos].ptype == 'k' else None)
        if not legal:
            reason = 'pawn check at home'

    if legal and reverse_castle:
        opieces = wpieces if BOARD[pos].team == 'b' else bpieces
        for piece in opieces:
            if not legal:
                break
            legal = not check_attacking(piece, ((0, 4) if BOARD[pos].team == 'b' else (7, 4)))
        if not legal:
            reason = 'king cannot castle to escape check, so king cannot reverse castle into check'

    if reason and prin and not GENETICS:
        print("reason:", reason)

    return legal, forced_give, reverse_castle

@PROFILE
def reverse_castle_check(ptype, team, pos, newpos):
    """check for reverse castling"""
    global BOARD
    legal = False
    row, col = pos
    row1, col1 = newpos
    if team == 'b':
        legal = row == 0 and row1 == 0
    else:
        legal = row == 7 and row1 == 7
    rook_col = None
    if legal:
        brow = BOARD[row]
        count = 0
        for space in brow:
            if space is not None:
                count += 1
        legal = count == 2
        if legal:
            legal = (brow[2] and brow[3]) or (brow[5] and brow[6])
            if legal:
                if brow[2] and brow[3]:
                    legal = brow[2].ptype == 'k' and brow[3].ptype == 'r'
                    king, rook = brow[2], brow[3]
                    rook_col = 0
                elif brow[5] and brow[6]:
                    legal = brow[6].ptype == 'k' and brow[5].ptype == 'r'
                    king, rook = brow[6], brow[5]
                    rook_col = 7
    if legal:
        if ptype == 'k':
            legal = pos == king.pos and newpos == (row, 4)
        if ptype == 'r':
            legal = pos == rook.pos and newpos == (row, rook_col)
    return legal

@PROFILE
def check_legal_pawn(pos, newpos, team):
    """check legality of pawn movement"""
    global wpieces
    global bpieces
    pieces = bpieces if team == 'w' else wpieces
    legal = False
    forced_give = False
    row, col = pos
    new_row, new_col = newpos


    # black's home rows are 0, 1, so row-new_row = 1
    rowchk = row-new_row == 1 if team == 'b' else row-new_row == -1
    rowchk = rowchk and new_row not in (0, 7)
    if rowchk:
        forced_give = abs(new_col-col) == 1
        legal = forced_give or col == new_col
    legal = legal and not (forced_give and allonboard(pieces))
    return legal, forced_give

@PROFILE
def check_legal_bishop(pos, newpos):
    """check legality of bishop movement"""
    global BOARD
    legal = True
    row, col = pos
    new_row, new_col = newpos
    row_delta = abs(new_row-row)
    col_delta = abs(new_col-col)
    legal = row_delta == col_delta
    if legal:
        legal = not check_impinging(pos, newpos)
    return legal


@PROFILE
def check_legal_rook(pos, newpos):
    """Check legality of rook movement"""
    global BOARD
    legal = True
    row, col = pos
    new_row, new_col = newpos
    row_delta = new_row-row
    col_delta = new_col-col
    why = ''
    if row_delta and col_delta:
        legal = False
    
    if row_delta:
        for i in range(row+1, new_row+1):
            if not legal:
                break
            legal = not bool(BOARD[i, col])
    elif col_delta:
        for i in range(col+1, new_col+1):
            if not legal:
                break
            legal = not bool(BOARD[row, i])
    return legal

class GiveParams:
    def __init__(self, spawn_piece, spawn_pos, promote_type):
        """init"""
        self.spawn_piece = spawn_piece
        self.spawn_pos = spawn_pos
        self.promote_type = promote_type
        if promote_type is not None: # piece should not be promoted yet
            assert spawn_piece.ptype == 'p', spawn_piece.ident()
            assert not spawn_piece.promoted, "sanity check"

    def dump(self):
        print('GiveParams', self.spawn_pos, self.promote_type)
        if self.spawn_piece is not None:
            self.spawn_piece.ident()
        print('GiveParams END')


@PROFILE
def count_board():
    """counts how many pieces are on the board"""
    global BOARD
    count = 0
    for i in BOARD:
        for j in i:
            if j is not None:
                count += 1
    return count

@PROFILE
def set_reverse_check():
    """set reverse check"""
    assert None, "turns out not to be needed"
    global BOARD
    global REVERSE_CHECK
    if REVERSE_CHECK is not None:
        REVERSE_CHECK.reverse_check = False
        assert not BOARD[REVERSE_CHECK.pos].reverse_check
    REVERSE_CHECK = None
    for row in BOARD:
        for piece in row:
            if piece is not None:
                if piece.pre_reverse_check:
                    piece.reverse_check = True
                    REVERSE_CHECK = piece
                    piece.pre_reverse_check = False
                    return


class Move:
    def __init__(self, ptype, pos, newpos, give_params, whose_turn_is_it, reverse_castle, demote):
        """init"""
        self.ptype = ptype
        self.pos = pos
        self.newpos = newpos
        self.oldptype = None
        self.oldptype_spawn = None
        self.give_params = give_params
        self.whose_turn_is_it = whose_turn_is_it
        self.reverse_castle = reverse_castle
        self.demote = demote
        self.other_piece = None

    def dump(self):
        """print"""
        print('Move', self.ptype, self.pos, self.newpos, self.oldptype,
              self.oldptype_spawn, self.whose_turn_is_it,
              self.reverse_castle, self.demote, self.other_piece)
        self.give_params.dump()


    @PROFILE
    def undo(self):
        """undo the move"""
        global BOARD

        # de-spawn, demote
        if self.give_params.spawn_piece is not None:
            self.give_params.spawn_piece.remove()
            assert not self.give_params.spawn_piece.ontheboard, (self.give_params.spawn_piece, BOARD)
            assert not self.give_params.spawn_piece.promoted, (self.give_params.spawn_piece, BOARD)
            if self.oldptype_spawn is not None:
                assert self.oldptype_spawn == self.give_params.spawn_piece.ptype, (
                    self.give_params.spawn_piece.ident(), self.oldptype_spawn, self.give_params.spawn_piece.ptype)

        # precount check
        precount = count_board()
        piece_to_move = BOARD[self.newpos]
        assert piece_to_move is not None, BOARD

        # promote
        if self.demote:
            piece_to_move.promote(self.oldptype)
            assert piece_to_move.ptype == self.oldptype, piece_to_move.ident()

        # undo update
        piece_to_move.unlock()
        piece_to_move.update(self.pos)

        # reverse reverse castle
        if self.reverse_castle:
            row, col = self.newpos
            row1, col1 = self.pos
            jump = (row, int(np.ceil((col+col1)/2)))
            self.other_piece.unlock()
            self.other_piece.update(jump)

        # reverse reverse castle check
        if self.reverse_castle:
            assert not piece_to_move.is_home(), piece_to_move.pos
            assert not other_piece.is_home(), other_piece.pos

        # checks
        count = count_board()
        assert count == precount, (count, precount)
        assert self.check_turn(undo=True), "sanity check"

    @PROFILE
    def execute(self):
        """update board; does not check for move legality"""
        global BOARD

        # precount check
        precount = count_board()

        piece_to_move = BOARD[self.pos]
        assert piece_to_move is not None, BOARD

        # reverse castle
        if self.reverse_castle:
            row, col = self.pos
            row1, col1 = self.newpos
            jump = (row, int(np.ceil((col+col1)/2)))
            self.other_piece = BOARD[jump]
            self.other_piece.move_home()
            self.other_piece.lock()

        # update
        piece_to_move.update(self.newpos)

        # demote
        if self.demote:
            self.oldptype = piece_to_move.ptype
            piece_to_move.demote()
            if piece_to_move.team == 'b':
                assert piece_to_move.pos[0] != 7, piece_to_move.ident()
            else:
                assert piece_to_move.pos[0] != 0, piece_to_move.ident()

        # reverse castle check
        if self.reverse_castle:
            assert piece_to_move.is_home(), (piece_to_move.pos, piece_to_move.ident())
            assert other_piece.is_home(), other_piece.pos
            piece_to_move.lock()

        # count check
        count = count_board()
        assert count == precount, (count, precount)

        # spawn
        if self.give_params.spawn_piece is not None:
            assert not self.give_params.spawn_piece.ontheboard, (self.give_params.spawn_piece, BOARD)
            self.give_params.spawn_piece.place(self.give_params.spawn_pos)
            # promote piece
            if self.give_params.promote_type is not None:
                assert not self.give_params.spawn_piece.promoted, spawn_piece.ident()
                self.oldptype_spawn = self.give_params.spawn_piece.ptype
                self.give_params.spawn_piece.promote(self.give_params.promote_type)

        # check the turn
        assert self.check_turn(), "sanity check"


    @PROFILE
    def check_turn(self, undo=False):
        """check to make sure team is moving its own pieces"""
        turn = self.whose_turn_is_it
        piece_to_move = BOARD[self.newpos if not undo else self.pos]
        spawn_piece = self.give_params.spawn_piece
        legal = False
        if turn == 'b':
            #assert piece_to_move.team == 'b', piece_to_move
            legal = piece_to_move.team == 'b'
            if spawn_piece is not None:
                #assert spawn_piece.team == 'w', spawn_piece
                legal = legal and spawn_piece.team == 'w'
        else:
            # assert piece_to_move.team == 'w', piece_to_move
            legal = piece_to_move.team == 'w'
            if spawn_piece is not None:
                #assert spawn_piece.team == 'b', spawn_piece
                legal = legal and spawn_piece.team == 'b', spawn_piece
        return legal

def get_bids():
    """bid process determines who goes first
    each team bids how many of their first turns they will give back a piece
    bids go back and forth until both teams accept"""
    baccept = False
    waccept = False
    bbid = None 
    wbid = None
    # get initial bids
    _, bbid = get_bid('b', bbid, wbid)
    _, wbid = get_bid('w', bbid, wbid)
    while not baccept or not waccept:
        baccept, bbid = get_bid('b', bbid, wbid)
        if baccept and waccept:
            break
        waccept, wbid = get_bid('w', bbid, wbid)
    return bbid, wbid

def accept_bid(team, bid, other_bid):
    """ask if team accepts bid of other team"""
    title = "Bid Accept? your bid = "+str(bid)
    message = full(other_team(team))+" bids "+str(other_bid)+" do you accept?"
    accept = tkinter.messagebox.askyesno(title, message)
    return accept

def get_bid(team, bbid, wbid):
    """Get the bid from the user"""
    accept = False
    bid = bbid if team == 'b' else wbid
    other_bid = wbid if team == 'b' else bbid
    if bid is None:
        while bid is None:
            bid = prompt(text="bid, "+full(team),
                        title="Enter bid" , default='0')
            if bid is None:
                bid = 0
            try:
                bid = int(bid)
            except ValueError:
                messagebox.showerror('Input error',
                                    'Please enter an integer for your bid')
                bid = None

        accept = True
    if not accept:
        accept = accept_bid(team, bid, other_bid)
        if not accept:
            title = full(other_team(team))+' bid: ' + str(wbid) + " your bid: " + str(bbid)
            rraise = 0
            while rraise <= 0:
                rraise = prompt(text="raise bid by how much?", title=title , default='1')
                if rraise is None:
                    rraise = 0
                    accept = True
                    break
                try:
                    rraise = int(rraise)
                except ValueError:
                    messagebox.showerror('Input error',
                                         'Please enter an integer for your raise')
            bid += rraise
    return accept, bid

@PROFILE
def highlight_legal(piece_to_move, grid, color=BLUE):
    moves = get_legal_moves(piece_to_move)
    for newpos in moves:
        highlight(newpos, grid, color)


@PROFILE
def get_legal_moves(piece_to_move):
    """get all legal moves for a piece"""
    pos = piece_to_move.pos
    ret = []
    #global SNAP
    #snap = SNAP
    snap = board_snapshot()
    key = (snap, id(piece_to_move))
    if key in get_legal_moves.cache:
        ret = get_legal_moves.cache[key]
    else:
        for i, row in enumerate(BOARD):
            for j, piece in enumerate(row):
                newpos = (i, j)
                if newpos == pos:
                    continue
                legal, forced_give, reverse_castle = check_legal_move(piece_to_move.ptype, pos, newpos)
                if legal:
                    ret.append(newpos)
                if piece_to_move.promoted and ((pos[0] == 0 and piece_to_move.team == 'w') or \
                (pos[0] == 7 and piece_to_move.team == 'b')):
                    legal2, forced_give2, reverse_castle2  = check_legal_move('p', pos, newpos)
                    if legal2 and not legal:
                        ret.append(newpos)
        #piece_to_move.ident()
        #print("legal moves:", ret)
        get_legal_moves.cache[key] = ret
    return ret
get_legal_moves.cache = {}

def clear_caches():
    """cache invalidation"""
    get_legal_moves.cache = {}
    find_king.cache = {}

@PROFILE
def count_legal(team):
    count = 0
    only_legal = []
    for i, row in enumerate(BOARD):
        for j, piece in enumerate(row):
            pos = (i, j)
            if piece is None:
                continue
            if piece.team != team:
                continue
            toadd = len(get_legal_moves(piece))
            if toadd:
                only_legal.append(piece)
            count += toadd
    return count, only_legal

def check_bid(bid, move):
    """check the bids"""
    legal = bid <= 0 or move.give_params.spawn_piece is not None
    return legal

@PROFILE
def get_piece(team, ontheboard=True, onlypawn=False, idx=None):
    """return a random/non-random
    piece which is on the board"""
    global wpieces
    global bpieces
    pieces = bpieces if team == 'b' else wpieces
    ret = []
    for piece in pieces:
        if onlypawn and piece.ptype != 'p':
            continue
        if piece.ontheboard == ontheboard:
            ret.append(piece)
    #if ontheboard:
    #    assert seq_bounds(team)[0] == len(ret), (seq_bounds(team), len(ret))
    #else:
    #    assert seq_bounds(other_team(team))[3] == len(ret), (seq_bounds(team), len(ret))
    if ret:
        if idx is not None:
            try:
                ret = ret[idx]
            except IndexError:
                print(ret)
                print(len(ret))
                print(idx)
                raise
        else:
            ret = ret[np.random.randint(0, len(ret))]
    else:
        ret = None
    return ret

@PROFILE
def get_legal_move(piece_to_move, idx=None):
    """return a random legal move"""
    moves = get_legal_moves(piece_to_move)
    leftright = 'r' if np.random.randint(0,2) else 'l'
    if moves:
        if idx is None:
            try:
                newpos = moves[np.random.randint(0,len(moves))]
            except ValueError:
                print("lowhigh", 0,len(moves))
                raise
        else:
            try:
                newpos = moves[idx]
            except IndexError:
                print('err:', len(moves), idx)
                raise
    else:
        newpos = None
    return newpos, leftright

@PROFILE
def full(team):
    """return full team name"""
    if team == 'w':
        ret = 'white'
    elif team == 'b':
        ret = 'black'
    else:
        ret = None
    return ret

@PROFILE
def sum_bounds(bounds):
    """find total number of trial moves"""
    total = bounds[0]
    for i in range(bounds[0]):
        toadd = bounds[1][i]
        if toadd is None:
            continue
        try:
            total += toadd
        except TypeError:
            print('err:', bounds)
            raise
    bounds = bounds[2:]
    total += sum(bounds)
    return total

@PROFILE
def seq_bounds(team):
    """return a tuple of move sequence bounds"""
    global wpieces
    global bpieces
    pieces = bpieces if team == 'b' else wpieces
    opieces = bpieces if team == 'w' else wpieces
    oteam = other_team(team)
    ret = []
    # number of on the board pieces
    total1 = 0
    counts = onboard_count(pieces)
    for ptype in counts:
        total1 += counts[ptype]
    ret.append(total1)
    # list of legal move counts
    move_counts = []
    for piece in pieces:
        assert piece is not None
        if piece.ontheboard:
            assert piece.pos is not None
            count = len(get_legal_moves(piece))
            if count:
                move_counts.append(count)
            else:
                move_counts.append(None)
    ret.append(move_counts)
    # 2
    ret.append(2)
    # number of other team's off the board pieces
    total = 0
    counts = offboard_count(opieces)
    for ptype in counts:
        total += counts[ptype]
    ret.append(total)
    # 2
    ret.append(2)
    # 4
    ret.append(4)
    # 2
    ret.append(2 if not allonboard(opieces) else 1)
    # 2
    ret.append(2)
    return ret

@PROFILE
def random_move_from_seq(team, bounds):
    """get a random move represented as an integer list"""
    #print('init', bounds)
    assert isinstance(bounds[0], int), bounds[0]
    count = len(bounds)
    assert count == 8, bounds
    ret = []
    move_count = None
    while move_count is None: # select a piece with legal moves
        piece_idx = np.random.randint(0, bounds[0])
        assert piece_idx < bounds[0], (bounds[0], piece_idx)
        move_count = bounds[1][piece_idx]
    ret.append(piece_idx)
    piece_idx = ret[0]
    bound = bounds[1][piece_idx]
    if not bound:
        print("no legal moves")
        return None
    ret.append(np.random.randint(0, bound)) # select random move
    # select the rest
    assert ret[0] < bounds[0]
    bounds = bounds[2:]
    for bound in bounds:
        if not bound:
            ret.append(None)
        else:
            ret.append(np.random.randint(0, bound))
    assert len(ret) == 8, ret
    for i in ret:
        assert isinstance(i, int) or i is None, (i, ret)
    ret[-2] = 'r' if ret[-2] else 'l'
    return ret

@PROFILE
def get_move(whose_turn_is_it, grid, skip_color_pos, random_move=False, from_seq=()):
    """get the move from the user
    from_seq is an ordered list of integers
    0: piece to move (from the list of on the board pieces)
    1: index of a legal move (not including gifts) for that piece (from list of legal moves) 
    2: demote pawn?  0 no 1 yes
    3: piece from other team to give back (from list of off the board pieces)
    4: reverse en passant (0: pos (no en passant), 1,2: en passant)
    5: promotion piece 0:'q', 1:'n', 2:'b', 3:'r'
    6: leftright (0=left 1=right)
    7: promote pawn? 0 no 1 yes
    """
    # input piece to move TODO
    global BOARD
    global wpieces
    global bpieces
    #print('from', from_seq)
    #print('to', seq_bounds(whose_turn_is_it))
    if from_seq:
        assert len(from_seq) == 8, from_seq
    if skip_color_pos is None and not random_move and not from_seq:
        pos, leftright = mouse()
    elif random_move and skip_color_pos is None:
        pos, leftright = get_piece(whose_turn_is_it).pos, None
    elif from_seq and skip_color_pos is None:
        idx = from_seq[0]
        piece = get_piece(whose_turn_is_it, idx=idx)
        if piece is not None:
            pos, leftright = piece.pos, from_seq[6] # seq0
        else:
            print('selected piece is None')
            return None, False, None
    else:
        assert skip_color_pos is not None
        pos, leftright = skip_color_pos, None

    leftright = 'l' if allonboard(wpieces if whose_turn_is_it == 'b' else bpieces) else leftright
    piece_to_move = BOARD[pos]
    # quick checks:
    legal = piece_to_move is not None and not piece_to_move.locked
    legal = legal and whose_turn_is_it == piece_to_move.team
    legal = legal and not piece_to_move.pawn_is_home()
    if not legal or piece_to_move is None:
        if not GENETICS:
            print("wrong team/pawn at home.",
                full(whose_turn_is_it), "to move", skip_color_pos, pos)
            if piece_to_move is not None:
                piece_to_move.ident()
            print("END wrong team/pawn at home.")
        return None, False, None
    if not random_move and not from_seq:
        # highlight the piece to move position
        highlight(pos, grid, YELLOW)
        # highlight the legal moves
        highlight_legal(piece_to_move, grid)

    # input new position
    #if REVERSE_CHECK is None:
    #    newpos, _ = mouse()
    #else:
    lr2 = None
    if not random_move and not from_seq:
        newpos, lr2 = mouse()
        leftright = 'r' if lr2 == 'r' else leftright
        #leftright = leftright if skip_color_pos is None else lr2
    elif random_move:
        newpos, _ = get_legal_move(piece_to_move)
    else:
        idx = from_seq[1]
        newpos, _ = get_legal_move(piece_to_move, idx=idx) # seq1
        leftright = leftright if skip_color_pos is None else from_seq[6]
        if newpos is None: # selected piece has no legal moves
            for i in range(100):
                newpos, _ = get_legal_move(piece_to_move, idx=i) # seq1
                if newpos is not None:
                    break
            if newpos is None: # selected piece has no legal moves
                if not GENETICS:
                    print("no legal moves for")
                    piece_to_move.ident()
                    print("END no legal moves for")
                return None, False, None


    #highlight(newpos, grid, BLUE)

    # check the legality of the move
    legal, forced_give, reverse_castle = check_legal_move(piece_to_move.ptype, pos, newpos, prin=True)
    if not legal and not GENETICS:
        print('not legal:')
        piece_to_move.ident()
        print('END not legal:')

    # special case:
    # check to see if a promoted piece is moving like a pawn from a promotion row
    # in this case, the move is legal if the piece is demoted back to a pawn
    # it can also just be legal on its own, which is why the program asks
    demote = False
    if piece_to_move.promoted and (
            (pos[0] == 0 and piece_to_move.team == 'w') or (
                pos[0] == 7 and piece_to_move.team == 'b')):
        legal2, forced_give2, reverse_castle2  = check_legal_move('p', pos, newpos)
        assert not reverse_castle2, "promoted pieces should not be involved in castling"
        if legal2:
            if not random_move and not from_seq:
                demote = messagebox.askyesno(title='Demote?', message='Demote promoted piece back to pawn?')
            elif from_seq:
                demote = from_seq[2] # seq2
            else:
                demote = np.random.randint(0,2)
            if demote:
                legal, forced_give, reverse_castle = legal2, forced_give2, reverse_castle2

    if not legal:
        if not GENETICS:
            print("illegal move:", pos, newpos, piece_to_move.ptype)
        return None, False, None

    # ask for gift
    spawn_piece, spawn_pos = None, None
    pieces = wpieces
    if not reverse_castle and ((forced_give and (piece_to_move.ptype == 'p' or piece_to_move.promoted)) or (
            leftright == 'r' and piece_to_move.ptype != 'p')):
        if not random_move and not from_seq:
            spawn_piece = givebox(wpieces if whose_turn_is_it == 'b'\
                                  else bpieces)
        elif random_move:
            spawn_piece = get_piece(other_team(whose_turn_is_it), ontheboard=False)
        else:
            idx = from_seq[3]
            if idx is not None:
                spawn_piece = get_piece(other_team(whose_turn_is_it), ontheboard=False, idx=idx) # seq3
    if piece_to_move.ptype == 'p' and spawn_piece is not None:
        assert pos[1] != newpos[1], (pos, newpos)
    give_params = GiveParams(None, None, None)
    reverse_en_passant = False
    if spawn_piece is not None:
        if reverse_en_passant_possible(pos, newpos, piece_to_move, spawn_piece):
            rev_list = [pos, (pos[0]-1, pos[1]), (pos[0]+1, pos[1])]
            if not random_move and not from_seq:
                messagebox.showinfo(title="Reverse en-passant!",
                                    message="Select a green square to do a reverse en passant move"+\
                                    " (and get an extra turn)."+\
                                    "  Select the yellow square for the regular reverse capture.")
                highlight((pos[0]-1, pos[1]), grid, GREEN)
                highlight((pos[0]+1, pos[1]), grid, GREEN)
                spawn_pos, _ = mouse()
            elif random_move:
                spawn_pos = random.choice(rev_list)
            else:
                spawn_pos = rev_list[from_seq[4]] # seq4
            if spawn_pos != pos:
                reverse_en_passant = True
                if spawn_pos[1] != pos[1] or spawn_pos[0] not in (pos[0]+1, pos[0]-1):
                    print("illegal en passant selection")
                    return None, False, None
        else:
            spawn_pos = tuple(pos)
        highlight(spawn_pos, grid, BLUE)

        # promotion
        if spawn_piece.ptype == 'p' and not reverse_en_passant:
            promote = (spawn_pos[0] == 0 and spawn_piece.team == 'w') or (spawn_pos[0] == 7 and spawn_piece.team == 'b')
            if not promote:
                if not random_move and not from_seq:
                    promote = messagebox.askyesno(title='Promote?', message='Promote pawn you are giving back?')
                elif from_seq:
                    promote = from_seq[7]
                else:
                    promote = np.random.randint(0, 2)
            if promote:
                promote = ['q', 'n', 'b', 'r']
                if not random_move and not from_seq:
                    promote = givebox(wpieces if whose_turn_is_it == 'b' else bpieces, ignore_count=True)
                else:
                    if not from_seq:
                        assert random_choice
                        promote = random.choice(promote)
                    else:
                        promote = promote[from_seq[5]] # seq5
                give_params.promote_type = promote if not isinstance(promote, list) else None

        give_params = GiveParams(spawn_piece, spawn_pos, give_params.promote_type)
        # check legality of that gift
        legal = check_legal_give(give_params, piece_to_move, newpos)
        reverse_en_passant, give_params = reverse_en_passant_check(give_params, piece_to_move, newpos)
        if not legal:
            if not GENETICS:
                print("likely illegal gift")
            return None, False, None
    ret =  Move(piece_to_move.ptype, piece_to_move.pos,
                newpos, give_params, whose_turn_is_it, reverse_castle, demote)
    if spawn_piece is not None:
        if spawn_piece.promoted:
            assert ret.give_params.spawn_piece.promoted
    legal = legal and (spawn_piece is not None or not forced_give)
    if random_move:
        print("rand:", newpos)
        piece_to_move.ident()
        print("END rand:", newpos)
    if legal:
        assert ret is not None
    return ret, legal, reverse_en_passant


@PROFILE
def reverse_en_passant_possible(pos, newpos, piece_to_move, spawn_piece):
    """check if a proposed move results in possible reverse en passant"""
    ret = False
    row, col = pos
    row1, col1 = newpos
    if piece_to_move.ptype == 'p' and spawn_piece.ptype == 'p':
        ret = spawn_piece.team == 'b' and row == 2 and row1 == 3
        ret = ret or spawn_piece.team == 'w' and row == 5 and row1 == 4
        #assert newpos[1] != pos[1], (pos, newpos) # sanity check on legality of pawn movement
    if ret:
        # pawn that spawns must be able to spawn and then move back two spaces
        # so those spaces must be unoccupied
        # (pos is unoccupied due to pawn moving out of the way to newpos)
        ret = BOARD[(pos[0]-1, pos[1])] is None and BOARD[(pos[0]+1, pos[1])] is None
    return ret
        

@PROFILE
def reverse_en_passant_check(give_params, piece, newpos):
    """check to see if the gift implies en passant
    if so, the giving team gets an extra turn"""
    oteam = give_params.spawn_piece.team
    row1, col1 = give_params.spawn_pos
    row, col = piece.pos
    nrow, ncol = newpos

    ret = give_params.spawn_piece.ptype == 'p' and piece.ptype == 'p'
    if ret:
        # placed in the original column
        ret = col1 == col
    if ret:
        # row check
        if oteam == 'w':
            ret = row1 == nrow or row1 == nrow + 2
            ret = ret and nrow == 4
        else:
            ret = row1 == nrow or row1 == nrow - 2
            ret = ret and nrow == 3
    if ret:
        # sanity check:  gifting piece should be one away
        # from the other team's pawn's home row
        test = row == 5 if oteam == 'w' else row == 2
        assert test, piece.pos
        # skip the turn,
        # placing the gifted pawn on the proper home row
        spawn_pos = (6, col1) if oteam == 'w' else (1, col1)
        give_params.spawn_pos = spawn_pos
    return ret, give_params

@PROFILE
def game(invec=None, gen=None):
    """keeps track of game state"""
    # init the board
    global BOARD
    global SNAP
    global bpieces
    global wpieces
    BOARD = [[None for i in range(8)] for j in range(8)]
    BOARD = np.array(BOARD)
    clear_caches()
    assert not get_legal_moves.cache
    bpieces = Pieces('b')
    wpieces = Pieces('w')
    # draw the grid.
    # grid is the visual version of BOARD
    # update display after grid changes with update_display(WIN, grid, 8, WIDTH)
    grid = make_grid(8, WIDTH)
    update_display(grid, 8, WIDTH)
    if not DRAW:
        pygame.display.iconify()
    # get the bids
    #bbid, wbid = get_bids()
    bbid, wbid = 0, 0  # bid mode is deprecated for now

    # if bids are equal, black moves first
    whose_turn_is_it = 'b'
    if bbid < wbid:
        whose_turn_is_it = 'w'

    # game action
    fifty_move_count = 0
    winner = None
    snaps = {}
    if invec is not None:
        invec = np.array_split(invec, NUM_GENES)
    count = -1
    while winner is None:
        if not GENETICS:
            print(full(whose_turn_is_it), "to move")

        legal_count, only_legal = count_legal(whose_turn_is_it)
        skip_color_pos = None
        if len(only_legal) == 1: # only one piece has a legal move
            highlight(only_legal[0].pos, grid, YELLOW)
            skip_color_pos = only_legal[0].pos

        if not legal_count: # no legal moves for whose_turn_is_it
            # winner is either the team with no legal moves if they are at home
            # or the other team
            print("no legal moves for", full(whose_turn_is_it))
            winner = whose_turn_is_it if check_for_win(
                bpieces if whose_turn_is_it == 'b' else wpieces) else other_team(whose_turn_is_it)
            break

        # check the proposed move
        legal = False
        tried = False
        next_move_human = False
        if whose_turn_is_it == 'b':
            count += 1
            if invec is not None:
                if count >= len(invec):
                    tried = True
        bounds = seq_bounds(whose_turn_is_it)
        legal_loop_count = -1
        game.computer_move_cache = set()
        cache_hit = 0
        while not legal:
            legal_loop_count += 1

            #print("only legal:")
            #for i in only_legal:
            #    i.ident()

            # check to see if the team whose turn it is has any legal moves 
            # proceed with asking user for move
            #from_seq = () if whose_turn_is_it else random_move_from_seq('w')
            #from_seq = () if whose_turn_is_it == 'b' else random_move_from_seq('w')
            if COMPUTER_ONLY:
                if not tried and invec is not None and whose_turn_is_it == 'b':
                    bounds1 = list(bounds)
                    bounds1[1] = bounds1[1][min(bounds1[0]-1, int(invec[count][0]))] # piece selection
                    bounds1 = [np.nan if i is None else i for i in bounds1]
                    from_seq = [(int(invec[count][i]) if int(invec[count][i]) < bounds1[i] else bounds1[i]-1) for i in range(8)]
                    from_seq = [None if np.isnan(i) else i for i in from_seq]
                    from_seq[-2] = 'r' if from_seq[-2] else 'l'
                    from_seq2 = random_move_from_seq(whose_turn_is_it, bounds)
                    tried = True
                else:
                    from_seq = random_move_from_seq(whose_turn_is_it, bounds)
                    if next_move_human:
                        from_seq = ()
            else:
                if not ONLY_HUMAN:
                    from_seq = () if not whose_turn_is_it == 'w' else random_move_from_seq(whose_turn_is_it, bounds)
                    if whose_turn_is_it == 'w':
                        if str(from_seq) in game.computer_move_cache:
                            cache_hit += 1
                            continue
                        if cache_hit > 1000:
                            print("computer can't think of a move to make, so computer loses.  This is a bug.  Please notify dev.")
                            winner = other_team(whose_turn_is_it)
                            return winner
                        game.computer_move_cache.add(str(from_seq))
                else:
                    from_seq = ()
            #if from_seq and str(from_seq) in tried:
            #    continue
            #tried.add(str(from_seq))
            #if sum_bounds(seq_bounds(whose_turn_is_it)) < len(tried):
                #print("no legal moves for", full(whose_turn_is_it))
                #winner = other_team(whose_turn_is_it)
                #break
            legal = from_seq is not None
            if not legal:
                continue
            #print("getting move for", full(whose_turn_is_it), "skip color pos", skip_color_pos)
            move, legal, reverse_en_passant = get_move(whose_turn_is_it, grid,
                                                       skip_color_pos,
                                                       from_seq=from_seq)
                                                       #random_move=whose_turn_is_it=='w')
            # reset the board colors
            #if move is not None:
            #    move.dump()

            if move is not None and legal:
                move.execute()
                legal = count_legal(other_team(whose_turn_is_it))[0]
                if not legal and legal_loop_count > 100:
                    if GENETICS or (not HUMAN_ONLY and whose_turn_is_it == 'w'):
                        print("draw:", full(whose_turn_is_it),
                              "has only one legal move which results in no legal moves for",
                              full(other_team(whose_turn_is_it)))
                        return None
                    elif COMPUTER_ONLY:
                        next_move_human = True
                    print("move results in no legal moves for other team")
                    print("which I've decided is not legal")
                move.undo()
            remove_highlight(grid, skip_color_pos=skip_color_pos)
            update_display(grid, 8, WIDTH)

        # make the move, update the board
        if winner is not None:
            break
        move.execute()
        remove_highlight(grid)
        update_display(grid, 8, WIDTH)
        SNAP = board_snapshot()
        snap = SNAP

        # check for a draw
        if move.give_params.spawn_piece is None and move.ptype != 'p':
            fifty_move_count += 1
        else:
            fifty_move_count = 0
        if fifty_move_count == 50:
            print("draw. no pawn moves or gifts have occurred in fifty moves")
            winner = None
            break
        #snap = board_snapshot()
        if snap not in snaps:
            snaps[snap] = 1
        else:
            snaps[snap] += 1
            if snaps[snap] == 3:
                print("draw. three-fold repetition rule")
                winner = None
                break

        # a move happened, so decrement the bids
        if whose_turn_is_it == 'b':
            bbid -= 1
        else:
            wbid -= 1

        # check for win/loss
        #bwinning = check_winning(bpieces) or check_losing(wpieces)
        bwinning = check_for_win(bpieces) or check_losing(wpieces)
        #wwinning = check_winning(wpieces) or check_losing(bpieces)
        wwinning = check_for_win(wpieces) or check_losing(bpieces)
        assert not (bwinning and wwinning) # sanity check
        winner = 'b' if bwinning else winner
        winner = 'w' if wwinning else winner

        # swap turns
        whose_turn_is_it = turn(whose_turn_is_it, reverse_en_passant)

    if DRAW and not GENETICS:
        input(str(full(winner))+" wins.")
    return winner
game.computer_move_cache = set()

@PROFILE
def check_for_win(pieces):
    """first team to move all their
    pieces that are on the board
    back to their starting chess configuration wins.
    """
    ret = True
    team = pieces[0].team
    for piece in pieces:
        if not piece.ontheboard:
            continue
        if not piece.is_home():
            #print("not home:", piece.team, piece.ptype, piece.pos, piece.home)
            ret = False
            break
    if ret:
        print(full(team), "has all pieces on the board at home.")
        print("all on the board?", allonboard(pieces))
    return ret

@PROFILE
def check_losing(team_pieces):
    """check if team is losing"""
    # check to see if all the pawns are home
    global wpieces
    global bpieces
    ret = not allhome(team_pieces) # losing if ret = True
    team = team_pieces[0].team
    oteam = other_team(team)
    opieces = wpieces if oteam == 'w' else bpieces
    for piece in team_pieces:
        if not ret:
            break
        if not piece.ontheboard:
            if piece.ptype == 'p' or piece.promoted:
                ret = False
                break
            else:
                continue
        if not (piece.ptype == 'p' or piece.promoted):
            continue
        if not piece.is_home():
            ret = False
            break
    if ret:
        # all the pawns are home, is this a problem?
        all_home_but_knights = True
        for piece in team_pieces:
            if piece.ptype == 'n' or not piece.ontheboard:
                continue
            if not piece.is_home():
                all_home_but_knights = False
                break
        ret = not all_home_but_knights 
        if ret:
            print("pawn wall-off prevents", full(team),
                  "from winning, so", full(team), "loses.") 
    if not ret:
        ret = not pawn_scan(team)
        if ret:
            print(full(team), "loses; no way for all their pawns to get home")
            print(full(other_team(team)), "all pieces on the board:", allonboard(opieces))
    if not ret:
        ret = not bishop_wall_check(team)
        if ret:
            print(full(team), "loses; no way for all their bishops to get home")
    return ret

@PROFILE
def alllocked(pieces):
    """check if there are no possible moves"""
    ret = True
    for piece in pieces:
        ret = ret and (piece.locked or not piece.ontheboard)
    return ret

@PROFILE
def turn(whose_turn_is_it, reverse_en_passant):
    """swap turns unless reverse en passant"""
    assert reverse_en_passant is not None
    if not reverse_en_passant:
        if whose_turn_is_it == 'b':
            whose_turn_is_it = 'w'
        else:
            assert whose_turn_is_it == 'w', whose_turn_is_it
            whose_turn_is_it = 'b'
    return whose_turn_is_it


def main():
    if GENETICS:
        num_generations = 5
        num_parents_mating = 2
        sol_per_pop = 5 # number of 10 game matches
        num_genes = 8*NUM_GENES # number of moves
        init_range_low = 0
        init_range_high = 20
        mutation_percent_genes = 30
        ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating, 
                            fitness_func=fitness,
                            sol_per_pop=sol_per_pop, 
                            num_genes=num_genes,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            gene_type=int,
                            on_generation=pr_gen,
                            mutation_percent_genes=mutation_percent_genes)
        ga_instance.run()
        ga_instance.plot_result()
    else:
        game()

def pr_gen(ga_instance):
    """
    print the generation number
    """
    pr_gen.gen += 1
    print(pr_gen.gen)
    print(fitness.scores)
pr_gen.gen = 0

    #winner = game()
    #if winner in ('b', 'w'):
    #    print(full(winner), "wins the game.  congratulations?")
    #else:
    #    print('')

# genetic algorithm
def fitness(solution, solution_idx):
    ret = []
    for i in range(10):
        winner = game(solution, solution_idx)
        print("winner:", full(winner))
        if winner == 'b':
            score = 1
        elif winner is None:
            score = 0
        else:
            score = -1
        fitness.scores.append(score)
        ret.append(score)
    ret = np.mean(ret)
    return ret
fitness.scores = []

if __name__ == '__main__':
    main()
