"""reverse chess objects used globally"""
PTYPES = ['p', 'k', 'r', 'b', 'q', 'n']

IMAGES = {}
for team in ['b', 'w']:
    for ptype in PTYPES:
        for boo in (True, False):
            IMAGES[(team, ptype, boo)] = './chessimages/'+team+'_'
    IMAGES[(team, 'p', False)] += 'pawn.png'
    IMAGES[(team, 'p', True)] += 'pawn_p.png'
    IMAGES[(team, 'k', False)] += 'king.png'
    IMAGES[(team, 'r', False)] += 'rook.png'
    IMAGES[(team, 'r', True)] += 'rook_p.png'
    IMAGES[(team, 'b', False)] += 'bishop.png'
    IMAGES[(team, 'b', True)] += 'bishop_p.png'
    IMAGES[(team, 'q', False)] += 'queen.png'
    IMAGES[(team, 'q', True)] += 'queen_p.png'
    IMAGES[(team, 'n', False)] += 'knight.png'
    IMAGES[(team, 'n', True)] += 'knight_p.png'

def offboard_count(pieces):
    """count the pieces not on the board
    of a specific team
    """
    ret = {}
    for piece in pieces:
        ptype = piece.ptype if not piece.promoted else 'p'
        if ptype not in ret:
            ret[ptype] = 0
        if not piece.ontheboard:
            ret[ptype] += 1
    if 'q' in ret:
        assert ret['q'] <= 1, ret
    if 'k' in ret:
        assert ret['k'] <= 1, ret
    if 'b' in ret:
        assert ret['b'] <= 2, ret
    if 'r' in ret:
        assert ret['r'] <= 2, ret
    if 'n' in ret:
        assert ret['n'] <= 2, ret
    return ret


def onboard_count(pieces):
    """count the pieces not on the board
    of a specific team
    """
    ret = {}
    for piece in pieces:
        ptype = piece.ptype if not piece.promoted else 'p'
        if ptype not in ret:
            ret[ptype] = 0
        if piece.ontheboard:
            ret[ptype] += 1
    return ret

