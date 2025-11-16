#!/usr/bin/python3

import sys

def tenant_matching2(possible_tenants, mins=None, exclude=None):
    """each apartment has some list of possible tenants
    can every apartment find a tenant?
    smart solution
    """
    assert None, "lol; i couldn't make it work, too hard"
    # setup
    if mins is None:
        exclude = set() if exclude is None else exclude
    else:
        exclude = {} if exclude is None else exclude
    solution = {}

    # select pawn
    if possible_tenants:
        col = next(iter(possible_tenants))
    else:
        return {}
    tenants = set(possible_tenants[col]) # tenant=col, exclude is used columns

    if mins is not None: # this is the part that probably doesn't scale
        diffs = mins[col] # column difference
        # sort by distance between pawn's column and proposed matching column

        offby = -1
        # exclude all columns which are already matched to closer pawns
        last = None
        for tenant in diffs: # (maybe) inefficient part 1
            if tenant not in exclude:
                continue
            diff = diffs[tenant] - exclude[tenant]
            if diff >= 0:
                offby = max(diff, offby)
                last = tenant if offby == diff else last
                tenants -= {tenant}
            else:
                return None
        # every pawn must have a column, even if all columns are matched to closer pawns 
        if not tenants: 
            return None

        # sort by distance between pawn's column and proposed matching column
        sl2 = sorted(list(tenants), key=lambda tenant: diffs[tenant])

        # measure the maximum difference by which the previous pawns match better
        # if this gain is less than the loss we obtain by using the next best remaining column for this pawn
        # then it doesn't make sense to consider the previous pawns as well-matched
        # example:
        # r = {0: {0, 4}, 1:{0, 4}}
        # m = {0: {0:3, 4:1}, 1:{4:2, 0:6}}
        # m = {0: {0:3, 4:1}, 1:{4:2, 0:4}}
        # without this check, we obtain the wrong solution: {0: 4, 1: 0}
        # instead of the correct solution:  {0: 0, 1: 4}
        # NOT CORRECT
        if diffs[sl2[0]] > offby and offby > 0:
            return None

    else:
        # exclude columns which are already matched
        tenants = tenants-exclude
        sl2 = sorted(list(tenants))

    # treat current pawn as matched, match the rest
    passd = dict(possible_tenants)
    del passd[col]
    rest = None
    idx = 0
    while rest is None and idx < len(tenants):

        # match the current pawn
        if mins is None:
            skip = set(exclude)
        else:
            skip = dict(exclude)
        try:
            if col == 1:
                print("pawn", col, "sol", sl2, "idx", idx)
            solution[col] = sl2[idx]
        except IndexError:
            print(sl2)
            print(solution)
            raise

        if mins is None:
            # add current matched column to list of skipped columns
            skip.add(sl2[idx])
        else:
            # inform the rest how good this match is
            if idx == len(tenants) - 1:
                skip[sl2[idx]] = -1*float('inf') # last solution is forced to accept
            else:
                skip[sl2[idx]] = diffs[sl2[idx]]
        # get the rest of the solution
        rest = tenant_matching(passd, exclude=skip, mins=mins)

        # if the solution does not exist,
        # try the next closest non-skipped column
        idx += 1

    if rest is None:
        # rest of the solution does not exist
        return None

    # concatenate the already matched pawns with the rest
    solution.update(rest)
    #break

    # check to make sure columns are only used once
    collision_chk(solution)
    return solution

def collision_chk(solution):
    rset = set()
    for val in solution.values():
        try:
            assert val not in rset
        except TypeError:
            val = str(val)
            assert val not in rset
        rset.add(val)

def tenant_matching_solutions(mins, used=None):
    """get all solutions to get pawns home"""
    if used is None:
        used = set()
    if not mins:
        return {}
    pawn = sorted(list(mins))[0]
    solution = {}
    sold = {}
    count = 0
    for col in mins[pawn]:
        if col in used:
            continue
        solution[pawn] = col
        dcopy = dict(mins)
        del dcopy[pawn]
        used2 = set(used)
        used2.add(col)
        if dcopy:
            rest = tenant_matching_solutions(dcopy, used=used2)
            for key in rest:
                item = rest[key]
                solution2 = dict(solution)
                solution2.update(item)
                collision_chk(solution2)
                key2 = str(solution2)
                if key2 not in sold:
                    sold[key2] = solution2
        else:
            count += 1
            solution2 = dict(solution)
            key = str(solution2)
            collision_chk(solution2)
            sold[key] = solution2
    if count:
        assert count == len(sold)
    collision_chk(sold)
    return sold

def tenant_matching(mins):
    """figure out which solution needs
    the smallest amount of column movement"""
    llen = len(mins)
    sols = tenant_matching_solutions(mins)
    total_sol = float('inf')
    solution = {}
    for sol in sols.values():
        if len(sol) < llen:
            continue
        total = 0
        for pawn in sol:
            total += mins[pawn][sol[pawn]]
        total_sol = min(total, total_sol)
        solution = sol if total == total_sol else solution
    return solution


if __name__ == '__main__':
    #a=tenant_matching({0:set([3,1,2]), 1:set([3,1]), 2:set([2,3]), 3:set([1,0])})
    #print(a)
    r = {0: {1}, 1: {5, 6, 7}, 2: {1, 2, 3, 4, 5}, 3: {1, 2, 3}} # {0: 1, 1: 5, 2: 2, 3: 3}
    m = {0: {1: 1}, 1: {5: 2, 6: 1, 7: 0}, 2: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 3: {1: 1, 2: 0, 3: 1}}
    r = {0: {3, 4}, 1: {0, 3, 4, 5, 6, 7}, 2: {3, 4, 5, 6, 7}, 3: {4, 5, 6, 7}, 4: {0, 3, 4, 5}, 5: {3, 4, 5, 6, 7}}
    m = {0: {3: 0, 4: 1}, 1: {0: 3, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4}, 2: {3: 3, 4: 2, 5: 1, 6: 0, 7: 1}, 3: {4: 2, 5: 1, 6: 0, 7: 1}, 4: {0: 0, 3: 3, 4: 4, 5: 5}, 5: {3: 2, 4: 1, 5: 0, 6: 1, 7: 2}}

    r = {0: {0, 4}, 1:{0, 4}}
    m = {0: {0:3, 4:1}, 1:{4:2, 0:6}}
    #solution = tenant_matching(r, mins=m)
    m = {0: {1: 1}, 1: {5: 2, 6: 1, 7: 0}, 2: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, 3: {1: 1, 2: 0, 3: 1}}
    # solution count:
    # 5, {2,3,4}, {{1,3}, {1,2}, {1,2,3}} # 7
    # 6, {2,3,4,5} {
    # 7
    solution = tenant_matching(m)
    print(solution)
    min_off_board = 0
    for pawn in solution:
        print("p, diff", pawn, m[pawn][solution[pawn]])
        min_off_board += m[pawn][solution[pawn]]
    print(min_off_board)

    # actual: {0:1, 1:5, 2:2, 3:3}

    #print(a) # prints {0: 1, 1: 3, 2: 2, 3: 0}
