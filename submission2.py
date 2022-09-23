from __future__ import annotations

import enum
import itertools
import random
import time
from collections import Generator
from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import List
from typing import Optional
from typing import Tuple

from isolation import Board


class OpenMoveEvalFn:
    def score(self, game: Board, my_player=None):
        """Score the current game state
        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.

        Note:
            If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                game (Board): The board and game state.
                my_player (Player object): This specifies which player you are.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        mym = len(game.get_player_moves(my_player))
        oppm = len(game.get_opponent_moves(my_player))

        return mym - oppm


class BailException(Exception):
    pass


QuickBoardPos = Tuple[int, int]
AvailableMove = Tuple[QuickBoardPos, QuickBoardPos]
AvailableMoves = List[AvailableMove]

MIN_SCORE = -100000
MAX_SCORE = 100000
WINNING_SCORE = MAX_SCORE / 2


class QuickBoard:
    """slowest part of my code is the isolation.py implementation..., make my own"""
    state: bytearray # TODO can actually just store empty/not empty as bool, investigate speed later
    p_my_pos: QuickBoardPos
    p_o_pos: QuickBoardPos
    my_turn: bool

    __slots__ = ["state", "p_my_pos", "p_o_pos", "my_turn"]

    STATE_USED: ClassVar[int] = 0
    STATE_P_ME: ClassVar[int] = 1
    STATE_P_OTHER: ClassVar[int] = 2
    STATE_EMPTY: ClassVar[int] = 3
    STATE_UNK: ClassVar[int] = 5

    def get_cell(self, x: int, y: int):
        return self.state[x + y * 7]

    def set_cell(self, x: int, y: int, value: int):
        self.state[x + y * 7] = value

    def get_active_moves(self):
        if self.my_turn:
            return self.get_moves(self.p_my_pos)
        else:
            return self.get_moves(self.p_o_pos)

    def get_inactive_moves(self):
        if self.my_turn:
            return self.get_moves(self.p_o_pos)
        else:
            return self.get_moves(self.p_my_pos)

    def get_moves(self, pos: QuickBoardPos) -> AvailableMoves:
        oy, ox = pos
        st, empty = self.state, self.STATE_EMPTY
        avail = []

        def ray(dx: int, dy: int):
            for i in range(1, 7):
                x = ox + dx * i
                y = oy + dy * i
                if 0 <= x <= 6 and 0 <= y <= 6 and self.get_cell(x=x, y=y) == empty:
                    avail.append(((y, x), (y - dy, x - dx)))
                else:
                    break

        for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ray(x, y)
        return avail

    def __init__(self, board: Board = None, player=None, copy_of: QuickBoard = None) -> None:
        if copy_of is not None:
            self.state = copy_of.state.copy()
            self.my_turn = copy_of.my_turn
            self.p_my_pos, self.p_o_pos = copy_of.p_my_pos, copy_of.p_o_pos
            return

        if board.width != board.height:
            raise RuntimeError("board height != width")
        self.state = bytearray(49)
        self.my_turn = True

        my_y, my_x = board.get_player_position(my_player=player)
        o_y, o_x = board.get_opponent_position(my_player=player)

        s = board.get_state()
        for row_num, row in enumerate(s):
            for col_num, cell in enumerate(row):
                x = col_num
                y = row_num
                if cell == Board.BLANK:
                    v = self.STATE_EMPTY
                elif cell == Board.BLOCKED or cell == Board.TRAIL:
                    v = self.STATE_USED
                else:
                    v = self.STATE_UNK
                self.set_cell(x=x, y=y, value=v)

        self.set_cell(x=my_x, y=my_y, value=self.STATE_P_ME)
        self.set_cell(x=o_x, y=o_y, value=self.STATE_P_OTHER)

        self.p_my_pos = my_y, my_x
        self.p_o_pos = o_y, o_x

    @staticmethod
    def other_player(player: int):
        return player ^ 3

    def forecast_move(self, move: AvailableMove) -> QuickBoard:
        new = QuickBoard(copy_of=self)
        new.my_turn = not new.my_turn

        if self.my_turn:
            py, px = self.p_my_pos
            new.p_my_pos, (sy, sx) = move
            my, mx = new.p_my_pos
        else:
            py, px = self.p_o_pos
            new.p_o_pos, (sy, sx) = move
            my, mx = new.p_o_pos

        new.set_cell(x=px, y=py, value=self.STATE_USED)
        new.set_cell(x=sx, y=sy, value=self.STATE_USED)
        new.set_cell(x=mx, y=my, value=self.STATE_USED)

        return new, None, None


def calc_score(game: Board, is_my_move: bool,
               depth: int,
               my_moves: AvailableMoves = None,
               o_moves: AvailableMoves = None):
    """Score the current game state.
    Returns:
        float: The current state's score, based on your own heuristic.
    """

    if my_moves is None:
        if is_my_move:
            my_moves = game.get_active_moves()
        else:
            my_moves = game.get_inactive_moves()

    if o_moves is None:
        if is_my_move:
            o_moves = game.get_inactive_moves()
        else:
            o_moves = game.get_active_moves()

    my_count = len(my_moves)
    o_count = len(o_moves)

    if (is_my_move and my_count == 0) or (not is_my_move and o_count > 0 and my_count == 0):
        # a loss is found, optimize to maximize depth and minimize opponent moves and max my other piece moves
        return MIN_SCORE / 2 - o_count + depth * 20

    if (is_my_move and my_count > 0 and o_count == 0) or (not is_my_move and o_count == 0):
        return WINNING_SCORE - depth

    return my_count - o_count


class MiniMaxTree:
    max_depth: int
    bail: Callable[[], bool]
    omef: CustomEvalFn

    def __init__(self, max_depth: int, bail, player) -> None:
        self.max_depth = max_depth
        self.omef = CustomEvalFn(player)
        self.bail = bail

    def check_bail(self):
        if self.bail():
            raise BailException()

    def explore(self, board: Board, depth: int, do_max: bool,
                vmax: int = MIN_SCORE, vmin: int = MAX_SCORE):
        self.check_bail()

        score = MIN_SCORE if do_max else MAX_SCORE

        moves = board.get_active_moves()

        def get_score():
            if do_max:
                return calc_score(board, is_my_move=do_max, my_moves=moves, depth=depth)
            return calc_score(board, is_my_move=do_max, o_moves=moves, depth=depth)

        if depth > self.max_depth:
            return get_score(), None, self.max_depth
        result = None
        md = 0
        if len(moves) == 0:
            return get_score(), None, depth - 1
        # tmp = []
        for i, move in enumerate(moves):
            run, _, d = self.explore(
                board=board.forecast_move(move)[0],
                depth=depth + 1,
                vmax=score if do_max else MIN_SCORE,
                vmin=MAX_SCORE if do_max else score,
                do_max=not do_max
            )
            md = max(d, md)

            if do_max:
                if run > score:
                    score = run
                    result = move
                if run > vmin:
                    break
            else:
                if run < score:
                    score = run
                    result = move
                if score < vmax:
                    break
        return score, result, md


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: ((int,int),(int,int),(int,int)): Your best move
        """

        def should_bail():
            return time_left() < 20

        best_move = None

        qb = QuickBoard(game, player=self)

        md = 0
        try:
            # print(game.print_board())
            for i in range(3, 50):  # pretty confident I can always go depth 3
                # print(f"############### ITERDEPTH={i}")
                tree = MiniMaxTree(max_depth=i, bail=should_bail, player=self)
                score, (best_move, _), md = tree.explore(qb, depth=0, do_max=True)
                if abs(score) > 100:
                    if score > 0:
                        # print(f"Expect guarenteed WIN {score} {describe_move('A', best_move)} in {i} moves")
                        break
                    elif score < 0:
                        # print(f"Expect guarenteed LOSS @ depth={i} {score} {describe_move('A', best_move)}")
                        break
                if score >= WINNING_SCORE or md != i:
                    # print("FINISHED MOVE STEP")
                    break
        except BailException:
            # print(f"BAILED MOVE STEP {md}")
            pass

        if best_move is None:
            return [None, None, None]

        # print(f"OFFER MOVE= {best_move}")
        print(md)
        return best_move


@dataclass
class CustomEvalFn:
    player: Any

    def score(self, game: Board, is_my_move: bool,
              depth: int,
              my_moves: AvailableMoves = None,
              o_moves: AvailableMoves = None):
        """Score the current game state.
        Returns:
            float: The current state's score, based on your own heuristic.
        """

        if my_moves is None:
            if is_my_move:
                my_moves = game.get_active_moves()
            else:
                my_moves = game.get_inactive_moves()

        if o_moves is None:
            if is_my_move:
                o_moves = game.get_inactive_moves()
            else:
                o_moves = game.get_active_moves()

        my_count = len(my_moves)
        o_count = len(o_moves)

        if (is_my_move and my_count == 0) or (not is_my_move and o_count > 0 and my_count == 0):
            # a loss is found, optimize to maximize depth and minimize opponent moves and max my other piece moves
            return MIN_SCORE / 2 - o_count + depth * 20

        if (is_my_move and my_count > 0 and o_count == 0) or (not is_my_move and o_count == 0):
            return WINNING_SCORE - depth

        return my_count - o_count


def describe_hist(history):
    x = []
    for h in history:
        if len(h) == 1:
            x.append(describe_move("A", h[0][0]))
        else:
            a, b = h
            x.append(describe_move("A", a[0]))
            x.append(describe_move("B", b[0]))
    return ", ".join(x)


def describe_move(pre, move):
    try:
        (a1, a2) = move
        return f"{pre}{a1}{a2}"
    except TypeError:
        return "ILLEGAL_MOVE"
