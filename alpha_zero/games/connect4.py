import numpy as np

from games.abstract_game import Game


def check_four_in_a_row(line_of_stones, target_stone):
    n_in_a_row = 0
    for s in line_of_stones:
        if s == target_stone:
            n_in_a_row += 1
            if n_in_a_row == 4:
                return True
        else:
            n_in_a_row = 0
    return False


def get_diagonal_offset(row_idx, col_idx):
    return col_idx - row_idx


class Connect4(Game):

    def __init__(self):
        super().__init__()
        self.board = np.zeros((6, 6))
        self.players = [-1, 1]
        self.current_player = 1

    def get_previous_player(self) -> int:
        return self.current_player * -1

    def get_current_player(self) -> int:
        return self.current_player

    def evolve(self, move) -> tuple:

        assert move in self.get_valid_moves(), f"Move {move} is not valid."

        self.board[move] = self.current_player

        row_idx, col_idx = move[0], move[1]
        diag_offset = get_diagonal_offset(row_idx, col_idx)

        row = self.board[row_idx]
        col = self.board[:, col_idx]
        diag = np.diagonal(self.board, offset=diag_offset)

        flipped_col_idx = 5 - col_idx  # row_idx remains the same
        flipped_diag_offset = get_diagonal_offset(row_idx, flipped_col_idx)
        oppo_diag = np.diagonal(np.fliplr(self.board), offset=flipped_diag_offset)

        if check_four_in_a_row(row, self.current_player) \
                or check_four_in_a_row(col, self.current_player) \
                or check_four_in_a_row(diag, self.current_player) \
                or check_four_in_a_row(oppo_diag, self.current_player):
            done, winner = True, self.current_player
        elif 0 not in self.board:  # filled
            done, winner = True, 0
        else:
            done, winner = False, None

        self.current_player = 1 if self.current_player == -1 else -1

        return done, winner

    def get_valid_moves(self) -> list:
        advanced_indices = np.where(self.board == 0)
        moves = list(zip(advanced_indices[0], advanced_indices[1]))
        return moves

    def __repr__(self):
        return np.array2string(self.board)
