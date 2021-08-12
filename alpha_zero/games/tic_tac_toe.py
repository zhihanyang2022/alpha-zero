import numpy as np

from games.abstract_game import Game


def check_connect3(array):
    filled = 0 not in array
    uniquely = np.unique(array).size == 1
    winner = None if not (filled and uniquely) else np.unique(array)[0]
    return filled and uniquely, winner


class TicTacToe(Game):

    def __init__(self):
        super().__init__()
        self.board = np.zeros((3, 3))
        self.players = [-1, 1]
        self.current_player = 1  # np.random.choice(self.players)

    def get_previous_player(self) -> int:
        return self.current_player * -1

    def get_current_player(self):
        return self.current_player

    def evolve(self, move: tuple) -> tuple:

        assert move in self.get_valid_moves(), f"Move {move} is not valid."

        self.board[move] = self.current_player
        self.current_player = 1 if self.current_player == -1 else -1

        for row in self.board:
            filled_uniquely, winner = check_connect3(row)
            if filled_uniquely:
                return True, winner

        for col in self.board.T:
            filled_uniquely, winner = check_connect3(col)
            if filled_uniquely:
                return True, winner

        diag = np.diagonal(self.board)
        filled_uniquely, winner = check_connect3(diag)
        if filled_uniquely:
            return True, winner

        oppo_diag = np.diagonal(np.fliplr(self.board))
        filled_uniquely, winner = check_connect3(oppo_diag)
        if filled_uniquely:
            return True, winner

        if 0 not in np.unique(self.board):
            return True, 0  # 0 means nobody is winner

        return False, None

    def get_valid_moves(self) -> list:
        advanced_indices = np.where(self.board == 0)
        moves = list(zip(advanced_indices[0], advanced_indices[1]))
        return moves

    def __repr__(self):
        return np.array2string(self.board)
