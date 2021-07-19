import numpy as np

from games.abstract_game import Game


class Gomoku(Game):

    def get_current_player(self) -> int:
        pass

    def evolve(self, move) -> tuple:
        pass

    def get_valid_moves(self) -> list:
        pass
