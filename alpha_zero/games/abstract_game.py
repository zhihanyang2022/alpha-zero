from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):

    def __init__(self):
        self.board = None

    @abstractmethod
    def get_previous_player(self) -> int:
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        pass

    @abstractmethod
    def evolve(self, move) -> tuple:
        pass

    @abstractmethod
    def get_valid_moves(self) -> list:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
