from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):

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
    def get_first_person_view(self) -> np.array:
        pass
