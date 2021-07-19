from abc import ABC, abstractmethod


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
