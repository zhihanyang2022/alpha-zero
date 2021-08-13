import numpy as np
import random


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node:

    """
    Represent a node (a board state) in the MCTS search tree.
    Nodes are connected by moves as edges.
    """

    def __init__(self, parent, prior_prob):

        # for connecting to other nodes in the tree
        self.parent = parent
        self.children = {}

        # for evaluating the value of this node
        self.prior_prob = prior_prob
        self.visit_cnt = 0
        self.value_sum = 0

        # flags for sanity check
        self.expanded = False

        self.c_puct = 5

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def expand(self, guide: dict):
        assert not self.expanded
        for move, prob in zip(guide['moves'], guide['probs']):
            self.children[move] = Node(parent=self, prior_prob=prob)
        self.expanded = True

    def backup(self, leaf_value: float):
        if not self.is_root():
            self.parent.backup(-leaf_value)  # assume that last action was taken by opponent
        self.visit_cnt += 1
        self.value_sum += leaf_value

    def select(self):
        return max(self.children.items(),  # each child is a (action, node) tuple
                   key=lambda action2node: action2node[1].get_value())

    def get_value(self):
        """Estimate the value of the current node using the PUCT algorithm. Helper method to self.select_child."""
        value = 0 if self.visit_cnt == 0 else self.value_sum / self.visit_cnt
        ucb = self.c_puct * self.prior_prob * np.sqrt(self.parent.visit_cnt) / (1 + self.visit_cnt)
        return value + ucb

    def get_move(self, temp):  # for evaluation
        assert self.is_root()
        assert 0 <= temp <= 1
        if temp > 0:
            moves, visit_cnts = [], []
            for move, child in self.children.items():
                moves.append(move)
                visit_cnts.append(child.visit_cnt)
            probs = softmax(1.0 / temp * np.log(np.array(visit_cnts) + 1e-10))
            assert np.allclose(np.sum(probs), 1)
            return random.choices(moves, weights=probs, k=1)[0]
        else:
            best_visit_cnt = -np.inf
            best_move = None
            for i, (move, child) in enumerate(self.children.items()):
                if child.visit_cnt > best_visit_cnt:
                    best_visit_cnt = child.visit_cnt
                    best_move = move
            return best_move

    def get_move_and_pi_vec(self, board_width, board_height, temp, alpha):  # for self-play (need pi_vec for later nn training)

        assert self.is_root()
        assert 0 <= temp <= 1
        assert temp > 0 and alpha is None  # first-phase of self-play; exploration based on visit counts
        assert temp == 0 and alpha is not None  # second-phase of self-play; exploration based on dirichlet noise

        if temp > 0:
            moves, visit_cnts = [], []
            for move, child in self.children.items():
                moves.append(move)
                visit_cnts.append(child.visit_cnt)
            probs = softmax(1.0 / temp * np.log(np.array(visit_cnts) + 1e-10))

        else:
            best_visit_cnt = -np.inf
            best_move_idx = None
            num_moves = 0
            moves = []
            for i, (move, child) in enumerate(self.children.items()):
                if child.visit_cnt > best_visit_cnt:
                    best_visit_cnt = child.visit_cnt
                    best_move_idx = i
                num_moves += 1
                moves.append(move)
            probs = np.zeros((num_moves, ))
            probs[best_move_idx] = 1
            probs = probs * 0.75 + np.random.dirichlet([alpha] * num_moves) * 0.25
            assert np.allclose(np.sum(probs), 1)

        pi_vec = np.zeros((board_width * board_height,))
        for move, prob in zip(moves, probs):
            index = move[0] * board_width + move[1]
            pi_vec[index] = prob

        return random.choices(moves, weights=probs, k=1)[0], pi_vec
