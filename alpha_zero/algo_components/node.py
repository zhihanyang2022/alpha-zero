import numpy as np


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

        self.c_puct = 3

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

    def get_move(self):
        assert self.is_root()
        best_move_so_far = None
        best_cnt_so_far = -np.inf
        for move, node in self.children.items():
            # print(move, node.get_value(), node.value_sum / node.visit_cnt, node.visit_cnt)
            if node.visit_cnt > best_cnt_so_far:
                best_move_so_far = move
                best_cnt_so_far = node.visit_cnt
        return best_move_so_far
