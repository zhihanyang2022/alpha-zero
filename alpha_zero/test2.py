import numpy as np


np.random.seed(42)

board = np.random.randint(low=0, high=2, size=(6, 6))

print(board)
print(np.diagonal(board, offset=1))



