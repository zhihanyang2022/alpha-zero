# AlphaZero for Connect4 从零开始学习下四子棋 🤔

**Author:** Zhihan Yang @ Carleton College (MN, USA)

**Keywords:** board game, MCTS, deep convolutional neural network, self-play, PyTorch

This is a learning resource, and likely will not be tractable for games using bigger boards. 

Feel free to ask questions through Github Issues.

**作者：** 杨之涵 @ 卡尔顿学院

**关键词：** 棋类游戏，蒙特卡洛树搜，卷积神经网络，自我对弈，PyTorch

这个代码库可以帮助大家了解AlphaZero，但是对于棋盘更大的游戏估计完全跑不动。

欢迎通过Github Issues提问。

## Why Connect4 为什么选择四子棋

Connect4 is a middle ground between Connect3 (tic-tac-toe) and Connect5 (Gomoku or Gobang). It is much more difficult than Connect3, but it is also much easier than Connect5. Also, in Connect4, if the first-hand player plays optimally, it will win for sure; this makes it easier to verify how well AlphaZero learned. Here, we use a 6x6 board for Connect4.

四子棋是三子棋（tic-tac-toe）和五子棋的过渡。之所以选择四子棋，是因为四子棋比三子棋难很多，但又比五子棋简单很多。此外，在四子棋中，一个完美玩家在先手的情况下可以百分百获胜，方便我们验证AlphaZero学习的结果。在这里，我们使用6x6的棋盘。

## Example game plays vs human 人机对弈结果

Before we talk about theory and code, let's see what AlphaZero can do after 3000 self-play games. 

During training, AlphaZero uses 500 MCTS simulations for each move; during evaluation (for the games below), AlphaZero uses 50-1000 MCTS iterations (randomly picked between this range) to induce some stochasticity. 

In all games below, AlphaZero is the first-hand player and holds the black stone, and the values in the background show the prior move probabilities predicted by the convolutional neural network. The human player (me) is the second-hand player and holds the white stone.

The final winning move AlphaZero is shadowed.

在讨论理论和代码之前，我们来看看AlphaZero在3000局self-play之后能达到什么样的效果。

在训练当中，AlphaZero每一步使用500次MCTS simuation；在测试中（以下对弈），AlphaZero每一步使用50到1000次（`np.random.randint`）MCTS simuation来产生一定的随机性。

在以下对弈中，AlphaZero是先手玩家并持有黑棋。棋盘上显示的数字是卷积神经网络预测出的prior move probabilities。人类玩家（我）是后手并持有白棋。

AlphaZero最后的放棋位置用深色标出。

Game 1:

![Image](readme_images/game1.png?raw=true)

Game 2:

![Image](readme_images/game2.png?raw=true)

Game 3:

![Image](readme_images/game3.png?raw=true)

## Theory tutorial 理论教程

A detailed tutorial is available in this repo; see `alphazero.pdf`.

这个代码库包含一份详细的理论教程，请见`alphazero.pdf`.

## Code tutorial 代码教程

## Potential improvements 可以提升的地方

## References 对我很有帮助的资源

