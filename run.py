from Qnetwork2 import DQNAgent
from chessagent import ChessAgent1
import chess
import torch
board = chess.Board()

chessplayer = ChessAgent1(colour='WHITE')


target_net = DQNAgent()

chessplayer.QLEARNING(500, 0.5)

chessplayer.plotRewards('rewards')
chessplayer.plotgamelength('gamelength')

chessplayer.plotEpsilon('epsilon')


#%%





