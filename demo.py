import copy
import gym
import chess
import gym_chess
from helpers import custom_reset, evaluate_minimax
from generatingEndgame import generate_endgame_FEN
from tqdm import tqdm

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

import random
import chess.svg
import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

env = gym.make('Chess-v0')
#env.reset = custom_reset
obs = env.reset()


class MainWindow(QWidget):
    def __init__(self, obs):
        super().__init__()

        self.setGeometry(100, 100, 1100, 1100)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)

        self.chessboard = obs

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def paintEvent(self, pos):
        self.chessboard = pos
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg) 

app = QApplication([])
window = MainWindow(obs)
window.show()
app.exec()