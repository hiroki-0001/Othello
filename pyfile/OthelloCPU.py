from OthelloBoardSystem import *
import random
import sys

# オセロ盤の評価値
board_evaluation_value = [
    [30, -12, 0, -1, -1, 0, -12, 30],
    [-12, -15, -3, -3, -3, -3, -15, -12],
    [0, -3, 0, -1, -1, 0, -3, 0,],
    [-1, -3, -1, -1, -1, -1, -3, -1],
    [-1, -3, -1, -1, -1, -1, -3, -1],
    [0, -3, 0, -1, -1, 0, -3, 0,],
    [-12, -15, -3, -3, -3, -3, -15, -12],
    [30, -12, 0, -1, -1, 0, -12, 30],
]

# 非常に大きな値
INF = 100000000 

class AI:
    
    def getdata(self, board_data):
        self.board_data = board_data
    
    def evaluation_value_calculation(self):
        for y in range(tablesize):
            for x in range(tablesize):
                val += board_evaluation_value[y][x] * self.board_data[y][x]
        return val

    # 1手読みの探索
    def search(board):
        max_score = -INF
        res = -1
        for coord in range(tablesize):
            if board.check_legal(coord):
                score = -evaluate(b.move(coord))
                if max_score < score:
                    max_score = score
                    res = coord
        return res
    
class RandomAI:
    
    def getdata(self, board_data):
        self.board_data = board_data
        
    def move(self):
        legal_list_y = []
        legal_list_x = []
        for y in range(tablesize):
            for x in range(tablesize):
                if(self.board_data[y][x] == DiscColor.LEGAL):
                    legal_list_y.append(y)
                    legal_list_x.append(x)
        
        if len(legal_list_y) > 0 and len(legal_list_x) > 0:
            n = random.randint(0, len(legal_list_y) - 1)
            return legal_list_y[n], legal_list_x[n]


