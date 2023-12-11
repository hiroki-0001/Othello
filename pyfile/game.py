import sys
import OthelloBoardSystem
import numpy as np

class Game(OthelloBoardSystem.RealBoard):
    
    def __init__(self):
        super().__init__()
        self.white_count = 2
        self.black_count = 2
        self.blank_count = 60
        
    def read_data(self, BoardData):
        self.cell = BoardData
        
    def pass_system(self):
        super().turnchange()
    
    def count_system(self):
        self.white_count = np.sum(self.cell == OthelloBoardSystem.DiscColor.WHITE)
        self.black_count = np.sum(self.cell == OthelloBoardSystem.DiscColor.BLACK)
        self.blank_count = np.sum(self.cell == OthelloBoardSystem.DiscColor.EMPTY)
        
    def game_set(self):
        print("game set !!!")
        self.count_system()
        print('white = ', self.white_count)
        print('black = ', self.black_count)
        if self.white_count > self.black_count:
            print('white WIN !!')
        if self.white_count < self.black_count:
            print('Black WIN !!')
        if self.white_count == self.black_count:
            print('Draw')
        sys.exit()
        
    def input_point(self):
        print('石を置く座標を(1~8で)入力してください。(x,y)=(9,9)でpass、(0,0)で終了します。')
        x = input('x >> ')
        y = input('y >> ')
        try:
            x = int(x) - 1
            y = int(y) - 1
        except:
            self.input_point()
        return x, y
    
    def one_turn_play(self):  # ①〜⑤と❶をまとめる（❶に関しては上に記述）　
        if super().check_put_place():  #  ❶ 盤面に石が置ける場所があるかどうか
            (x,y) = self.input_point()   #  ① 座標を入力
            super().put_stone(x,y)  # Boardクラスで作ったやつ。石をおいてひっ繰り返してTrueを返すか、何もせずFalseを返す
            if not super().put_stone(x,y):
                    if (x,y) == (8,8):       #  ② パスするとき
                        self.pass_system()
                    elif (x,y) == (-1,-1):   #  ③ ゲームをやめる時
                        self.game_set()
                    #石をおけない時は もう一度同じことをする
                    while False:
                        self.one_turn_play()     
        else:
            self.pass_system()
        
    def gameplay(self):
        while self.blank_count > 0:
            super().display()
            print('-----'*10)
            self.turn += 1
            print('turn : ', self.turn, end = '  ')
            if self.current == -1:
                print(', turn black')
            if self.current == 1:
                print(', turn white')
            self.one_turn_play()
            self.count_system()
            print('white : ', self.white_count, ', black : ', self.black_count, ', blank : ', self.blank_count)
        self.game_set()