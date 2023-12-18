import numpy as np
from enum import IntEnum

"""
定数
"""
# dx, dyは組み合わせることで8方向を表現する
# 左から順に[右, 上, 左, 下, 右上, 左上, 右下, 左下]となる
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]
# オセロの盤面のマスの数, 基本は8x8
tablesize = 8

def inside(y, x):
    return 0 <= y < tablesize and 0 <= x < tablesize

class DiscColor(IntEnum):
    """
    石の色用
    """
    BLACK = -1  # 黒石
    WHITE = 1   # 白石
    EMPTY = 0   # 空白
    LEGAL = 2   # 合法手(着手できるマス)
class Disc():
    """
    石の情報格納用
    """
    def __init__(self):
        # 石の色[DiscColor]
        self.color = None
        # 石の座標。盤の左上原点で単位は盤の1辺[(float, float)]Y座標、X座標の順
        self.position = None
        # 対応する盤のマスの位置[(int, int)]Y座標、X座標の順
        self.cell = None
class Board():
    
    def __init__(self):
        self.grid = [[DiscColor.EMPTY for _ in range(tablesize)] for _ in range(tablesize)]
        self.grid[3][3] = DiscColor.WHITE
        self.grid[3][4] = DiscColor.BLACK
        self.grid[4][3] = DiscColor.BLACK
        self.grid[4][4] = DiscColor.WHITE
        self.player = DiscColor.BLACK
        self.n_blackstones = 2
        self.n_whitestones = 2
    
    def read_data(self, BoardData):
        self.grid = BoardData
    
    def check_legal(self):
        """
        オセロにおける合法手があるか確認・表示する
        
        戻り値: 合法手があればTrue なければFalse (bool型)
        """
        
        # 盤面の合法手表示をなくす
        for ny in range(tablesize):
            for nx in range(tablesize):
                if self.grid[ny][nx] == DiscColor.LEGAL:
                    self.grid[ny][nx] = DiscColor.EMPTY
        
        # 返す値
        have_legal = False
        
        # 各マスについて合法かどうかチェック
        for y in range(tablesize):
            for x in range(tablesize):
                # すでに石が置いてあれば必ず非合法
                if self.grid[y][x] != DiscColor.EMPTY:
                    continue
                
                #8方向それぞれが合法かどうか確認する
                legal_flag = False
                for dr in range(8):
                    dr_legal_flag1 = False
                    dr_legal_flag2 = False
                    ny = y
                    nx = x
                    for _ in range(tablesize - 1):
                        ny += dy[dr]
                        nx += dx[dr]
                        if not inside(ny, nx): #着手する座標が盤面内か判定
                            dr_legal_flag1 = False
                            break
                        elif self.grid[ny][nx] == DiscColor.EMPTY or self.grid[ny][nx] == DiscColor.LEGAL:
                            dr_legal_flag1 = False
                            break
                        elif self.grid[ny][nx] != self.player:
                            dr_legal_flag1 = True
                        elif self.grid[ny][nx] == self.player:
                            dr_legal_flag2 = True
                            break
                    if dr_legal_flag1 and dr_legal_flag2:
                        legal_flag = True
                        break
                # 合法だったらgridの値を更新
                if legal_flag:
                    self.grid[y][x] = DiscColor.LEGAL
                    have_legal = True
        # 合法手が1つ以上あるかを返す
        return have_legal
                        
    def move(self, y, x):
        """
        オセロでの着手を行う
        y: aa
        x: aa
        戻り値: 着手した場合はTrue できなかった場合はFalse (bool型)
        """
        # 置けるかの判定
        if not inside(y, x):
            print('盤面外です')
            return False
        if self.grid[y][x] != DiscColor.LEGAL:
            print('非合法手です')
            return False
        
        # ひっくり返した枚数(着手したぶんはカウントしない)
        n_flipped = 0
        
        # 8方向それぞれ合法か見ていき、合法ならひっくり返す
        for dr in range(8):
            dr_legal_flag = False
            dr_n_flipped = 0
            ny = y
            nx = x
            for d in range(tablesize - 1):
                ny += dy[dr]
                nx += dx[dr]
                if not inside(ny, nx):
                    dr_legal_flag = False
                    break
                elif self.grid[ny][nx] == DiscColor.EMPTY or self.grid[ny][nx] == DiscColor.LEGAL:
                    dr_legal_flag = False
                    break
                elif self.grid[ny][nx] != self.player:
                    dr_legal_flag = True
                elif self.grid[ny][nx] == self.player:
                    dr_n_flipped = d
                    break
            if dr_legal_flag:
                n_flipped += dr_n_flipped
                for d in range(dr_n_flipped):
                    ny = y + dy[dr] * (d + 1)                
                    nx = x + dx[dr] * (d + 1)
                    self.grid[ny][nx] = self.player
                    print(ny)
                    print(nx)
            
        
        # 着手部分の更新
        self.grid[y][x] = self.player
        
        print("n_flipped = ", n_flipped)
        print("self.player = ", self.player)
        print("AI  = ", self.player * -1)
        
        # 石数の更新
        if self.player == DiscColor.BLACK:
            self.n_blackstones += n_flipped + 1
            self.n_whitestones -= n_flipped
        else:
            self.n_blackstones -= n_flipped
            self.n_whitestones += n_flipped + 1
        
        # 手番の更新
        self.player *= -1
        print(self.player)
        
        # ひっくり返したのでTrueを返す
        return True
    
    
    # 標準入力からの入力で着手を行う
    def move_stdin(self):
        coord = input(('黒' if self.player == DiscColor.BLACK else '白') + ' 着手: ')
        try:
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('A')
            if not inside(y, x):
                x = ord(coord[0]) - ord('a')
                if not inside(y, x):
                    print('座標を A1 や c5 のように入力してください')
                    self.move_stdin()
                    return
            if not self.move(y, x):
                self.move_stdin()
        except:
            print("例外処理")
            print('座標を A1 や c5 のように入力してください')
            self.move_stdin()
    
    # 盤面などの情報を表示
    def print_info(self):
        
        #盤面表示 X: 黒 O: 白 *: 合法手 .: 非合法手
        print('  A B C D E F G H')
        for y in range(tablesize):
            print(y + 1, end=' ')
            for x in range(tablesize):
                if self.grid[y][x] == DiscColor.BLACK:
                    print('B', end=' ')
                elif self.grid[y][x] == DiscColor.WHITE:
                    print('W', end=' ')
                elif self.grid[y][x] == DiscColor.LEGAL:
                    print('*', end=' ')
                else:
                    print('.', end=' ')
            print('')
        
        # 石数表示
        print('黒 B ', self.n_blackstones, '-', self.n_whitestones, ' W 白')