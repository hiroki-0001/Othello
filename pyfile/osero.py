import numpy as np

class Board():
    
    def __init__(self):
        self.white = 1
        self.black = -1
        self.blank = 0
        self.tablesize = 8
        self.cell = np.zeros((self.tablesize, self.tablesize))
        self.cell = self.cell.astype(int)
        # 石の初期位置設定
        self.cell[3][3] = self.cell[4][4] = self.white
        self.cell[3][4] = self.cell[4][3] = self.black
        self.current = self.black
        self.pass_count = 0
        self.turn = 1
        
    def turnchange(self):
        self.current *= -1
        
    def stonenumber(self):
        return self.stones
    
    # 盤内にあるかどうか判定
    def rangecheck(self, x, y):
        if x == None:
            x = -1
        if y == None:
            y = -1
        if x < 0 or self.tablesize <= x or y < 0 or self.tablesize <= y:
            return False
        return True
    
    #(dx,dy)方向に敵石があり、その先に自石があるか判定
    def can_reverse_one(self, x, y, dx, dy):
        if not self.rangecheck(x+dx, y+dy):
            return False
        length = 0
        if not self.cell[x+dx][y+dy] == -self.current: # (dx,dy)方向が敵石じゃない時False
            return False
        else:
            while self.cell[x+dx][y+dy] == -self.current:
                x += dx
                y += dy
                length += 1
                if self.cell[x+dx][y+dy] == self.current: # (dx, dy)方向のその先に自石があるかの判定
                    return length
                elif not self.cell[x+dx][y+dy] == -self.current:
                    continue
                else:
                    return False
            else:
                return False
    
    #着手した座標でひっくり返せる石があるか判定
    def can_reverse_stone(self, x, y):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == dy == 0: # 着手した座標先であるかの確認
                    continue
                elif not self.rangecheck(x+dx, y+dy): #調べた範囲が盤の外であるかの確認
                    continue
                elif not self.can_reverse_one(x, y, dx, dy): # (dx,dy)方向に敵石があり、その先に自石があるかどうかの確認
                    continue
                else:
                    return True
    
    # 着手できる座標かどうかの判定
    def check_can_reverse(self, x, y):
        if not self.rangecheck(x, y): # 盤内にあるかチェック
            return False
        elif not self.cell[x][y] == self.blank: # すでに石がおいてあるかチェック
            return False
        elif not self.can_reverse_stone(x, y): #着手した座標でひっくり返せる石があるかチェック
            return False
        else:
            return True
        
    def reverse_stone(self, x, y): #座標に石をおいて石をひっくり返す
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                length = self.can_reverse_one(x, y, dx, dy)
                if length == None:
                    length = 0
                if length > 0:
                    for i in range(length):
                        k = i+1
                        self.cell[x + dx*k][y + dy*k] *= -1
    
    # １回のターン内の行動
    def put_stone(self, x, y):
        if self.check_can_reverse(x, y):
            self.pass_count = 0
            self.cell[x][y] = self.current
            self.reverse_stone(x,y)
            self.turnchange()
            return True
        else:
            return False
        
    #❶盤面上に石が置ける場所があるか
    def check_put_place(self):
        for i in range(self.tablesize):
            for j in range(self.tablesize):
                if self.check_can_reverse(i,j):
                    return True
                else:continue
        return False
    
    def display(self):  # 盤面の状況を表示
        print('==='*10)   #  *(下の文を参照）
        for y in range(self.tablesize):
            for x in range(self.tablesize):
                if self.cell[x][y] == self.white:
                    print('W', end = '  ')
                elif self.cell[x][y] == self.black:
                    print('B', end = '  ')
                else:
                    print('*', end = '  ')
            print('\n', end = '')