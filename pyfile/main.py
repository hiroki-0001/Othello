from OthelloBoardSystem import *
from OthelloCPU import *
import OthelloRecognizerSystem 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1 or sys.argv[1] == 'game':
    
    # AIの手番の選択
    try:
        ai_player = int(input('AIの手番 -1: 黒(先手) 1: 白(後手) : '))
    except:
        print('-1か1を入力してください')
        exit()
    if ai_player != -1 and ai_player != 1:
        print('-1か1を入力してください')
        exit()
    
    ai = RandomAI() # RandomAI
    board = Board()
    
    while True:
        # 合法手生成とパス判定
        if not board.check_legal():
            board.player *= -1
        
            # 終局
            if not board.check_legal():
                break
            
        board.print_info()
        if board.player == ai_player:
            ai.getdata(board.grid)
            y, x = ai.move()
            board.move(y, x)
        else:
            board.move_stdin()

elif  sys.argv[1] == 'robot':

    hint = OthelloRecognizerSystem.Hint()
    image = cv2.imread("image/sample1.JPG")
    recognizer = OthelloRecognizerSystem.Recognizer()
    ret, result = recognizer.analyzeBoard(image, hint)
    board = Board()

    if ret:
        # 結果を配列に格納する。-2:不明、0:空き、-1:黒、1:白
        BoardData = np.ones((8, 8), dtype=np.int8) * 0
        BoardData[result.isUnknown == True] = -2
        for d in result.disc:
            BoardData[d.cell[0], d.cell[1]] = int(d.color)
        
        board.read_data(BoardData)
        board.print_info()
        
    else:
        print("正常に認識できませんでした")

else:
    message = '''
使い方

【オセロゲームで遊ぶ場合】
    python main.py 
    または
    python main.py game

【オセロロボットで遊ぶ場合】
    python main.py robot
    
    '''
    print(message)