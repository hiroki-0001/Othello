from OthelloBoardSystem import *
import OthelloRecognizerSystem 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1 or sys.argv[1] == 'game':
    board = Board()

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

【オセロで遊ぶ場合】
    python main.py 
    または
    python main.py game

【オセロロボットで遊ぶ場合】
    python main.py robot
    
    '''
    print(message)