import OthelloBoardSystem
import OthelloRecognizerSystem 
import game
import numpy as np
import cv2
import matplotlib.pyplot as plt

hint = OthelloRecognizerSystem.Hint()
image = cv2.imread("image/sample1.JPG")

recognizer = OthelloRecognizerSystem.Recognizer()
ret, result = recognizer.analyzeBoard(image, hint)

if ret:
    # 結果を配列に格納する。-2:不明、0:空き、-1:黒、1:白
    BoardData = np.ones((8, 8), dtype=np.int8) * 0
    BoardData[result.isUnknown == True] = -2
    for d in result.disc:
        BoardData[d.cell[0], d.cell[1]] = int(d.color)
    
    print(BoardData)
    game = game.Game()
    game.read_data(BoardData)
    game.gameplay()
    
        
    
    
    
else:
    print("正常に認識できませんでした")