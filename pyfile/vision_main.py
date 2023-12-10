import numpy as np
import cv2
import OthelloRecognizerSystem as ors
import matplotlib.pyplot as plt

hint = ors.Hint()

image = cv2.imread("image/sample1.JPG")

recognizer = ors.Recognizer()
ret, result = recognizer.analyzeBoard(image, hint)

if ret:
    # 成功した場合は結果を描画
    CELL = 40
    SIZE = CELL * 8
    # 盤の部分を切り出し
    board = recognizer.extractBoard(image, result.vertex, (SIZE, SIZE))
    
    # 結果を配列に格納する。-2:不明、-1:空き、0:黒、1:白
    bd = np.ones((8, 8), dtype=np.int8) * -1
    bd[result.isUnknown == True] = -2
    for d in result.disc:
        # 配列を更新しつつ、石の場所に円を描画
        if d.color == ors.DiscColor.BLACK:
            color = (0, 0, 0)
            line = (255, 255, 255)
        else:
            color = (255, 255, 255)
            line = (0, 0, 0)
        bd[d.cell[0], d.cell[1]] = int(d.color)
        x = int(d.position[1] * SIZE)
        y = int(d.position[0] * SIZE)
        cv2.circle(board, (x, y), 8, line, -1)
        cv2.circle(board, (x, y), 7, color, -1)
    
    # 空きマス・不明マスの描画
    for j in range(0, 8):
        for i in range(0, 8):
            x = int((i + 0.5) * CELL)
            y = int((j + 0.5) * CELL)
            if bd[j, i] == -1:
                # 空きマス
                cv2.rectangle(board, (x - 4, y - 4), (x + 4, y + 4), (0, 255, 0), -1)
            elif bd[j, i] == -2:
                # 不明マス
                cv2.rectangle(board, (x - 4, y - 4), (x + 4, y + 4), (128, 128, 128), -1)
    board = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    plt.imshow(board)
    plt.show()
    plt.imsave('vision2.png', board)
else:
    print("正常に認識できませんでした")