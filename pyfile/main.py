from OthelloBoardSystem import *
from OthelloCPU import *
import OthelloRecognizerSystem 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import pyrealsense2 as rs
import time

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
    # 認識関連の設定
    hint = OthelloRecognizerSystem.Hint()
    recognizer = OthelloRecognizerSystem.Recognizer()

    # ストリーム(Color/Depth)の設定
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # ストリーミング開始
    pipeline.start(config)
    time.sleep(1) #画像全体が明るくなるまで待つ
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # フレームをnumpy arrayに変換
        color_image = np.asanyarray(color_frame.get_data())
        # フレームを表示
        cv2.imwrite("image.jpg", color_image)
        image = cv2.imread("image.jpg")
        
        # image = cv2.imread("image/sample4.JPG")
        
        ret, result = recognizer.analyzeBoard(image, hint)

        if ret:
            # 結果を配列に格納する。-2:不明、0:空き、-1:黒、1:白
            BoardData = np.ones((8, 8), dtype=np.int8) * 0
            BoardData[result.isUnknown == True] = DiscColor.UNKNOWN
            for d in result.disc:
                BoardData[d.cell[0], d.cell[1]] = int(d.color)
            
            board.read_image_data(BoardData)
            
            # 合法手生成とパス判定
            if not board.check_legal():
                board.player *= -1
            
                # 終局
                if not board.check_legal():
                    break
                
            board.print_info()
            if board.player == ai_player:
                print("CPUのターンです")
                ai.getdata(board.grid)
                y, x = ai.move()
                board.move(y, x)
                pass # robotの操作を実装する
            else:
                print("playerのターンです")
                board.move_stdin()
                
            
            board.print_info()
            
        else:
            print("オセロ盤を正常に認識できませんでした")

    # ストリーミング停止
    pipeline.stop()
    
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