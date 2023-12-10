import cv2
import numpy as np
import math
import functools
from matplotlib import pyplot as plt
from enum import IntEnum

###
### 入出力用のclass・Enumの定義
###
class DiscColor(IntEnum):
    """
    石の色用
    """
    BLACK = 0
    WHITE = 1
    
class Hint():
    """
    認識するにあたってアプリ側から与えるヒント情報
    """
    def __init__(self):
        # 35mm換算焦点距離[float]
        self.focal = None
        # 画像上の中心点[(int,int)]
        self.center = None
        # 認識モード[Mode]
        self.mode = None
    
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

class Result():
    """
    認識結果
    """
    def __init__(self):
        # 検出した石の情報[List[Disc]]
        self.disc = []
        # 不明マス(手などの障害物が写っている等)[boolの2次元配列]
        self.isUnknown = np.array([[False] * 8 for i in range(8)])

        # 写真上での盤の頂点の座標[List[(float, float)]]
        self.vertex = []

        # カメラ位置を原点とした時の盤の4頂点の3次元座標(カメラ座標系。単位=盤の1辺)[List[(float, float, float)]]
        self.vertex3d = []

        # 画像変換後の左上を原点とした時のカメラ位置を盤上に投影した点の2次元座標(単位px)[(float, float)]
        # 実際の盤の写真だと、斜めから撮った際に石の厚みで石がきれいな円にならないので、
        # 天面や底面を判定するために使用する。Y座標、X座標の順
        self.cameraPosition_px = None

        # 盤の中心を原点とした時のカメラ位置(単位=盤の1辺)[(float, float, float)]
        # AR処理で使用することを想定(本モジュール外)
        self.cameraPosition_bd = None

        # どのRecognizerで処理を行ったかを表す
        # AutomaticRecognizerでanalyzeまたはdetectBoardした場合のみ設定される
        self.recognizerType = None

    def clearDiscInfo(self):
        self.disc = []
        self.isUnknown = np.array([[False] * 8 for i in range(8)])

###
### ユーティリティ的な関数の定義
###
def intersection(line0, line1):
    """
    2直線の交点を求める
    引数の直線は、直線上の2点が与えられているものとする
    (参照サイト http://imagingsolution.blog107.fc2.com/blog-entry-137.html)
    """
    S1 = float(np.cross(line0[1] - line0[0], line1[0] - line0[0]))
    S2 = float(np.cross(line0[1] - line0[0], line0[0] - line1[1]))
    return line1[0] + (line1[1] - line1[0]) * S1 / (S1 + S2)

def getRidgeEdge(distComponent, maxCoord, direction):
    """
    最大値〜最大値-1の範囲で、指定された方向から見て最も遠い点と近い点を見つける。
    緑領域からの距離が最大値近辺で、カメラから見て最も遠い点と近い点を見つけるための関数。
    これにより、石の天面の中心と底面の中心を求める
    """
    # 最大値
    maxValue = distComponent[maxCoord]
    # 最大値-1以上の点の座標群
    ridge = np.array(np.where(distComponent >= maxValue - 1)).T
    # 隣の石を検出しないよう、maxCoordからの距離がmaxValue以内という制約を設ける
    ridge = ridge[np.apply_along_axis(
        lambda pt: np.linalg.norm( np.array(pt) - maxCoord ) <= maxValue , 
        axis=1, 
        arr=ridge)]
    # 内積の値
    dotValue = np.apply_along_axis(
        lambda pt: np.dot(np.array(pt) - maxCoord, direction),
        axis=1,
        arr=ridge
    )
    # 内積が最大になる点の座標と最小になる点の座標を返す
    maxEdgePoint = np.array(ridge[np.argmax(dotValue)])
    minEdgePoint = np.array(ridge[np.argmin(dotValue)])
    return maxEdgePoint, minEdgePoint

def argmax(matrix):
    """
    行列の成分で最大の値の座標を求める
    """
    return np.unravel_index(matrix.argmax(), matrix.shape)

def getParallelogramDiagonal(v0, v1, vc):
    """
    3次元座標内に存在し、直線上に並んでいる3点v0,vc,v1(位置ベクトル)に対して、
    v0'=a*v0, v1'=b*v1, vcはv0'とv1'の中点となるようなv0',v1'を求める(a,bはスカラー)
    写真上の盤の対角の頂点v0,v1と、対角線の交点vcがわかっている時に、3次元空間のどこの点から
    投影されたかを求めるという意味。
    """
    # 各ベクトルの長さを算出
    n_v0 = np.linalg.norm(v0)
    n_v1 = np.linalg.norm(v1)
    n_vc = np.linalg.norm(vc)
    
    # v0〜vc間、v1〜vc間の角度のcosを算出
    cos_t0 = np.dot(v0, vc)/(n_v0 * n_vc)
    cos_t1 = np.dot(v1, vc)/(n_v1 * n_vc)
    # sinの値を計算
    sin_t0 = np.sqrt(1.0 - (cos_t0 ** 2))
    sin_t1 = np.sqrt(1.0 - (cos_t1 ** 2))
    # 幾何的な考察により、求めるv0',v1'は、定数kを使って、
    # v0' = k * sin_t1 * v0/n_v0,
    # v1' = k * sin_t0 * v1/n_v1
    # と書ける
    vc0 = sin_t1 * v0 / n_v0 + sin_t0 * v1 / n_v1
    # と置けば、v0'とv1'の中点は、k * vc0 / 2と書ける

    # これがvcに一致することからkの値を求める
    k = np.sqrt((2 * n_vc) ** 2 / (np.linalg.norm(vc0) ** 2))
    return k * sin_t1 * v0 / n_v0, k * sin_t0 * v1 / n_v1

def getParallelogramRatio(vtx, center, img_size, focal, img_center):
    """
    写真の35mm換算焦点距離がfocalだと仮定した時に、写真上のvtxの4点が平行四辺形となるような
    3次元空間上の4点vtx3d(カメラpx座標系)と、短辺と長辺の比rと、その間の角度radを求める
    vtx: 写真上の4点(2次元座標)
    center: vtxの4点の対角線の交点
    img_size: 写真のサイズ(px)
    focal: 35mm換算焦点距離
    img_center: 画像の中心点(カメラの正面の点)
    """
    height, width, _ = img_size

    # 与えられた写真の場合に、焦点距離に対応するカメラからの距離(px)を求める
    z0 = float(focal) * np.sqrt(width ** 2 + height ** 2) / np.sqrt(24 ** 2 + 36 ** 2)
    # 画像の中心点をx0,y0とする。
    if img_center is None:
        x0 = float(width) / 2.0
        y0 = float(height) / 2.0
    else:
        y0, x0 = img_center

    # カメラの3次元上の座標を原点、写真平面がz = z0とした場合、
    # 写真の座標系での点(x,y)は、この3次元空間上は(x-x0, y-y0, z0)となる
    # この座標系は、カメラの向きをz軸の正の方向とし、写真とx,yの方向は合っているものとする

    # 交点の位置ベクトルを求める
    vc = np.array([center[0] - x0, center[1] - y0, z0], dtype=np.float32)

    # 1組目の対角線の各頂点への位置ベクトル
    v0 = np.array([vtx[0][0] - x0, vtx[0][1] - y0, z0], dtype=np.float32)
    v2 = np.array([vtx[2][0] - x0, vtx[2][1] - y0, z0], dtype=np.float32)
    # 対角線の頂点に変換
    v0_d, v2_d = getParallelogramDiagonal(v0, v2, vc)

    # 2組目の対角線の各頂点への位置ベクトルについても同様の計算を行う
    v1 = np.array([vtx[1][0] - x0, vtx[1][1] - y0, z0], dtype=np.float32)
    v3 = np.array([vtx[3][0] - x0, vtx[3][1] - y0, z0], dtype=np.float32)
    v1_d, v3_d = getParallelogramDiagonal(v1, v3, vc)

    # 縦横の長さの比を求める
    ratio = np.linalg.norm(v1_d - v0_d) / np.linalg.norm(v3_d - v0_d)
    # 角度を求める
    rad = math.acos(np.dot(v1_d - v0_d, v3_d - v0_d) / (np.linalg.norm(v1_d - v0_d) * np.linalg.norm(v3_d - v0_d)))
    return ratio, rad, np.array([v0_d, v1_d, v2_d, v3_d])

###
### 認識用のクラスの定義
###

class Recognizer():
    
    # 認識時に盤の画像を切り出す際のサイズの定義(単位:px)
    # 1マスの辺の長さ(公式盤の1mm=1pxくらい)
    _CELL_SIZE = 42
    # 周囲の余白(余白をつけておいた方が縁付近の特殊な考慮が不要になるので)
    _BOARD_MARGIN = 13
    # 切り出し後の画像のサイズ
    _EXTRACT_IMG_SIZE = _CELL_SIZE * 8 + _BOARD_MARGIN * 2
    # 盤面抽出時の近似の係数
    _EPSILON_COEFF = 0.004
    
    # 石の色を決定する際に色を収集する半径
    _RADIUS_FOR_DISC_COLOR = 10
    
    # マスの石以外の部分の色が特殊(白黒緑以外)な場合に無視するためのフィルタ。各マスの中央以外の部分のマスクを行う
    _COLORED_MASK = np.ones((_EXTRACT_IMG_SIZE, _EXTRACT_IMG_SIZE) \
        , dtype=np.uint8) * 255
    
    # オセロクエスト対策用に使用するフィルタ。各マスの中央のマスクを行う
    _OQ_MASK = np.zeros((_EXTRACT_IMG_SIZE, _EXTRACT_IMG_SIZE), dtype=np.uint8)
    
    # 石の色を収集するための円形のフィルタ
    _CIRCLE_FILTER = np.zeros((_RADIUS_FOR_DISC_COLOR * 2 + 1, _RADIUS_FOR_DISC_COLOR * 2 + 1), dtype=np.int8)
    _CIRCLE_FILTER = cv2.circle(_CIRCLE_FILTER, (_RADIUS_FOR_DISC_COLOR, _RADIUS_FOR_DISC_COLOR), \
        _RADIUS_FOR_DISC_COLOR, (1), -1)
    
    #
    # 定数値
    #
    _KERNEL3 = np.ones((3, 3), dtype=int)
    _KERNEL5 = np.ones((5, 5), dtype=int)
    _KERNEL9 = np.ones((9, 9), dtype=int)
    
    def analyzeBoard(self, image, hint):
        """
        盤の範囲の認識(detect_board)と石の位置・色の認識(detect_disc)を実行

        image: ndarrayで元となる画像を指定
        戻り値: 認識成否(bool)と、成功した場合は結果情報(Result)
        """
        # 盤の検出処理
        ret, result = self.detectBoard(image, hint)
        if ret:
            # 成功した場合は石の検出処理
            return self.detectDisc(image, result)
        else:
            return False, None
    
    def detectBoard(self, image, hint):
        """
        盤の範囲の認識

        image: ndarrayで元となる画像を指定
        戻り値: 認識成否(bool)と、成功した場合は結果情報(Result)
        """

        # 盤の凸包候補を取得する
        ret, hull = self.detectConvexHull(image)
        if ret and hull.shape[0] >= 4:
            # 成功し、かつ4点以上ある場合は結果の設定処理
            return self.resultForDetectBoard(image.shape, hint, hull)
        else:
            return False, None

    def detectConvexHull(self, image):
        """
        盤の範囲候補の凸包を取得するための内部関数

        image: ndarrayで元となる画像を指定
        戻り値: 認識成否(bool)と、凸包情報(ndarray)
        """
        height, width, _ = image.shape[:3]
        #ぼかし処理
        image = cv2.blur(image, (3,3))
        # hsv形式に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 緑色を検出
        low_color1 = np.array([45,89,30])
        high_color1 = np.array([90,255,255])
        green1 = cv2.inRange(hsv,low_color1,high_color1)
        low_color2 = np.array([45,64,89])
        high_color2 = np.array([90,255,255])
        green2 = cv2.inRange(hsv,low_color2,high_color2)

        green = cv2.bitwise_or(green1,green2)
        
        # モルフォロジー変換
        kernelSize = max(1, int(0.0035 * max(width, height))) * 2 + 1
        kernel = np.ones((kernelSize, kernelSize), dtype=int)
        green = cv2.dilate(green, kernel) # 膨張
        green = cv2.erode(green, kernel) # 収縮
        
        # 白色を検出
        lower = np.array([0, 0, 128])
        upper = np.array([180, 50, 255])
        white = cv2.inRange(hsv, lower, upper)
        greenWhite = cv2.bitwise_or(green, white)
        
        #領域の輪郭抽出
        contours, hierarchy = cv2.findContours(greenWhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 凸包
        for i, c in enumerate(contours):
            hull = cv2.convexHull(c) # 凸包
            if cv2.pointPolygonTest(hull,(width / 2 , height / 2), False) > 0: # 中心を含むか判定
                mask = np.zeros((height, width),dtype=np.uint8)
                cv2.fillPoly(mask, pts=[hull], color=(255)) # 輪郭範囲内を塗りつぶした画像
                break
        
        green = cv2.bitwise_and(green, green, mask=mask)
        contours, hierarchy = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        greenContours = functools.reduce(lambda x, y: np.append(x, y,axis=0),contours)
        hull = cv2.convexHull(greenContours)
        
        return True, hull.astype(np.float32)
        
    
    def resultForDetectBoard(self, size, hint, hull):
        """
        盤の範囲認識処理で、凸包取得後に結果を設定するための内部関数
        盤の頂点情報やカメラ位置の情報等を設定する

        size: 画像サイズ(width, height)
        hull: _detectConvexHullで取得した凸包
        戻り値: 認識成否(bool)と、盤の認識結果(Result)
        """
        height, width, _ = size

        # 細かい凹凸の排除
        epsilon = 0.004 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        #長い線分のトップ4を4辺とする
        count = len(approx)
        distances = []
        for k in range(0, count):
            distances.append(np.linalg.norm(approx[k][0] - approx[(k + 1) % count][0]))
        # 長い順にソート
        distances.sort()
        
        # 4位以上の線分を抽出。
        lines = []
        for k in range(0, count):
            if np.linalg.norm(approx[k][0] - approx[(k + 1) % count][0]) >= distances[count - 4] :
                lines.append([approx[k][0], approx[(k + 1) % count][0]])
                
        # 最悪、同点4位があるかもしれないので、4つになるまで同点4位のものを削除する
        if len(lines) > 4:
            for k in reversed(range(0, len(lines))):
                if np.linalg.norm(approx[k][0] - approx[(k + 1) % count][0]) == distances[count - 4] :
                    lines.remove(lines[k])
                    if len(lines) <= 4:
                        break
                    
        # 4辺の交点を頂点とする
        vtx = []
        for k in range(0,4):
            vtx.append(intersection(lines[k], lines[(k + 1) % 4]))
        
        # 上の点から順番に並べる
        vtx = sorted(vtx, key=lambda pt: pt[1])
        # 左上の方から時計回りになるように、上2点、下2点でそれぞれ並べ変える
        if vtx[0][0] > vtx[1][0]:
            vtx = [vtx[1], vtx[0], vtx[2], vtx[3]]
        if vtx[2][0] < vtx[3][0]:
            vtx = [vtx[0], vtx[1], vtx[3], vtx[2]]
        
        # 面積が画像の短辺を一辺とする正方形より大きい場合は、盤がはみ出していると判断して対象外とする
        S = abs(float(np.cross(vtx[1] - vtx[0], vtx[2] - vtx[0])))
        if S >= min(width, height) ** 2:
            return False, None
        
        # 隣接する2点が画像の端近辺の場合ははみ出しているとみなす
        limit = 2.0
        for k in range(0, 4):
            x0, y0 = vtx[k]
            x1, y1 = vtx[(k + 1) % 4]
            if ( abs(x0) <= limit and abs(x1) <= limit ) \
                or ( abs(y0) <= limit and abs(y1) <= limit ) \
                or ( abs(x0 - width) <= limit and abs(x1 - width) <= limit ) \
                or ( abs(y0 - height) <= limit and abs(y1 - height) <= limit ):
                return False, None
        
        result = Result()
        ret, result = self.setCameraInfo(result, hint, vtx, size, False)
        if ret:
            # 盤の枠線を考慮して少しだけサイズを広げる
            vtx = self.adjustVertexes(vtx)
            result.vertex = vtx
            return ret, result
        return False, None
        
    def setCameraInfo(self, result, hint, vtx, img_size, force):
        """
        カメラの座標を計算し結果に設定する

        result: 設定対象のResultインスタンス
        hint: 認識時に使用するHint情報
        vtx: 写真上の4点(2次元座標)
        img_size: 写真のサイズ(px)
        force: 盤のチェックが失敗しても強制的に続行するかどうか
        """
        # 対角線の交点を求める
        center = intersection([vtx[0],vtx[2]], [vtx[1],vtx[3]])

        # 写真の4頂点に射影された元が正方形かどうかの判定
        # 画角が与えられていない場合は画角を求める
        if hint.focal is None:
            # まずはlog(focal) = 0と仮定して、4頂点の原像が平行四辺形だった場合の短辺長辺の比と、
            # その間の角度、3次元上の座標を求める
            focal_log = 0.0
            img_center = np.array([img_size[0] / 2, img_size[1] / 2]) # 画像の中心
            ratio, rad, vtx3d = getParallelogramRatio(vtx, center, img_size, math.exp(focal_log), img_center)
            # 正方形であればratio = 1.0, rad = pi/2 になるはずなので、誤差を求めておく
            error = (ratio - 1.0) ** 2 + (rad / (math.pi / 2) - 1.0) ** 2
            scale = 1.0
            # 誤差が小さくなるように、logの値を範囲を狭めながら近づけていく
            for i in range(0, 5):
                # 現在の最善の値
                f_log_loop_best = focal_log
                # 現在の最善のfocal_logが例えば1.4であれば、1.31,...,1.49まで試して一番良いものを選ぶ
                for j in range(-9, 10):
                    cur_focal = math.exp(focal_log + j * scale)
                    cur_ratio, cur_rad, cur_vtx3d = getParallelogramRatio(vtx, center, img_size, cur_focal, img_center)
                    cur_error = (cur_ratio - 1.0) ** 2 + (cur_rad / (math.pi / 2) - 1.0) ** 2
                    if cur_error < error:
                        error = cur_error
                        vtx3d = cur_vtx3d
                        f_log_loop_best = focal_log + j * scale
                focal_log = f_log_loop_best
                # 1桁細かい部分で再実行
                scale *= 0.1
        else:
            # ヒントとしてfocalが与えられている場合は、その値を元に誤差を算出
            ratio, rad, vtx3d = getParallelogramRatio(vtx, center, img_size, hint.focal, hint.center)
            error = (ratio - 1.0) ** 2 + (rad / (math.pi / 2) - 1.0) ** 2
        
        # 正方形っぽくなかったら失敗とする
        if error >= 0.002 and force == False :
            return False, None

        # 盤をwarpPerspectiveで変換した後のカメラ位置を求める
        # 座標系は盤の左上隅(vtx[0])からBORAD_MARGIN分拡張した点を原点とし、
        # 盤のカメラ側をz軸の負の方向とする

        # まずマージン補正前のx軸・y軸方向となるベクトルを3次元座標上で求める
        xVec = vtx3d[1] - vtx3d[0] # vtx3d[1]は右上の頂点
        yVec = vtx3d[3] - vtx3d[0] # vtx3d[3]は左下の頂点
        # 各ベクトルの長さを求めておく
        xLen = np.linalg.norm(xVec)
        yLen = np.linalg.norm(yVec)
        # 法線ベクトルを求める(z軸)
        zVec = np.cross(xVec, yVec)
        # zVecの長さを標準化しておく
        zVec = zVec / np.linalg.norm(zVec)

        # 原点(カメラ位置)と、盤面のpx距離を求める
        cameraZ = np.dot(vtx3d[0], zVec)
        # 原点(カメラ位置)から盤面に下ろした法線の足は、 cameraZ * zVec
        # その盤面上のx座標、y座標を求める(px単位)
        cameraX = np.dot(cameraZ * zVec - vtx3d[0], xVec) / xLen
        cameraY = np.dot(cameraZ * zVec - vtx3d[0], yVec) / yLen

        # マージン、縮尺を補正してカメラ位置を記憶しておく
        # 縮尺は一辺が_CELL_SIZE * 8になるようにする
        cameraPosAdjustedX = (cameraX / xLen) * self._CELL_SIZE * 8 + self._BOARD_MARGIN
        cameraPosAdjustedY = (cameraY / yLen) * self._CELL_SIZE * 8 + self._BOARD_MARGIN
        result.cameraPosition_px = np.array([cameraPosAdjustedY, cameraPosAdjustedX], dtype=np.float32)

        # 盤の中心を原点とし、盤の一辺を1とする座標系でカメラ位置を求める
        cameraPosNormalizedX = cameraX / xLen - 0.5
        cameraPosNormalizedY = cameraY / yLen - 0.5
        boardSize = (xLen + yLen) / 2
        cameraPosNormalizedZ = cameraZ / boardSize
        result.cameraPosition_bd = np.array([cameraPosNormalizedX, cameraPosNormalizedY, cameraPosNormalizedZ], dtype=np.float32)

        # 頂点の情報を設定
        result.vertex = vtx
        result.vertex3d = list(map(lambda v: v / boardSize, vtx3d))

        return True, result
        
    def extractBoard(self, image, vertex, size, ratio = 1.0, margin = 0, outer=(0, 0, 0), fillMargin = True):
        """
        検出した盤を正方形に変換した画像を取得

        image: ndarrayで元となる画像を指定
        vertex: 盤の範囲の認識結果(頂点情報)
        size: 変換後の画像サイズ
        ratio: 頂点位置の補正用(周りの枠部分を少し拡大する用途を想定)
        margin: 変換後の画像のマージン。sizeの内数
        outer: 外側の色
        戻り値: 変換した画像(ndarray)
        """
        height = size[1]
        width = size[0]
        # 変換元の各頂点
        src = np.array(vertex, dtype=np.float32)
        # 変換後の各頂点
        dst = np.array([
            [margin, margin],
            [width - 1 - margin, margin],
            [width - 1 - margin, height - 1 - margin],
            [margin, height - 1 - margin]
        ], dtype=np.float32)
        
        # 変換行列
        trans = cv2.getPerspectiveTransform(src, dst)
        # 変換
        board = cv2.warpPerspective(image, trans, (int(width), int(height)), \
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=outer)
        # marginを塗りつぶす
        if (margin > 0):
            cv2.rectangle(board, (0, 0), (width - 1, margin), outer, -1)
            cv2.rectangle(board, (0, 0), (margin, height - 1), outer, -1)
            cv2.rectangle(board, (width - margin, 0), (width - 1, height - 1), outer, -1)
            cv2.rectangle(board, (0, height - margin), (width - 1, height - 1), outer, -1)
            
        return board
    
    def adjustVertexes(self, vtx):
        """
        求めた頂点の調整
        基底クラスでは何もしない。
        """
        return vtx
    
    def detectDisc(self, image, result):
        """
        石の位置・色の認識を実行するための内部関数
        """
        # 結果の石情報のクリア
        result.clearDiscInfo()
        
        # 盤を切り出した画像の取得
        board = self.extractBoard(image, result.vertex, \
            [Recognizer._EXTRACT_IMG_SIZE, Recognizer._EXTRACT_IMG_SIZE], \
            ratio=1.0, margin=Recognizer._BOARD_MARGIN, outer=(96), fillMargin=False)
        
        # 盤の外部(枠外)を抽出
        outer = cv2.inRange(board, (254, 0, 0), (254, 0, 0))
        
        # 盤面をHSVに変換
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)        
        
        # 緑色抽出
        lower = np.array([45,89,30])
        upper = np.array([90,255,255])
        green1 = cv2.inRange(hsv,lower,upper)

        lower = np.array([45,64,89])
        upper = np.array([90,255,255])
        green2 = cv2.inRange(hsv,lower,upper)
        green = cv2.bitwise_or(green1,green2)
        green = cv2.bitwise_or(green, outer)
        
        # 緑色以外の領域(≒石の領域)
        notGreen = cv2.bitwise_not(green)
        
        # 色のついている(白黒以外の)領域(Sの値が小さい)
        colored = cv2.inRange(hsv, np.array([0, 127, 30]), np.array([180, 255, 255]))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 100, 50]), np.array([180, 255, 255])))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 75, 150]), np.array([180, 255, 255])))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 50, 200]), np.array([180, 255, 255])))

        # グレースケールの盤の画像
        grayBoard = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

        info = self.prepareInfoForDetectDisc(grayBoard)
        
        disc, info, result = self.processColoredAndExtractDiscForDetectDisc(notGreen, colored, outer, info, result)
        
        # 石の外側からの距離を求める
        dist = cv2.distanceTransform(disc, cv2.DIST_L2, 5)
        # 外側からの距離が13以上のブロックに分ける。これによって隣接している石が分離できる
        _, distThr = cv2.threshold(dist, 13.0, 255.0, cv2.THRESH_BINARY)
        distThr = np.uint8(distThr) # 型変換が必要
        
        # # 連結成分を取得
        labelnum, labelimg, data, center = cv2.connectedComponentsWithStats(distThr)
        for i in range(1, labelnum): # i = 0 は背景なので除外
            x, y, w, h, s = data[i]
            
            if s > 2500:
                # 大きすぎる場合は何か違うものが写っているとみなして、不明マスと扱う
                result = self._setColorUnknown(labelimg, x, y, w, h, i, result)
                continue
            
            distComponent = dist[y:y+h, x:x+w]
            
            # 普通は連結領域1つに1つの石だが、双峰っぽくなっている場合もあるのでループして消しこみながら判定する
            while True:
                # 領域内の距離の最大値とその場所を求める
                maxCoord = argmax(distComponent)
                maxVal = distComponent[maxCoord]
                if maxVal < 13.0:
                    #検出を終えている場合はループを抜ける
                    break
                # 石の色の判定
                result = self.detectDiscColor(distComponent, x, y, info, result, maxCoord)

                # 判定済みの箇所を消し込む
                cv2.circle(distComponent, (maxCoord[1], maxCoord[0]), int(maxVal * 1.4), 0, -1)

        return True, result
    
    def detectDiscColor(self, distComponent, x, y, info, result, maxCoord):
        """
        石の色を判断する
        """
        # カメラの位置から石の位置に向けた方向を求める
        direction = maxCoord + np.array([y, x]) - result.cameraPosition_px

        # maxCoord付近の最大値に近い点の内、この向きに対して遠い方が石の表面の中心、近い方が石の底面の中心
        
        maxEdgePoint, minEdgePoint = getRidgeEdge(distComponent, maxCoord, direction)
        # 色を判定する範囲
        startX = maxEdgePoint[1] + x - Recognizer._RADIUS_FOR_DISC_COLOR
        startY = maxEdgePoint[0] + y - Recognizer._RADIUS_FOR_DISC_COLOR
        endX = maxEdgePoint[1] + x + Recognizer._RADIUS_FOR_DISC_COLOR + 1
        endY = maxEdgePoint[0] + y + Recognizer._RADIUS_FOR_DISC_COLOR + 1

        # maxEdgePointを中心とする円内で、binBoardWideで黒の面積を求める
        subBinWide = info["wide"][startY:endY, startX:endX]
        circleWide = subBinWide[Recognizer._CIRCLE_FILTER == 1]
        areaWide = circleWide[circleWide == 0].shape[0]

        # maxEdgePointを中心とする円内で、binBoardNarrowで黒の面積を求める
        subBinNarrow = info["narrow"][startY:endY, startX:endX]
        circleNarrow = subBinNarrow[Recognizer._CIRCLE_FILTER == 1]
        areaNarrow = circleNarrow[circleNarrow == 0].shape[0]

        # 底面の座標
        bottomCoord = minEdgePoint + np.array([y, x]) - np.array([Recognizer._BOARD_MARGIN, Recognizer._BOARD_MARGIN])
        bottomCoord = bottomCoord / (Recognizer._CELL_SIZE * 8)
        # セルの位置
        bottomIndex = np.array([0, 0])
        bottomIndex[0] = min(7, max(0, int(bottomCoord[0] / 0.125)))
        bottomIndex[1] = min(7, max(0, int(bottomCoord[1] / 0.125)))

        # 色の判定
        if areaWide >= 10:
            # 黒
            self.setDisc(result, DiscColor.BLACK, bottomCoord, bottomIndex)
        elif areaNarrow >= 26:
            # 黒
            self.setDisc(result, DiscColor.BLACK, bottomCoord, bottomIndex)
        else:
            # 白
            self.setDisc(result, DiscColor.WHITE, bottomCoord, bottomIndex)
        
        return result
    
    def setDisc(self, result, color, coord, index):
        """
        結果に石情報を設定
        """
        if result.isUnknown[tuple(index)]:
            # 不明マスの場合は無視する
            return result
        
        # 石情報を追加
        disc = Disc()
        disc.color = color
        disc.position = coord
        disc.cell = index
        result.disc.append(disc)
        return result
    
    def prepareInfoForDetectDisc(self, grayBoard):
        """
        石の認識に必要な各種情報を準備するための内部関数(石の位置・色の認識用)
        """
        # 二値化
        # 見た目の色を判断するための広域的な二値化
        binBoardWide = cv2.adaptiveThreshold(grayBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -20)
        # 反射した石は少しまだらになるので、それが検出できるように近傍で二値化したものも作っておく
        binBoardNarrow = cv2.adaptiveThreshold(grayBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        binBoardNarrow = cv2.blur(binBoardNarrow, (3, 3))
        _, binBoardNarrow = cv2.threshold(binBoardNarrow, 168, 255, cv2.THRESH_BINARY)

        return { "wide": binBoardWide, "narrow": binBoardNarrow}
    
    def processColoredAndExtractDiscForDetectDisc(self, notGreen, colored, outer, info, result):
        """
        着色領域の処理と石の領域を取得するための内部関数(石の位置・色の認識用)
        """
        # 各マスの中央付近以外にある着色領域は緑色と同じ扱いにする。不明マス扱いにはしない。
        # (最終手のマス全体が着色されているようなアプリの対応)
        coloredOut = cv2.bitwise_and(colored, self._COLORED_MASK)
        notColoredOut = cv2.bitwise_not(coloredOut)
        disc = cv2.bitwise_and(notGreen, notColoredOut)

        # 着色領域を二値化画像(0が黒)にorで加算して白っぽくしておく
        # (最終手の石の中央に赤四角等が表示されているようなアプリで、黒石と判定しないための対処)
        colored = cv2.dilate(colored, self._KERNEL5)
        binBoardWide = cv2.bitwise_or(info["wide"], colored)
        info["wide"] = binBoardWide

        # オセロクエストの旧スタイルの黒石に緑色っぽい画素が含まれているのでその対策
        # 各マスの中央付近はノイズを除去しておく
        # 石の狭間でわずかに見えている緑色を消してしまわないよう、マスの中央付近以外はノイズ除去対象外とする
        discDenoised = cv2.dilate(disc, self._KERNEL5)
        discDenoised = cv2.erode(discDenoised, self._KERNEL5)
        discDenoised = cv2.bitwise_and(discDenoised, discDenoised, mask=self._OQ_MASK)
        disc = cv2.bitwise_or(disc, discDenoised)

        return disc, info, result
    
    def _setColorUnknown(self, labelimg, x, y, w, h, idx, result):
        """
        labelimgの中でidxの値が入っているマスをUnknown扱いとする
        チェック範囲はx〜x+w, y〜y+hとする
        """
        # 盤のマスごとにチェックする
        startI = max(0, int((x - self._BOARD_MARGIN) / self._CELL_SIZE))
        startJ = max(0, int((y - self._BOARD_MARGIN) / self._CELL_SIZE))
        endI = min(7, int((x + w - self._BOARD_MARGIN) / self._CELL_SIZE))
        endJ = min(7, int((y + h - self._BOARD_MARGIN) / self._CELL_SIZE))

        for i in range(startI, endI + 1):
            topX = i * self._CELL_SIZE + self._BOARD_MARGIN
            bottomX = (i + 1) * self._CELL_SIZE + self._BOARD_MARGIN
            for j in range(startJ, endJ + 1):
                # 既にUnknownの場合は判定しない
                if result.isUnknown[j, i]:
                    continue
                topY = j * self._CELL_SIZE + self._BOARD_MARGIN
                bottomY = (j + 1) * self._CELL_SIZE + self._BOARD_MARGIN
                # セルの範囲
                cell = labelimg[topY:bottomY, topX:bottomX]
                # その範囲内にidxの値が存在する場合はUnknownとする
                if len(cell[cell == idx]) > 0:
                    result.isUnknown[j, i] = True
        return result