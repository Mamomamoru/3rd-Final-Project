"""
Webカメラでシャッターを切り感情認識をし、画像を加工する
Webカメラ(OpenCV) -> FaceAPI -> dlibなど
"""

import sys
from io import BytesIO
import requests
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
from FaceSwapping_at_local import Face_Swapping
from faceswap import FaceSwap

class RequestFaceAPI():
    subscription_key = "fa8323c5ef174d38a30f27d39e56b1c3"
    face_api_url = 'https://japaneast.api.cognitive.microsoft.com/face/v1.0/detect'
    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    params = {
        'returnFaceId': 'false',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'smile,emotion',
    }

    def __init__(self, capture_img):
        self.cv_img = capture_img
        self.process()

    def process(self):
        # Face-APIの呼び出し
        faces = self.request()
        # 画像の加工
        self.retouch_image(faces)

    def retouch_image(self, faces):
        print(faces)

        if not faces:
            return

        face = faces[0] # 顔が1つ以外の場合は未定義
        attributes = face["faceAttributes"]
        emotion = attributes["emotion"]
        emotion['anger'] += emotion['disgust'] + emotion['contempt'] + 0.00001
        emotion['neutral'] += emotion['fear'] + emotion['sadness'] + 0.00001

        predicted_emotion = max(emotion, key=emotion.get)
        print(predicted_emotion)
        if predicted_emotion == 'happiness' or predicted_emotion == 'anger':
            retoucher = Face_Swapping(predicted_emotion)
            new_face = retoucher.to_swap(self.cv_img)

        elif predicted_emotion == 'surprise' or predicted_emotion == 'neutral':
            new_face = FaceSwap().process(self.cv_img, predicted_emotion)

        size = 50
        emoji = cv2.imread("emoji/" + predicted_emotion + ".png")
        emoji = cv2.resize(emoji, (size, size))
        self.cv_img[240:240+size, 570:620] = emoji

        cv2.rectangle(self.cv_img, (570, 300), (620, 420), (255, 255, 255), thickness=-1)
        cv2.rectangle(self.cv_img, (570, min(419, int(420 - 120 * attributes["smile"]))), (620, 420), (0, 255, 0), thickness=-1)
        cv2.rectangle(self.cv_img, (570, 300), (620, 420), (0, 0, 0), thickness=1)


        cv2.imshow("complete!", self.hconcat_resize_min([self.cv_img, new_face]))

    def hconcat_resize_min(self, im_list, interpolation=cv2.INTER_CUBIC):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in im_list]
        return cv2.hconcat(im_list_resize)

    def request(self):
        response = requests.post(self.face_api_url, params=self.params,
                                 headers=self.headers, data=self.encode_image())
        return response.json()

    def encode_image(self):
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_RGB2BGR)
        self.cv_img = cv2.flip(self.cv_img, 1)

        res, encjpg = cv2.imencode(".jpg", self.cv_img)
        if res is False:
            raise IOError("could not encode image!")

        return BytesIO(encjpg)


class VideoCaptureView(QGraphicsView):
    """ ビデオキャプチャ """
    repeat_interval = 30

    def __init__(self, parent=None):
        """ コンストラクタ（インスタンスが生成される時に呼び出される） """
        super(VideoCaptureView, self).__init__(parent)

        # 変数を初期化
        self.pixmap = None
        self.item = None

        # VideoCapture (カメラからの画像取り込み)を初期化
        self.capture = cv2.VideoCapture(0)

        if self.capture.isOpened() is False:
            raise IOError("failed in opening VideoCapture")

        # ウィンドウの初期化
        self.scene = QGraphicsScene()   # 描画用キャンバスを作成
        self.setScene(self.scene)
        self.setVideoImage()

        # タイマー更新 (一定間隔でsetVideoImageメソッドを呼び出す)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setVideoImage)
        self.timer.start(self.repeat_interval)

    def setVideoImage(self):
        """ ビデオの画像を取得して表示 """
        ret, self.cv_img = self.capture.read()                # ビデオキャプチャデバイスから画像を取得
        if ret is False:
            raise IOError("could not capture image!")

        self.cv_img = cv2.flip(self.cv_img, 1)                     # 左右反転
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)  # 色変換 BGR->RGB
        height, width, dim = self.cv_img.shape
        bytesPerLine = dim * width                       # 1行辺りのバイト数

        self.image = QImage(self.cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        if self.pixmap == None:                          # 初回はQPixmap, QGraphicPixmapItemインスタンスを作成
            self.pixmap = QPixmap.fromImage(self.image)
            self.item = QGraphicsPixmapItem(self.pixmap)
            self.scene.addItem(self.item)                # キャンバスに配置
        else:
            self.pixmap.convertFromImage(self.image)     # ２回目以降はQImage, QPixmapを設定するだけ
            self.item.setPixmap(self.pixmap)


class MyWindow(QMainWindow):

    def __init__(self):
        """ インスタンスが生成されたときに呼び出されるメソッド """
        super(MyWindow, self).__init__()
        self.initUI()

    def initUI(self):
        """ UIの初期化 """
        self.setWindowTitle("Emotion Detector")

        detect = QAction('&Detect', self)
        detect.triggered.connect(self.detector)

        exitAction = QAction('&Exit', self)
        exitAction.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(detect)
        fileMenu.addAction(exitAction)

        self.resize(800, 600)
        self.show()

    def detector(self):
        RequestFaceAPI(self.centralWidget().cv_img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)

    main = MyWindow()               # メインウィンドウmainを作成
    viewer = VideoCaptureView()       # VideoCaptureView ウィジエットviewを作成
    main.setCentralWidget(viewer)     # mainにviewを埋め込む
    main.show()

    app.exec_()

    viewer.capture.release()
