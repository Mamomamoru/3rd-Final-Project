"""happinessとanger用"""

import os
import glob
import numpy as np
import dlib
import cv2

# 顔検出に利用するファイル
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)  # 特徴抽出器


class NoFaces(Exception):
    pass


class Face:
    """
    numpy配列で画像を取得し、各行は入力画像内の特定の特徴点のx、y座標に対応する68x2の要素行列を返す
    """
    def __init__(self, image, rect):
            self.image = image
            self.landmarks = np.matrix(
                    [[p.x, p.y] for p in PREDICTOR(image, rect).parts()]
            )


class Face_Swapping:
    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11

    # 特徴点のうちそれぞれの部位を表している配列のインデックス
    FACE_POINTS = list(range(17, 68))       # 顔
    MOUTH_POINTS = list(range(48, 61))      # 口
    RIGHT_BROW_POINTS = list(range(17, 22)) # 右眉
    LEFT_BROW_POINTS = list(range(22, 27))  # 左眉
    RIGHT_EYE_POINTS = list(range(36, 42))  # 右目
    LEFT_EYE_POINTS = list(range(42, 48))   # 左目
    NOSE_POINTS = list(range(27, 35))       # 鼻
    JAW_POINTS = list(range(0, 17))         # あご

    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS +
                                    NOSE_POINTS + MOUTH_POINTS)

    # オーバーレイする特徴点
    OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
                                        NOSE_POINTS + MOUTH_POINTS]

    # 合成の割合(いい感じに調節する)
    COLOR_CORRECT_BLUR_FRAC = 0.7

    def __init__(self, emotion, before_after=True):
        self.detector = dlib.get_frontal_face_detector()
        self.emotion = emotion
        self._load_images()
        self.before_after = before_after

    def load_faces_from_image(self, image_path):
        """
        画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
        ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (image.shape[1] * self.SCALE_FACTOR, image.shape[0] * self.SCALE_FACTOR))

        # 画像パスから顔領域を検出して特徴点を抽出
        rects = self.detector(image, 1)

        # if len(rects) == 0:
        #     raise NoFaces
        # else:
        #     print("Number of faces detected: {}".format(len(rects)))

        faces = [Face(image, rect) for rect in rects]
        return image, faces

    def load_faces_from_cv_image(self, cv_img):
        """
        画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
        ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
        """
        image = cv_img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (image.shape[1] * self.SCALE_FACTOR, image.shape[0] * self.SCALE_FACTOR))

        # 画像パスから顔領域を検出して特徴点を抽出
        rects = self.detector(image, 1)

        # if len(rects) == 0:
        #     raise NoFaces
        # else:
        #     print("Number of faces detected: {}".format(len(rects)))

        faces = [Face(image, rect) for rect in rects]
        return image, faces

    def transformation_from_points(self, t_points, o_points):
        """
        画像を適当に合成させるために、特徴点から回転やスケールを調整する
        t_points: (target points) 対象の特徴点(入力画像)
        o_points: (origin points) 合成元の特徴点
        """
        t_points = t_points.astype(float)
        o_points = o_points.astype(float)

        # 特徴量の重心
        t_mean = np.mean(t_points, axis=0)
        o_mean = np.mean(o_points, axis=0)

        # 各特徴量との差分
        t_points -= t_mean
        o_points -= o_mean

        # 標準偏差
        t_std = np.std(t_points)
        o_std = np.std(o_points)

        # 標準化
        t_points /= t_std
        o_points /= o_std

        # 特異値分解を利用して、回転部分を計算
        # 与えられた行列に最も近い直行行列を求める
        U, S, Vt = np.linalg.svd(t_points.T * o_points)
        R = (U * Vt).T

        # 同次座標系を用いたアフィン変換(線形変換 + 平行移動)行列として完全な変換を返す
        # (画像処理でやったやつ)
        return np.vstack(
                [np.hstack(((o_std / t_std) * R, o_mean.T - (o_std / t_std) * R * t_mean.T)),
                 np.matrix([0., 0., 1.])]
        )

    def get_face_mask(self, face):
            image = np.zeros(face.image.shape[:2], dtype=float)
            for group in self.OVERLAY_POINTS:
                self._draw_convex_hull(image, face.landmarks[group], color=1)

            image = np.array([image, image, image]).transpose((1, 2, 0))
            image = (cv2.GaussianBlur(
                    image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
            image = cv2.GaussianBlur(
                    image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

            return image

    def warp_image(self, image, M, dshape):
        """
        合成元の画像を対象の画像に写像する
        """
        output_image = np.zeros(dshape, dtype=image.dtype)
        # cv2.warpAffine(イメージソース, 回転・移動を指定する行列, 画像の大きさ, flags)
        cv2.warpAffine(
            image,
            M[:2],
            (dshape[1], dshape[0]),
            dst=output_image, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP
        )
        return output_image

    def correct_colors(self, t_image, o_image, t_landmarks):
        """
        t_imageの色と一致するようにo_imageの色合いを変更
        オーバーレイされた領域のエッジ周りで不連続を引き起こしているのを修正
        """
        blur_amount = self.COLOR_CORRECT_BLUR_FRAC * np.linalg.norm(
                np.mean(t_landmarks[self.LEFT_EYE_POINTS], axis=0) -
                np.mean(t_landmarks[self.RIGHT_EYE_POINTS], axis=0)
        )
        blur_amount = int(blur_amount)

        # Gaussianフィルタではカーネル内の縦幅と横幅を奇数にする必要があるらしい
        if blur_amount % 2 == 0:
            blur_amount += 1

        # Gaussianの標準偏差はカーネルのサイズから自動で計算される(引数を0にすると)
        t_blur = cv2.GaussianBlur(t_image, (blur_amount, blur_amount), 0)
        o_blur = cv2.GaussianBlur(o_image, (blur_amount, blur_amount), 0)

        # ゼロ除算を避ける　
        o_blur += (128 * (o_blur <= 1.0)).astype(o_blur.dtype)

        return (o_image.astype(float) * t_blur.astype(float) / o_blur.astype(float))

    def to_swap(self, cv_img):
        """この関数を本体コードから呼びます"""

        #print("load original images")
        original, faces = self.load_faces_from_cv_image(cv_img)

        # base_imageに合成していく
        base_image = original.copy()

        for face in faces:
            similar_face = self._get_image_similar_to(face)
            similar_face_mask = self.get_face_mask(similar_face)

            # アフィン変換(同次座標系)を表す行列
            M = self.transformation_from_points(
                    face.landmarks[self.ALIGN_POINTS],
                    similar_face.landmarks[self.ALIGN_POINTS]
            )

            warped_image_mask = self.warp_image(similar_face_mask, M, base_image.shape)
            combined_mask = np.max(
                    [self.get_face_mask(face), warped_image_mask], axis=0
            )

            warped_image = self.warp_image(similar_face.image, M, base_image.shape)
            warped_corrected_image = self.correct_colors(base_image, warped_image, face.landmarks)
            base_image = base_image * (1.0 - combined_mask) + warped_corrected_image * combined_mask

        # path, ext = os.path.splitext(os.path.basename(image_path)) # 拡張子を取得
        #cv2.imwrite('output.jpg', base_image)
        #cv2.imshow("comp", base_image)
        # if self.before_after is True:
        #     before_after = np.concatenate((original, base_image), axis=1)
        #     cv2.imwrite('before_after_' + path + ext, before_after)
        ret = base_image.astype(np.uint8)
        ret = cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)
        return ret

    def _draw_convex_hull(self, image, points, color):
        "指定したイメージの領域を塗りつぶす"

        points = cv2.convexHull(points)
        cv2.fillConvexPoly(image, points, color=color)

    def _load_images(self):
        "画像をロードして、顔(特徴点など)を検出しておく"
        # self.emotionはFaceAPIで得られた感情
        # happiness または anger, disgust, contempt のみに対応

        self.images = []

        # if self.emotion == 'neutral' or 'fear' or 'sadness' or 'surprise':
        #   print(self.emotion)
        #   print('↑cannot swap')
        #   exit()

        # happiness系
        if self.emotion == 'happiness':
            for image_path in glob.glob(os.path.join('happiness_images', '*.jpg')):
                image, face = self.load_faces_from_image(image_path)
                self.images.append(face[0])
            print('happiness画像をロードしました.')

        # anger系
        elif self.emotion == 'anger':
            for image_path in glob.glob(os.path.join('anger_images', '*.jpg')):
                image, face = self.load_faces_from_image(image_path)
                self.images.append(face[0])
            print('anger画像をロードしました.')

    def _get_image_similar_to(self, face):
        "特徴点の差分距離が小さい画像を返す"

        # np.vectorize : 関数の引数にリストを挿入できるようにする
        get_distances = np.vectorize(
                lambda im: np.linalg.norm(face.landmarks - im.landmarks)
                )

        distances = get_distances(self.images)
        return self.images[distances.argmin()]
