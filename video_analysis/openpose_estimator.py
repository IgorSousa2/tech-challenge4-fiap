import os

import cv2
import numpy as np

# Modelo OpenPose (COCO, 18 keypoints) executado com o módulo DNN do OpenCV
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PROTO_PATH = os.path.join(MODELS_DIR, 'pose_deploy_linevec.prototxt')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'pose_iter_440000.caffemodel')

# Ordem dos keypoints do OpenPose (COCO)
KEYPOINT_NAMES = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                  "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]

# Pares de keypoints ligados por um segmento (esqueleto)
POSE_PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
              (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


class OpenPoseEstimator:
    """
    Estimador de pose OpenPose para uma única pessoa por imagem.
    Para vários indivíduos no quadro, recorte antes a região da pessoa (ver video_analysis.py).
    """

    def __init__(self, proto_path=PROTO_PATH, weights_path=WEIGHTS_PATH, input_height=368, threshold=0.1):
        if not (os.path.exists(proto_path) and os.path.exists(weights_path)):
            raise FileNotFoundError(
                "Modelos do OpenPose não encontrados em '{}'. "
                "Execute: python download_openpose_models.py".format(os.path.dirname(proto_path)))
        # O OpenPose original é em Caffe; o módulo DNN do OpenCV 4.x carrega o modelo sem compilar nada
        # (o OpenCV 5 removeu o leitor de Caffe, por isso o requirements fixa opencv-python<5)
        self.net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        self.input_height = input_height  # altura da imagem enviada à rede (a largura mantém a proporção)
        self.threshold = threshold  # confiança mínima do pico do mapa de calor

    def estimate(self, image):
        """
        Estima os keypoints do corpo em uma imagem.
        :param image: Imagem BGR (numpy array)
        :return: Tupla (xy, conf): arrays (18, 2) e (18,). Keypoints abaixo do limiar têm conf = 0.
        """
        height, width = image.shape[:2]
        in_height = self.input_height
        in_width = max(32, int(in_height * width / height))  # mantém a proporção da imagem

        # Prepara a imagem para a rede: escala os pixels para 0-1 e redimensiona
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (in_width, in_height), (0, 0, 0),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()  # mapas de calor: um por keypoint (+ mapas de afinidade, ignorados)

        xy = np.zeros((len(KEYPOINT_NAMES), 2))
        conf = np.zeros(len(KEYPOINT_NAMES))
        for i in range(len(KEYPOINT_NAMES)):
            heatmap = output[0, i, :, :]
            # O ponto mais quente do mapa é a posição mais provável da articulação
            _, peak, _, point = cv2.minMaxLoc(heatmap)
            if peak > self.threshold:
                # Converte da escala do mapa de calor para os pixels da imagem original
                xy[i] = (width * point[0] / heatmap.shape[1], height * point[1] / heatmap.shape[0])
                conf[i] = peak
        return xy, conf


def draw_skeleton(image, xy, conf, min_conf=0.1):
    """
    Desenha o esqueleto (segmentos e articulações) sobre a imagem.
    """
    for a, b in POSE_PAIRS:
        if conf[a] > min_conf and conf[b] > min_conf:
            cv2.line(image, (int(xy[a][0]), int(xy[a][1])), (int(xy[b][0]), int(xy[b][1])), (0, 255, 255), 2)
    for i in range(len(KEYPOINT_NAMES)):
        if conf[i] > min_conf:
            cv2.circle(image, (int(xy[i][0]), int(xy[i][1])), 4, (0, 0, 255), -1)
    return image
