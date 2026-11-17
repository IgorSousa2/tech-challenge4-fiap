import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

import pose_analysis
import report
from openpose_estimator import OpenPoseEstimator, draw_skeleton, MODELS_DIR
import alerts as alert_module  # alerts.py na pasta raiz (o caminho é adicionado por report)

# YOLOv8 (baixado automaticamente na primeira execução, para a pasta models/): localiza a pessoa
# e os objetos. A pose do corpo é estimada pelo OpenPose (ver openpose_estimator.py).
OBJECT_MODEL = os.path.join(MODELS_DIR, 'yolov8n.pt')

PERSON_CLASS = 0

# Classes COCO de interesse para objetos cortantes (áreas/instrumentos críticos)
CRITICAL_CLASSES = {43: 'faca', 76: 'tesoura'}

# Área segura em coordenadas normalizadas (x1, y1, x2, y2); None desativa a regra
SAFE_ZONE = (0.05, 0.05, 0.95, 0.95)

# Analisa 1 a cada N frames para reduzir o custo de processamento
FRAME_STRIDE = 3

PATIENT_ID = 'P001'


def pick_main_person(object_result, frame, estimator):
    """
    Seleciona a pessoa de maior caixa no frame (considerada o paciente) e estima a pose
    dela com o OpenPose, aplicado apenas ao recorte da pessoa.
    :return: Tupla (xy, conf, box) em coordenadas do frame, ou None se ninguém foi detectado
    """
    # Só as caixas da classe "pessoa" interessam; a de maior área é assumida como o paciente
    candidates = [b.xyxy[0].cpu().numpy() for b in object_result.boxes if int(b.cls[0]) == PERSON_CLASS]
    if not candidates:
        return None
    box = max(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    # Limita a caixa às bordas do frame (o YOLO pode devolver coordenadas levemente fora da imagem)
    frame_h, frame_w = frame.shape[:2]
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[2]), frame_w), min(int(box[3]), frame_h)
    if x2 - x1 < 20 or y2 - y1 < 20:  # recorte pequeno demais para estimar pose
        return None

    # O OpenPose enxerga só o recorte (uma pessoa), pois o modelo é de pessoa única
    xy, conf = estimator.estimate(frame[y1:y2, x1:x2])
    xy = xy + np.array([x1, y1])  # volta para as coordenadas do frame
    return xy, conf, box


def detect_critical_objects(object_result):
    """
    Retorna a lista de objetos cortantes detectados: (nome, caixa).
    """
    found = []
    for box in object_result.boxes:
        class_id = int(box.cls[0])
        if class_id in CRITICAL_CLASSES:
            found.append((CRITICAL_CLASSES[class_id], box.xyxy[0].cpu().numpy()))
    return found


def analyze_video(video_path, output_video_path):
    """
    Processa o vídeo com YOLOv8 (pessoa e objetos) e OpenPose (pose), extrai características
    posturais frame a frame e grava o vídeo anotado.
    :return: Tupla (DataFrame de características, fps de análise)
    """
    estimator = OpenPoseEstimator()  # carrega o OpenPose (pose do corpo)
    object_model = YOLO(OBJECT_MODEL)  # carrega o YOLOv8 (pessoa e objetos)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analysis_fps = fps / FRAME_STRIDE  # taxa efetiva após pular frames (usada nas regras de duração)

    # Vídeo de saída com o esqueleto desenhado, na taxa efetiva de análise
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, analysis_fps, (width, height))

    # Converte a área segura de coordenadas normalizadas (0 a 1) para pixels
    zone = None
    if SAFE_ZONE:
        zone = (SAFE_ZONE[0] * width, SAFE_ZONE[1] * height, SAFE_ZONE[2] * width, SAFE_ZONE[3] * height)

    rows = []  # uma linha de características por frame analisado
    for frame_index in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % FRAME_STRIDE != 0:  # analisa só 1 a cada FRAME_STRIDE frames
            continue

        # YOLOv8 detecta todas as pessoas e objetos do frame
        object_result = object_model(frame, verbose=False)[0]

        row = {"time_s": frame_index / fps, "detected": False, "trunk_angle": np.nan,
               "knee_left": np.nan, "knee_right": np.nan, "aspect_ratio": np.nan,
               "cx": np.nan, "cy": np.nan, "outside_safe_zone": False, "critical_object": False}

        person = pick_main_person(object_result, frame, estimator)
        annotated = frame.copy()  # cópia onde são desenhados esqueleto, caixas e textos
        if person is not None:
            xy, conf, box = person
            # Ângulos, proporção da caixa e centro do paciente neste frame
            row.update(pose_analysis.extract_pose_features(xy, conf, box))
            row["detected"] = True
            draw_skeleton(annotated, xy, conf)
            cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
            # Regra da área segura: o centro do paciente caiu fora do retângulo permitido?
            if zone:
                row["outside_safe_zone"] = not (zone[0] <= row["cx"] <= zone[2] and zone[1] <= row["cy"] <= zone[3])
            if not np.isnan(row["trunk_angle"]):
                cv2.putText(annotated, "Tronco: {:.0f} graus".format(row["trunk_angle"]), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

        # Objetos cortantes (instrumentos críticos) em vermelho
        for name, (x1, y1, x2, y2) in detect_critical_objects(object_result):
            row["critical_object"] = True
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated, name, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if zone:
            cv2.rectangle(annotated, (int(zone[0]), int(zone[1])), (int(zone[2]), int(zone[3])), (255, 200, 0), 1)

        out.write(annotated)
        rows.append(row)

    cap.release()
    out.release()
    return pd.DataFrame(rows), analysis_fps


def export_movement(df, start_time, csv_path):
    """
    Exporta a trajetória do paciente (centro da caixa a cada frame) em CSV:
    timestamp, patient_id, x, y.
    """
    movement = df[df["detected"]][["time_s", "cx", "cy"]].copy()  # só os frames com paciente
    # Converte o tempo do vídeo (s) em horário absoluto somando ao início da análise
    movement["timestamp"] = start_time + pd.to_timedelta(movement["time_s"], unit="s")
    movement["patient_id"] = PATIENT_ID
    movement = movement.rename(columns={"cx": "x", "cy": "y"})
    movement[["x", "y"]] = movement[["x", "y"]].round(1)
    movement[["timestamp", "patient_id", "x", "y"]].to_csv(csv_path, index=False)


def run_analysis(video_path, output_dir, start_time):
    """
    Executa a análise completa do vídeo e grava os arquivos de saída em output_dir.
    :return: Tupla (lista de alertas, resumo) ou (None, None) se o vídeo não pôde ser processado
    """
    os.makedirs(output_dir, exist_ok=True)
    df, analysis_fps = analyze_video(video_path, os.path.join(output_dir, 'output_video_pose.mp4'))
    if df is None or df.empty:
        return None, None

    # 1) Salva as características brutas e a trajetória; 2) aplica as regras de desvio;
    # 3) gera o relatório; 4) converte os desvios em alertas no formato padrão
    df.to_csv(os.path.join(output_dir, 'pose_features.csv'), index=False)
    export_movement(df, start_time, os.path.join(output_dir, 'movement.csv'))

    events = pose_analysis.detect_deviations(df, analysis_fps)
    summary = report.build_summary(df, events, video_path, analysis_fps)
    report.save_report(summary, events, os.path.join(output_dir, 'video_report.json'),
                       os.path.join(output_dir, 'video_report.md'))

    video_alerts = report.build_alerts(events, PATIENT_ID, start_time)
    alert_module.save_alerts(video_alerts, os.path.join(output_dir, 'video_alerts.json'))
    return video_alerts, summary


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'video.mp4')
    output_dir = os.path.join(script_dir, 'output')

    video_alerts, summary = run_analysis(video_path, output_dir, datetime.now())
    if video_alerts is None:
        return

    for alert in video_alerts:
        alert_module.notify_medical_team(alert)
    print("\n{} desvios detectados. Relatório em '{}'.".format(summary["deviations_total"], output_dir))


if __name__ == "__main__":
    main()
