import numpy as np

# Índices dos keypoints no formato COCO de 18 pontos do OpenPose
NOSE, NECK = 0, 1
R_SHOULDER, L_SHOULDER = 2, 5
R_HIP, L_HIP = 8, 11
R_KNEE, L_KNEE = 9, 12
R_ANKLE, L_ANKLE = 10, 13

# Confiança mínima (pico do mapa de calor do OpenPose) para considerar um keypoint válido
MIN_KEYPOINT_CONF = 0.2

# Limiares das regras de desvio (valores ilustrativos, ajustáveis ao tipo de procedimento)
MAX_TRUNK_INCLINATION = 35.0   # graus em relação à vertical
MIN_KNEE_ANGLE = 60.0          # graus; abaixo disso = flexão excessiva do joelho
MAX_KNEE_ASYMMETRY = 25.0      # graus de diferença entre os dois joelhos
FALL_ASPECT_RATIO = 1.0        # largura/altura da caixa; maior que 1 = corpo deitado
FALL_TRUNK_ANGLE = 60.0        # graus
MIN_EVENT_SECONDS = 0.5        # duração mínima para considerar um desvio
MAX_ABSENCE_SECONDS = 5.0      # tempo máximo sem detectar o paciente


def joint_angle(a, b, c):
    """
    Calcula o ângulo (em graus) formado no ponto b pelos segmentos b-a e b-c.
    :param a, b, c: Coordenadas (x, y) dos pontos
    :return: Ângulo entre 0 e 180 graus
    """
    # Vetores que saem do vértice b em direção a a e a c
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return float("nan")
    # Produto escalar = |ba||bc|cos(ângulo); o clip evita erro numérico fora de [-1, 1]
    cos_angle = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _valid(conf, *indexes):
    return all(conf[i] >= MIN_KEYPOINT_CONF for i in indexes)


def trunk_inclination(xy, conf):
    """
    Inclinação do tronco em relação à vertical (0 = em pé, 90 = deitado),
    usando o ponto médio dos ombros e o ponto médio dos quadris.
    """
    if not _valid(conf, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return float("nan")
    shoulders = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2  # ponto médio dos ombros
    hips = (xy[L_HIP] + xy[R_HIP]) / 2  # ponto médio dos quadris
    dx, dy = shoulders[0] - hips[0], shoulders[1] - hips[1]
    # Ângulo da linha quadril-ombro com a vertical: deslocamento horizontal (dx) sobre vertical (dy)
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def knee_angle(xy, conf, hip, knee, ankle):
    """
    Ângulo do joelho (180 = perna estendida).
    """
    if not _valid(conf, hip, knee, ankle):
        return float("nan")
    return joint_angle(xy[hip], xy[knee], xy[ankle])


def extract_pose_features(xy, conf, box):
    """
    Calcula as características posturais de uma pessoa detectada.
    :param xy: Array (18, 2) com as coordenadas dos keypoints do OpenPose
    :param conf: Array (18,) com a confiança de cada keypoint
    :param box: (x1, y1, x2, y2) da caixa da pessoa
    :return: Dicionário com trunk_angle, knee_left, knee_right, aspect_ratio, cx e cy
    """
    xy, conf = np.asarray(xy, dtype=float), np.asarray(conf, dtype=float)
    x1, y1, x2, y2 = box
    # aspect_ratio > 1 = caixa mais larga que alta (pessoa deitada); cx/cy = centro, usado na área segura
    return {
        "trunk_angle": trunk_inclination(xy, conf),
        "knee_left": knee_angle(xy, conf, L_HIP, L_KNEE, L_ANKLE),
        "knee_right": knee_angle(xy, conf, R_HIP, R_KNEE, R_ANKLE),
        "aspect_ratio": float((x2 - x1) / max(y2 - y1, 1e-6)),
        "cx": float((x1 + x2) / 2),
        "cy": float((y1 + y2) / 2),
    }


def find_runs(mask, min_length):
    """
    Encontra sequências consecutivas de True com pelo menos min_length elementos.
    :return: Lista de tuplas (índice inicial, índice final), ambos inclusivos
    """
    runs, start = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_length:
                runs.append((start, i - 1))
            start = None
    return runs


def detect_deviations(df, fps):
    """
    Aplica as regras de desvio sobre as características extraídas frame a frame.
    :param df: DataFrame com time_s, detected, trunk_angle, knee_left, knee_right,
               aspect_ratio e outside_safe_zone
    :param fps: Frames processados por segundo
    :return: Lista de eventos (dicionários) com tipo, severidade, tempos e mensagem
    """
    df = df.reset_index(drop=True)
    min_frames = max(1, int(round(MIN_EVENT_SECONDS * fps)))  # duração mínima convertida em frames
    events = []

    def add(kind, severity, mask, message, value_column=None, min_len=min_frames):
        # Cada trecho contínuo em que a regra (mask) vale por tempo suficiente vira um evento
        for start, end in find_runs(mask, min_len):
            segment = df.iloc[start:end + 1]
            event = {
                "type": kind,
                "severity": severity,
                "start_s": float(segment["time_s"].iloc[0]),
                "end_s": float(segment["time_s"].iloc[-1]),
                "message": message,
            }
            if value_column:
                event["peak_value"] = float(segment[value_column].abs().max())
            events.append(event)

    detected = df["detected"]

    # Possível queda: corpo na horizontal (caixa larga) e tronco muito inclinado.
    # Com a pessoa deitada o OpenPose costuma não achar ombros/quadris (tronco = NaN);
    # nesse caso a caixa horizontal sozinha já indica a queda.
    trunk = df["trunk_angle"]
    fall = detected & (df["aspect_ratio"] > FALL_ASPECT_RATIO) & ((trunk > FALL_TRUNK_ANGLE) | trunk.isna())
    # Fecha pequenas falhas de detecção (até 2 frames) para não quebrar o evento
    fall = (fall.astype(int).rolling(3, center=True, min_periods=1).max()
            .rolling(3, center=True, min_periods=1).min() > 0)
    add("possivel_queda", "alta", fall, "Paciente em posição horizontal: possível queda.")

    # Inclinação excessiva do tronco (exclui os trechos já classificados como queda)
    lean = detected & (df["trunk_angle"] > MAX_TRUNK_INCLINATION) & ~fall
    add("inclinacao_tronco", "media",
        lean, "Inclinação do tronco acima de {:.0f}°.".format(MAX_TRUNK_INCLINATION), "trunk_angle")

    # Flexão excessiva de joelho (considera o joelho mais dobrado dos dois)
    knee_min = df[["knee_left", "knee_right"]].min(axis=1)
    add("flexao_joelho", "media", detected & (knee_min < MIN_KNEE_ANGLE),
        "Flexão de joelho abaixo de {:.0f}°.".format(MIN_KNEE_ANGLE))

    # Assimetria entre os joelhos
    asymmetry = (df["knee_left"] - df["knee_right"]).abs()
    add("assimetria_joelhos", "baixa", detected & (asymmetry > MAX_KNEE_ASYMMETRY),
        "Diferença entre os joelhos acima de {:.0f}°.".format(MAX_KNEE_ASYMMETRY))

    # Saída da área segura (ex.: leito ou tapete)
    add("saida_area_segura", "media", detected & df["outside_safe_zone"],
        "Paciente fora da área segura delimitada.")

    # Paciente ausente do campo de visão por tempo prolongado (limite próprio, maior que o mínimo)
    absence_frames = max(1, int(round(MAX_ABSENCE_SECONDS * fps)))
    add("paciente_ausente", "media", ~detected,
        "Paciente não detectado por mais de {:.0f} s.".format(MAX_ABSENCE_SECONDS),
        min_len=absence_frames)

    return sorted(events, key=lambda e: e["start_s"])
