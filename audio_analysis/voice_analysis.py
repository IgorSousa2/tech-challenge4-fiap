import numpy as np
from scipy.io import wavfile

# Limiares das regras vocais (valores ilustrativos, precisam de calibração com dados reais)
MIN_PAUSE_SECONDS = 0.3        # silêncio mínimo para contar como pausa
MAX_PAUSE_RATIO = 0.35         # fração do tempo em pausa acima da qual a fala é considerada entrecortada
MAX_LONG_PAUSE_SECONDS = 3.0   # pausa única considerada longa
MIN_SPEECH_RATE_WPM = 100      # palavras por minuto (fala normal em português: ~130 a 180)
MIN_ENERGY_SLOPE_DB = -4.0     # queda de energia (dB por minuto) indicativa de cansaço
MIN_PITCH_VARIATION = 0.05     # coeficiente de variação do pitch abaixo do qual a fala é monótona


def load_audio(audio_path):
    """
    Lê um arquivo .wav e devolve o sinal mono normalizado entre -1 e 1.
    :return: Tupla (sinal, taxa de amostragem)
    """
    rate, data = wavfile.read(audio_path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)  # estéreo -> mono (média dos canais)
    peak = np.max(np.abs(data))  # normaliza pelo pico, para independer do volume da gravação
    return (data / peak if peak > 0 else data), rate


def frame_signal(signal, rate, frame_ms=30, hop_ms=10):
    """
    Divide o sinal em quadros curtos sobrepostos.
    :return: Tupla (matriz de quadros, tamanho do passo em segundos)
    """
    frame_len = int(rate * frame_ms / 1000)  # amostras por quadro (30 ms)
    hop = int(rate * hop_ms / 1000)  # avanço entre quadros (10 ms): os quadros se sobrepõem
    count = max(1, 1 + (len(signal) - frame_len) // hop)
    frames = np.stack([signal[i * hop:i * hop + frame_len] for i in range(count)])
    return frames, hop / rate


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


def estimate_pitch(frame, rate, fmin=75, fmax=400):
    """
    Estima a frequência fundamental (pitch) de um quadro por autocorrelação.
    :return: Pitch em Hz, ou NaN se o quadro não for claramente vozeado
    """
    frame = frame - frame.mean()
    # Autocorrelação: um sinal periódico (voz) se parece consigo mesmo deslocado de um período
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if corr[0] <= 0:
        return float("nan")
    # Só procura deslocamentos (lags) correspondentes a vozes humanas (75 a 400 Hz)
    low, high = int(rate / fmax), min(int(rate / fmin), len(corr) - 1)
    if high <= low:
        return float("nan")
    lag = low + int(np.argmax(corr[low:high]))  # o maior pico indica o período da voz
    # Pico de autocorrelação fraco = quadro sem voz (ruído ou fricativa)
    if corr[lag] / corr[0] < 0.3:
        return float("nan")
    return rate / lag  # frequência = amostras por segundo / amostras por período


def extract_voice_features(audio_path, word_count=None):
    """
    Extrai características vocais relacionadas a cansaço, dificuldade respiratória e disartria.
    :param audio_path: Caminho do arquivo .wav
    :param word_count: Número de palavras da transcrição (opcional, para taxa de fala)
    :return: Dicionário de características
    """
    signal, rate = load_audio(audio_path)
    frames, step = frame_signal(signal, rate)
    rms = np.sqrt((frames ** 2).mean(axis=1))  # energia (volume) de cada quadro

    # Quadros com fala: energia acima de 10% do nível típico de fala
    speech = rms > 0.1 * np.percentile(rms, 95)
    speech_idx = np.where(speech)[0]
    if len(speech_idx) == 0:
        raise ValueError("Nenhuma fala detectada no áudio.")

    first, last = speech_idx[0], speech_idx[-1]
    span_s = (last - first + 1) * step  # duração da fala, do primeiro ao último quadro com voz

    # Pausas internas (ignora o silêncio antes e depois da fala)
    pauses = find_runs(~speech[first:last + 1], int(MIN_PAUSE_SECONDS / step))
    pause_lengths = [(end - start + 1) * step for start, end in pauses]
    pause_total = sum(pause_lengths)

    # Tendência de energia ao longo do tempo (dB por minuto)
    times = speech_idx * step
    energy_db = 20 * np.log10(rms[speech_idx] + 1e-9)  # energia em decibéis
    # Reta ajustada (energia x tempo): inclinação negativa = voz perdendo força; *60 converte para dB/min
    slope_db_min = float(np.polyfit(times, energy_db, 1)[0] * 60) if len(speech_idx) > 10 else 0.0

    # Pitch nos quadros com fala (um a cada dois, para reduzir custo)
    pitches = np.array([estimate_pitch(frames[i], rate) for i in speech_idx[::2]])
    pitches = pitches[~np.isnan(pitches)]
    mean_f0 = float(pitches.mean()) if len(pitches) else float("nan")
    # Coeficiente de variação (desvio / média): mede o quanto a entonação varia
    f0_cv = float(pitches.std() / mean_f0) if len(pitches) > 5 else float("nan")

    features = {
        "duration_s": round(len(signal) / rate, 2),
        "speech_span_s": round(span_s, 2),
        "pause_ratio": round(pause_total / span_s, 3) if span_s else 0.0,
        "pause_count": len(pauses),
        "longest_pause_s": round(max(pause_lengths), 2) if pause_lengths else 0.0,
        "energy_slope_db_per_min": round(slope_db_min, 2),
        "mean_f0_hz": round(mean_f0, 1),
        "f0_variation": round(f0_cv, 3),
        "speech_rate_wpm": round(word_count / (span_s / 60), 1) if word_count and span_s else None,
    }
    return features


def detect_voice_indicators(features):
    """
    Aplica as regras sobre as características e devolve os indicadores encontrados.
    :return: Lista de dicionários com tipo, severidade e mensagem
    """
    indicators = []

    def add(kind, severity, message):
        indicators.append({"type": kind, "severity": severity, "message": message})

    if features["pause_ratio"] > MAX_PAUSE_RATIO:
        add("fala_entrecortada", "media",
            "Pausas ocupam {:.0f}% da fala: possível cansaço ou dificuldade respiratória.".format(
                features["pause_ratio"] * 100))
    if features["longest_pause_s"] > MAX_LONG_PAUSE_SECONDS:
        add("pausa_longa", "media",
            "Pausa de {:.1f} s durante a fala.".format(features["longest_pause_s"]))
    if features["energy_slope_db_per_min"] < MIN_ENERGY_SLOPE_DB:
        add("queda_de_energia", "media",
            "Volume da voz caindo {:.1f} dB/min: possível fadiga.".format(
                -features["energy_slope_db_per_min"]))
    if features["speech_rate_wpm"] is not None and features["speech_rate_wpm"] < MIN_SPEECH_RATE_WPM:
        add("fala_lenta", "media",
            "Fala lenta ({:.0f} palavras/min): possível disartria ou fadiga.".format(
                features["speech_rate_wpm"]))
    # (x == x é falso para NaN: ignora a regra quando o pitch não pôde ser medido)
    if features["f0_variation"] == features["f0_variation"] and features["f0_variation"] < MIN_PITCH_VARIATION:
        add("fala_monotona", "baixa",
            "Pouca variação de entonação (CV do pitch {:.3f}).".format(features["f0_variation"]))

    # Vários indicadores juntos aumentam a gravidade
    if len(indicators) >= 3:
        add("indicio_de_fadiga_ou_disartria", "alta",
            "{} indicadores vocais simultâneos: avaliação clínica recomendada.".format(len(indicators)))
    return indicators
