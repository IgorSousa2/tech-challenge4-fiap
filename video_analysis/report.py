import json
import os
import sys
from datetime import timedelta

# Torna o alerts.py (pasta raiz do projeto) importável a partir desta subpasta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from alerts import create_alert  # noqa: E402


def build_alerts(events, patient_id, start_time):
    """
    Converte os desvios detectados no vídeo em alertas para a equipe médica.
    :param events: Lista de eventos de detect_deviations
    :param patient_id: Identificador do paciente
    :param start_time: datetime correspondente ao início do vídeo
    """
    alerts = []
    for event in events:
        # Horário do alerta = início da análise + instante do desvio no vídeo
        timestamp = start_time + timedelta(seconds=event["start_s"])
        message = "{} (vídeo {:.1f}s a {:.1f}s)".format(event["message"], event["start_s"], event["end_s"])
        alerts.append(create_alert("video", patient_id, message, event["severity"], timestamp,
                                   {"type": event["type"], "peak_value": event.get("peak_value")}))
    return alerts


def build_summary(df, events, video_path, fps):
    """
    Resume o processamento do vídeo em um dicionário.
    """
    detected = df["detected"]
    by_type = {}
    for event in events:
        by_type[event["type"]] = by_type.get(event["type"], 0) + 1

    return {
        "video": os.path.basename(video_path),
        "duration_s": float(df["time_s"].iloc[-1]) if len(df) else 0.0,
        "frames_analyzed": int(len(df)),
        "analysis_fps": float(fps),
        "frames_with_patient": int(detected.sum()),
        "max_trunk_angle": float(df["trunk_angle"].max()) if detected.any() else None,
        "min_knee_angle": float(df[["knee_left", "knee_right"]].min().min()) if detected.any() else None,
        "critical_object_frames": int(df["critical_object"].sum()),
        "deviations_total": len(events),
        "deviations_by_type": by_type,
    }


def save_report(summary, events, json_path, md_path):
    """
    Salva o relatório automático de desvios em .json e .md.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump({"summary": summary, "deviations": events}, file, ensure_ascii=False, indent=2)

    lines = [
        "# Relatório de análise de vídeo",
        "",
        "- Vídeo: {}".format(summary["video"]),
        "- Duração analisada: {:.1f} s ({} frames)".format(summary["duration_s"], summary["frames_analyzed"]),
        "- Frames com paciente detectado: {}".format(summary["frames_with_patient"]),
        "- Frames com objeto cortante (faca/tesoura): {}".format(summary["critical_object_frames"]),
        "- Desvios detectados: {}".format(summary["deviations_total"]),
        "",
        "## Desvios",
        "",
    ]
    if events:
        lines += ["| Início (s) | Fim (s) | Severidade | Tipo | Descrição |", "|---|---|---|---|---|"]
        for e in events:
            lines.append("| {:.1f} | {:.1f} | {} | {} | {} |".format(
                e["start_s"], e["end_s"], e["severity"], e["type"], e["message"]))
    else:
        lines.append("Nenhum desvio detectado.")

    with open(md_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(lines) + "\n")
