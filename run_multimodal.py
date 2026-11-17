import argparse
import json
import os
import sys
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# Permite importar os módulos das subpastas (video_analysis e audio_analysis) como bibliotecas
for folder in ('video_analysis', 'audio_analysis'):
    sys.path.insert(0, os.path.join(script_dir, folder))

import alerts as alert_module

SEVERITY_ORDER = alert_module.SEVERITY_LEVELS  # baixa=1, media=2, alta=3


def risk_level(alerts):
    """
    Classifica o risco geral a partir da maior severidade entre os alertas.
    """
    if not alerts:
        return "normal"
    top = max(SEVERITY_ORDER[a["severity"]] for a in alerts)  # o risco segue o alerta mais grave
    return {3: "alto", 2: "moderado", 1: "baixo"}[top]


def fuse_alerts(video_alerts, audio_alerts, patient_id, timestamp):
    """
    Une os alertas das duas modalidades e aplica a regra de convergência:
    sinais de gravidade média ou alta em áudio E em vídeo geram um alerta multimodal.
    :return: Lista com todos os alertas (inclusive o de convergência, se houver)
    """
    all_alerts = list(video_alerts) + list(audio_alerts)

    def relevant(alerts):
        return [a for a in alerts if SEVERITY_ORDER[a["severity"]] >= SEVERITY_ORDER["media"]]

    video_rel, audio_rel = relevant(video_alerts), relevant(audio_alerts)
    # Convergência: o mesmo paciente apresenta sinais relevantes nas duas modalidades ao mesmo tempo
    if video_rel and audio_rel:
        all_alerts.append(alert_module.create_alert(
            "multimodal", patient_id,
            "Sinais convergentes: {} alerta(s) relevante(s) no vídeo e {} no áudio. "
            "Avaliação clínica prioritária.".format(len(video_rel), len(audio_rel)),
            "alta", timestamp, {"video_alerts": len(video_rel), "audio_alerts": len(audio_rel)}))
    return all_alerts


def save_report(patient_id, level, all_alerts, summaries, json_path, md_path):
    """
    Salva o relatório consolidado do paciente em .json e .md.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    ordered = sorted(all_alerts, key=lambda a: -SEVERITY_ORDER[a["severity"]])

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump({"patient_id": patient_id, "risk_level": level, "summaries": summaries,
                   "alerts": ordered}, file, ensure_ascii=False, indent=2, default=str)

    lines = ["# Relatório multimodal", "",
             "- Paciente: {}".format(patient_id),
             "- Risco geral: **{}**".format(level),
             "- Total de alertas: {}".format(len(ordered)), ""]
    for source in ("video", "audio", "multimodal"):
        subset = [a for a in ordered if a["source"] == source]
        if subset:
            lines += ["## Alertas de {}".format(source), ""]
            lines += ["- [{}] {}".format(a["severity"], a["message"]) for a in subset] + [""]

    with open(md_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Análise multimodal (vídeo + áudio) de um paciente.")
    parser.add_argument("--video", help="Caminho do vídeo clínico (.mp4)")
    parser.add_argument("--audio", help="Caminho do áudio da consulta (.wav)")
    parser.add_argument("--patient", default="P001", help="Identificador do paciente")
    parser.add_argument("--output", default=os.path.join(script_dir, 'output'), help="Pasta de saída")
    args = parser.parse_args()

    if not args.video and not args.audio:
        parser.error("Informe ao menos --video ou --audio.")

    start_time = datetime.now()  # referência de horário comum aos alertas das duas modalidades
    video_alerts, audio_alerts, summaries = [], [], {}

    # Cada modalidade é opcional: roda só a que foi informada
    if args.video:
        import video_analysis  # carregado só se houver vídeo (depende de YOLO/torch)
        video_analysis.PATIENT_ID = args.patient
        print("== Análise de vídeo ==")
        result, summary = video_analysis.run_analysis(args.video, os.path.join(args.output, 'video'), start_time)
        if result is None:
            print("Não foi possível processar o vídeo.")
        else:
            video_alerts, summaries["video"] = result, summary

    if args.audio:
        import audio_analysis
        audio_analysis.PATIENT_ID = args.patient
        print("== Análise de áudio ==")
        audio_alerts, summaries["audio"] = audio_analysis.run_analysis(
            args.audio, os.path.join(args.output, 'audio'), start_time)

    # Fusão: junta os alertas, aplica a regra de convergência e calcula o risco geral
    all_alerts = fuse_alerts(video_alerts, audio_alerts, args.patient, start_time)
    level = risk_level(all_alerts)

    print("\n== Alertas para a equipe médica ==")
    for alert in sorted(all_alerts, key=lambda a: -SEVERITY_ORDER[a["severity"]]):
        alert_module.notify_medical_team(alert)

    save_report(args.patient, level, all_alerts, summaries,
                os.path.join(args.output, 'multimodal_report.json'),
                os.path.join(args.output, 'multimodal_report.md'))
    alert_module.save_alerts(all_alerts, os.path.join(args.output, 'multimodal_alerts.json'))
    print("\nRisco geral: {} | {} alertas. Relatório em '{}'.".format(level, len(all_alerts), args.output))


if __name__ == "__main__":
    main()
