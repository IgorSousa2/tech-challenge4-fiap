import json
import os
import sys
from datetime import datetime

import speech_recognition as sr

import voice_analysis
import text_analysis

# Torna o alerts.py (pasta raiz do projeto) importável a partir desta subpasta
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import alerts as alert_module  # noqa: E402

PATIENT_ID = 'P001'


def transcribe_audio_to_text(audio_path):
    """
    Transcreve o áudio (mesma abordagem do transcribe_audio.py da aula).
    :return: Texto transcrito, ou string vazia se não foi possível transcrever
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="pt-BR")
        except sr.UnknownValueError:
            print("Não foi possível entender o áudio.")
        except sr.RequestError as e:
            print("Erro ao solicitar a transcrição; {0}".format(e))
    return ""


def get_transcription(audio_path):
    """
    Usa a transcrição já salva (mesmo nome do áudio, extensão .txt), se existir;
    caso contrário, transcreve o áudio e salva o resultado.
    """
    txt_path = os.path.splitext(audio_path)[0] + '.txt'  # consulta.wav -> consulta.txt
    if os.path.exists(txt_path):  # evita depender da internet quando a transcrição já existe
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    text = transcribe_audio_to_text(audio_path)
    if text:
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write(text)
    return text


def build_alerts(indicators, terms, sentiment, start_time):
    """
    Converte os achados de voz e de texto em alertas para a equipe médica.
    """
    alerts = []
    # Um alerta por indicador vocal, por termo crítico e, se for o caso, pelo sentimento negativo
    for ind in indicators:
        alerts.append(alert_module.create_alert("audio", PATIENT_ID, ind["message"],
                                                ind["severity"], start_time, {"type": ind["type"]}))
    for t in terms:
        alerts.append(alert_module.create_alert(
            "audio", PATIENT_ID,
            "Termo crítico citado na consulta: '{}' ({}x).".format(t["term"], t["count"]),
            t["severity"], start_time, {"type": "termo_critico", "term": t["term"]}))
    if sentiment["label"] == "negativo":
        alerts.append(alert_module.create_alert(
            "audio", PATIENT_ID, "Sentimento predominantemente negativo no relato (score {}).".format(
                sentiment["score"]), "baixa", start_time, {"type": "sentimento_negativo"}))
    return alerts


def save_report(audio_path, features, indicators, terms, sentiment, text, json_path, md_path):
    """
    Salva o relatório da análise de áudio em .json e .md.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump({"audio": os.path.basename(audio_path), "features": features,
                   "indicators": indicators, "critical_terms": terms,
                   "sentiment": sentiment, "transcription": text}, file, ensure_ascii=False, indent=2)

    lines = ["# Relatório de análise de áudio", "",
             "- Áudio: {}".format(os.path.basename(audio_path)),
             "- Duração: {} s | Pausas: {} ({:.0f}% da fala) | Maior pausa: {} s".format(
                 features["duration_s"], features["pause_count"],
                 features["pause_ratio"] * 100, features["longest_pause_s"]),
             "- Pitch médio: {} Hz (variação {}) | Tendência de energia: {} dB/min".format(
                 features["mean_f0_hz"], features["f0_variation"], features["energy_slope_db_per_min"]),
             "- Taxa de fala: {} palavras/min".format(features["speech_rate_wpm"] or "n/d"),
             "- Sentimento: {} (score {})".format(sentiment["label"], sentiment["score"]),
             "", "## Indicadores vocais", ""]
    lines += ["- [{}] {}".format(i["severity"], i["message"]) for i in indicators] or ["Nenhum indicador."]
    lines += ["", "## Termos críticos", ""]
    lines += ["- [{}] {} ({}x)".format(t["severity"], t["term"], t["count"]) for t in terms] or ["Nenhum termo."]
    lines += ["", "## Transcrição", "", text or "(não disponível)"]

    with open(md_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(lines) + "\n")


def run_analysis(audio_path, output_dir, start_time):
    """
    Executa a análise completa do áudio e grava os arquivos de saída em output_dir.
    :return: Tupla (lista de alertas, resumo)
    """
    # Texto: transcrição (ou .txt existente); o total de palavras alimenta a taxa de fala
    text = get_transcription(audio_path)
    word_count = len(text.split()) if text else None

    # Voz: características acústicas e as regras sobre elas
    features = voice_analysis.extract_voice_features(audio_path, word_count)
    indicators = voice_analysis.detect_voice_indicators(features)
    # Texto: termos críticos e sentimento (sem transcrição, ficam vazios/neutros)
    terms = text_analysis.find_critical_terms(text) if text else []
    sentiment = text_analysis.analyze_sentiment(text) if text else {"label": "neutro", "score": 0.0}

    save_report(audio_path, features, indicators, terms, sentiment, text,
                os.path.join(output_dir, 'audio_report.json'), os.path.join(output_dir, 'audio_report.md'))

    audio_alerts = build_alerts(indicators, terms, sentiment, start_time)
    alert_module.save_alerts(audio_alerts, os.path.join(output_dir, 'audio_alerts.json'))

    summary = {"audio": os.path.basename(audio_path), "voice_indicators": len(indicators),
               "critical_terms": len(terms), "sentiment": sentiment["label"], "features": features}
    return audio_alerts, summary


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'audio.wav')
    output_dir = os.path.join(script_dir, 'output')

    audio_alerts, _ = run_analysis(audio_path, output_dir, datetime.now())
    for alert in audio_alerts:
        alert_module.notify_medical_team(alert)
    print("\n{} alertas gerados. Relatório em '{}'.".format(len(audio_alerts), output_dir))


if __name__ == "__main__":
    main()
