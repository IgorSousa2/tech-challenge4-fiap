import json
import os
from datetime import datetime

# Níveis de severidade dos alertas (maior número = mais grave)
SEVERITY_LEVELS = {"baixa": 1, "media": 2, "alta": 3}


def create_alert(source, patient_id, message, severity, timestamp=None, details=None):
    """
    Cria um alerta no formato padrão usado por todos os módulos.
    :param source: Módulo que gerou o alerta (video, audio, multimodal)
    :param patient_id: Identificador do paciente
    :param message: Descrição do alerta em linguagem clara para a equipe médica
    :param severity: 'baixa', 'media' ou 'alta'
    :param timestamp: Momento da anomalia (se None, usa o horário atual)
    :param details: Dicionário com dados adicionais da anomalia
    :return: Dicionário representando o alerta
    """
    if severity not in SEVERITY_LEVELS:
        raise ValueError("Severidade inválida: {}".format(severity))

    return {
        "source": source,
        "patient_id": patient_id,
        "severity": severity,
        "message": message,
        "timestamp": str(timestamp if timestamp is not None else datetime.now()),
        "details": details or {},
    }


def notify_medical_team(alert):
    """
    Notifica a equipe médica sobre um alerta.
    Por enquanto apenas imprime no console; pode ser trocado por e-mail, Teams ou webhook.
    :param alert: Alerta criado por create_alert
    """
    print("[ALERTA {}] Paciente {} | {} | {} | {}".format(
        alert["severity"].upper(), alert["patient_id"], alert["source"],
        alert["timestamp"], alert["message"]))


def save_alerts(alerts, json_path):
    """
    Salva a lista de alertas (ordenada por severidade e horário) em um arquivo .json.
    :param alerts: Lista de alertas
    :param json_path: Caminho do arquivo .json de saída
    """
    ordered = sorted(alerts, key=lambda a: (-SEVERITY_LEVELS[a["severity"]], a["timestamp"]))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(ordered, file, ensure_ascii=False, indent=2)
