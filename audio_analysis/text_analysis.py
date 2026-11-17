import re
import unicodedata

# Termos críticos (já sem acento, em minúsculas) e a severidade associada
CRITICAL_TERMS = {
    "dor no peito": "alta",
    "falta de ar": "alta",
    "nao consigo respirar": "alta",
    "dificuldade para respirar": "alta",
    "desmaio": "alta",
    "desmaiei": "alta",
    "sangramento": "alta",
    "confusao": "alta",
    "formigamento": "media",
    "tontura": "media",
    "fraqueza": "media",
    "cansaco": "media",
    "cansado": "media",
    "cansada": "media",
    "enjoo": "media",
    "nausea": "media",
    "febre": "media",
    "dor de cabeca": "media",
    "palpitacao": "media",
    "dor": "baixa",
}

# Léxico simples de sentimento (substitui um serviço externo de análise de sentimento)
NEGATIVE_WORDS = {"dor", "pior", "ruim", "mal", "medo", "preocupado", "preocupada", "angustia",
                  "cansaco", "cansado", "cansada", "fraco", "fraca", "tontura", "sofrendo",
                  "piorou", "nao", "sem", "difícil", "dificil", "insuportavel"}
POSITIVE_WORDS = {"melhor", "bem", "otimo", "boa", "bom", "melhorou", "tranquilo", "tranquila",
                  "aliviado", "aliviada", "recuperando"}


def normalize(text):
    """
    Converte para minúsculas e remove acentos e pontuação, para facilitar a busca de termos.
    """
    text = unicodedata.normalize("NFKD", text.lower())  # separa letra e acento ("é" -> "e" + "´")
    text = "".join(c for c in text if not unicodedata.combining(c))  # descarta os acentos
    return re.sub(r"[^a-z0-9\s]", " ", text)  # troca pontuação por espaço


def find_critical_terms(text):
    """
    Procura termos críticos no texto transcrito.
    Termos mais específicos têm prioridade (ex.: 'dor no peito' não conta também como 'dor').
    :return: Lista de dicionários com termo, severidade e ocorrências
    """
    normalized = " " + re.sub(r"\s+", " ", normalize(text)) + " "
    found = []
    # Dos termos mais longos para os mais curtos, para "dor no peito" ser vista antes de "dor"
    for term in sorted(CRITICAL_TERMS, key=len, reverse=True):
        pattern = r"\b{}\b".format(re.escape(term))  # \b = palavra inteira ("dor" não casa em "corredor")
        count = len(re.findall(pattern, normalized))
        if count:
            found.append({"term": term, "severity": CRITICAL_TERMS[term], "count": count})
            # Remove o trecho encontrado para não contar de novo em termos mais curtos
            normalized = re.sub(pattern, " ", normalized)
    return found


def analyze_sentiment(text):
    """
    Estima o sentimento com base no léxico: pontuação de -1 (negativo) a 1 (positivo).
    :return: Dicionário com score, label e contagens
    """
    words = normalize(text).split()
    negative = sum(1 for w in words if w in NEGATIVE_WORDS)
    positive = sum(1 for w in words if w in POSITIVE_WORDS)
    total = negative + positive
    score = (positive - negative) / total if total else 0.0  # -1 = só negativas, +1 = só positivas

    if score <= -0.3:
        label = "negativo"
    elif score >= 0.3:
        label = "positivo"
    else:
        label = "neutro"
    return {"score": round(score, 2), "label": label, "negative_words": negative,
            "positive_words": positive}
