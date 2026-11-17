# Monitoramento Multimodal de Pacientes (Vídeo + Áudio + Texto) com YOLOv8

## Descrição

Este projeto implementa um sistema de monitoramento clínico multimodal, que analisa **vídeos** (sessões de fisioterapia ou cirurgias gravadas) e **áudios** (consultas médicas) para identificar sinais precoces de risco e alertar a equipe médica.

O sistema é dividido em módulos independentes, unificados por um orquestrador:

- **`video_analysis/`** — processa o vídeo com **OpenPose** (análise postural) e **YOLOv8** (detecção da pessoa, de objetos e de áreas críticas), extrai características posturais frame a frame, detecta desvios do padrão esperado e gera um relatório automático.
- **`audio_analysis/`** — processa o áudio da consulta, extrai características vocais (pausas, energia, pitch, taxa de fala), transcreve a fala e identifica termos críticos e sentimento no relato.
- **`run_multimodal.py`** — executa os dois módulos para o mesmo paciente, funde os resultados (regra de convergência entre áudio e vídeo), calcula o risco geral e emite os alertas para a equipe médica.
- **`alerts.py`** — formato padrão de alerta usado por todos os módulos.

O código segue a estrutura dos scripts desenvolvidos em aula (`facial_detection.py`, `transcribe_audio.py`, `detect_expression_video.py` etc.): funções pequenas e as mesmas bibliotecas (OpenCV, SpeechRecognition, ultralytics).

## Fluxo Multimodal

```
Vídeo clínico                       Áudio da consulta
      │                                    │
      ▼                                    ▼
YOLOv8 + OpenPose                   Características vocais
(postura, objetos, área segura)     + Transcrição (texto)
      │                                    │
      ▼                                    ▼
Regras de desvio postural           Indicadores vocais, termos
(queda, tronco, joelhos, ausência)  críticos e sentimento
      │                                    │
      └──────────────┬─────────────────────┘
                     ▼
        Fusão multimodal (run_multimodal.py)
        - une os alertas
        - convergência áudio + vídeo
        - risco geral (normal/baixo/moderado/alto)
                     │
                     ▼
        Alertas para a equipe médica + relatório
```

## Modelos e Técnicas por Tipo de Dado

| Dado | Técnica | Resultado |
|---|---|---|
| Vídeo (postura) | OpenPose (modelo COCO de 18 keypoints, via OpenCV DNN) + cálculo de ângulos | Inclinação do tronco, ângulo dos joelhos, possível queda |
| Vídeo (pessoa e objetos) | YOLOv8 (`yolov8n.pt`) | Localiza o paciente (recorte enviado ao OpenPose), instrumentos cortantes (faca/tesoura) e saída da área segura |
| Áudio (voz) | Análise acústica com NumPy/SciPy (energia, pausas, autocorrelação de pitch) | Indícios de fadiga, fala entrecortada, fala lenta e monótona |
| Texto (transcrição) | `SpeechRecognition` + léxico de termos críticos e de sentimento | Termos críticos (ex.: "falta de ar", "dor no peito") e sentimento |

## Estrutura do Projeto

```
.
├── run_multimodal.py              # orquestrador (fusão de vídeo + áudio)
├── alerts.py                      # formato padrão de alerta (usado por todos os módulos)
├── video_analysis/
│   ├── video_analysis.py          # processamento do vídeo (YOLOv8 + OpenPose)
│   ├── openpose_estimator.py      # estimador de pose OpenPose (OpenCV DNN)
│   ├── download_openpose_models.py# baixa os modelos do OpenPose (~200 MB)
│   ├── pose_analysis.py           # características posturais e regras de desvio
│   ├── report.py                  # relatório automático e alertas
│   └── requirements.txt
├── audio_analysis/
│   ├── audio_analysis.py          # orquestração da análise de áudio
│   ├── voice_analysis.py          # características e indicadores vocais
│   ├── text_analysis.py           # termos críticos e sentimento
│   └── requirements.txt
├── requirements.txt               # todas as dependências do projeto
├── test_data/                     # arquivos para testar o sistema
│   ├── ur_fall_videos/            # 5 vídeos de queda (UR Fall Detection Dataset)
│   ├── youtube/                   # 2 vídeos de teste (caminhada e levantar/sentar)
│   └── synthetic_audio/           # 6 áudios sintéticos de consulta (.wav + .txt)
└── output/                        # (gerado) relatórios e alertas
```

## Requisitos

- Python 3.10 ou superior
- Um vídeo clínico (`.mp4`) e/ou um áudio de consulta (`.wav`) para análise
- Conexão com a internet na primeira execução: o `ultralytics` baixa os pesos do YOLOv8 automaticamente, e o script `download_openpose_models.py` baixa os modelos do OpenPose (veja o passo 3). A transcrição automática do áudio (`SpeechRecognition`) também usa a internet; para evitar isso, basta salvar a transcrição em um `.txt` com o mesmo nome do áudio (ex.: `consulta.wav` e `consulta.txt`).
- GPU **não** é necessária: os modelos YOLOv8 usados (versão *nano*) rodam em CPU.

## Execução

### 1. Criar um ambiente virtual (opcional, mas recomendado)

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

O `requirements.txt` fixa `opencv-python<5`: o OpenCV 5 removeu o leitor de modelos Caffe, necessário para o OpenPose.

### 3. Baixar os modelos do OpenPose (uma única vez)

```bash
python video_analysis/download_openpose_models.py
```

Baixa o `pose_deploy_linevec.prototxt` e o `pose_iter_440000.caffemodel` (~200 MB) para `video_analysis/models/`. O OpenPose é executado pelo módulo DNN do OpenCV, sem necessidade de compilar a biblioteca original.

### 4. Executar a análise multimodal

```bash
python run_multimodal.py --video caminho/do/video.mp4 --audio caminho/do/audio.wav --patient P001
```

Também é possível informar apenas `--video` ou apenas `--audio`. Para alterar a pasta de saída, use `--output`.

Exemplo com os arquivos de teste incluídos:

```bash
python run_multimodal.py --video test_data/ur_fall_videos/fall-01.mp4 --audio test_data/synthetic_audio/06_caso_grave_combinado.wav
```

Ao final, são gerados em `output/`:

- `multimodal_report.md` e `multimodal_report.json` — relatório consolidado com o risco geral do paciente;
- `multimodal_alerts.json` — alertas ordenados por severidade;
- `video/` — vídeo anotado, `pose_features.csv`, `movement.csv`, `video_report.md/json` e alertas do vídeo;
- `audio/` — `audio_report.md/json` (com a transcrição) e alertas do áudio.

### 5. Executar cada módulo separadamente (opcional)

```bash
python video_analysis/video_analysis.py caminho/do/video.mp4
python audio_analysis/audio_analysis.py caminho/do/audio.wav
```

## Regras de Detecção

### Vídeo

- **Possível queda** — corpo na horizontal (caixa mais larga que alta) e tronco muito inclinado, ou tronco não detectável por estar deitado (severidade alta).
- **Inclinação excessiva do tronco** — acima de 35° em relação à vertical.
- **Flexão excessiva do joelho** — ângulo abaixo de 60°.
- **Assimetria entre os joelhos** — diferença acima de 25°.
- **Saída da área segura** — o paciente sai da região delimitada (ex.: leito ou tapete).
- **Paciente ausente** — não detectado por mais de 5 s.

### Áudio

- **Fala entrecortada / pausa longa** — pausas ocupando mais de 35% da fala ou pausa única acima de 3 s.
- **Queda de energia** — volume da voz caindo mais de 4 dB por minuto (possível fadiga).
- **Fala lenta** — menos de 100 palavras por minuto.
- **Fala monótona** — baixa variação de entonação.
- **Termos críticos** — termos como "dor no peito" e "falta de ar" (alta) ou "tontura" e "cansaço" (média).
- **Vários indicadores vocais simultâneos** (3 ou mais) elevam o alerta para severidade alta.

### Fusão multimodal

Alertas de severidade média ou alta **no vídeo e no áudio ao mesmo tempo** geram um alerta multimodal de severidade alta. O risco geral é definido pela maior severidade encontrada (`normal`, `baixo`, `moderado` ou `alto`).

## Vídeo de Demonstração

A demonstração completa do funcionamento do projeto está disponível em:

(inserir link do YouTube/Vimeo)

## Observações

- O sistema tem finalidade acadêmica e de demonstração (Tech Challenge 4).
- Os limiares das regras (ângulos, pausas, taxa de fala etc.) são ilustrativos e não foram calibrados clinicamente. Os indicadores vocais são heurísticas de apoio, não diagnóstico.
- Os alertas **não substituem avaliação médica profissional** — servem apenas como apoio à decisão da equipe.
- Os dados utilizados são simulados ou de uso público; nenhuma informação real de paciente é utilizada.
- A análise postural estima a pose de **uma pessoa por vez** (o paciente, identificado como a maior pessoa detectada pelo YOLOv8 no quadro).

## Autor
- Igor de Sousa
- RM 371788
- FIAP - Pós-Tech IA para Devs - 9IADT
- Grupo (inserir número do grupo)
