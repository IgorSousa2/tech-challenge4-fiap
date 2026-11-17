import os
import urllib.request

import openpose_estimator

# Prototxt oficial (repositório do OpenPose) e pesos treinados no COCO (~200 MB)
FILES = [
    (openpose_estimator.PROTO_PATH,
     "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/"
     "models/pose/coco/pose_deploy_linevec.prototxt"),
    (openpose_estimator.WEIGHTS_PATH,
     # O servidor original da CMU saiu do ar; o espelho do Hugging Face é tentado primeiro
     ["https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel",
      "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"]),
]


def download(urls, path):
    """
    Baixa um arquivo, exibindo o progresso. Tenta cada URL até uma funcionar.
    O download vai para um arquivo temporário para não deixar um arquivo parcial.
    """
    def progress(blocks, block_size, total):
        if total > 0:
            print("\r  {:.0f}%".format(min(100, blocks * block_size * 100 / total)), end="")

    if isinstance(urls, str):
        urls = [urls]
    for url in urls:  # tenta as fontes em ordem; passa à próxima se uma falhar
        try:
            urllib.request.urlretrieve(url, path + ".part", progress)
            os.replace(path + ".part", path)
            print()
            return
        except OSError as e:
            print("\n  Falhou em {}: {}".format(url, e))
    raise SystemExit("Não foi possível baixar {}.".format(os.path.basename(path)))


def main():
    os.makedirs(openpose_estimator.MODELS_DIR, exist_ok=True)
    for path, url in FILES:
        if os.path.exists(path):
            print("Já existe: {}".format(os.path.basename(path)))
            continue
        print("Baixando {} ...".format(os.path.basename(path)))
        download(url, path)
    print("Modelos do OpenPose prontos em '{}'.".format(openpose_estimator.MODELS_DIR))


if __name__ == "__main__":
    main()
