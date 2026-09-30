import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import SAM

# 1. Carregar imagem
image_path = "/content/uma-enorme-coluna-negra-de-fumaça-e-horizonte-no-rio-janeiro-durante-o-verão-251875642.webp"  # Substitua pelo nome do seu arquivo
img = cv2.imread(image_path)
h, w, _ = img.shape

# 2. Carregar modelo SAM na GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAM("sam2.1_b.pt")

# 3. Definir Bounding Box [x1, y1, x2, y2] cobrindo a fumaça
# A fumaça começa no topo esquerdo (x~0, y~0) e vai até a base direita (x~75%, y~78%)
bbox = [int(w * 0.0), int(h * 0.0), int(w * 0.78), int(h * 0.78)]

# 4. Definir múltiplos pontos dentro da parte escura da fumaça para reforçar
points = [
    [int(w * 0.68), int(h * 0.65)],  # Base da coluna escura
    [int(w * 0.60), int(h * 0.45)],  # Meio da fumaça
    [int(w * 0.30), int(h * 0.20)],  # Parte alta se espalhando
]
labels = [1, 1, 1]  # 1 indica que o ponto faz parte da pluma

# 5. Executar a segmentação direcionada
results = model(
    image_path, bboxes=[bbox], points=[points], labels=[labels], device=device
)

# 6. Exibir o Resultado com a Máscara da Pluma
plt.figure(figsize=(10, 12))

# Desenhar a máscara sobre a imagem original
res_img = results[0].plot()
res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

plt.imshow(res_img_rgb)
plt.title("Segmentação Precisa da Pluma de Fumaça")
plt.axis("off")
plt.show()

# 7. Isolar apenas a Máscara Binária da Pluma (0 = Fundo, 255 = Pluma)
if results[0].masks is not None:
    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara Binária da Pluma (Pronta para Análise de Área/Forma)")
    plt.axis("off")
    plt.show()
