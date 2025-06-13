
# Predicción de Engagement en Puntos de Interés Turísticos

Este proyecto forma parte del módulo de Deep Learning del bootcamp de KeepCoding.  
El objetivo es construir un modelo **multimodal** que prediga si un punto de interés turístico (POI) generará **engagement alto o bajo** en base a:

- **Metadatos estructurados** (visitas, likes, dislikes, bookmarks, tags, ubicación, etc.)
- **Imagen asociada** al POI (ruta `main_image_path`)

---

## Objetivo del proyecto

Desarrollar una red neuronal multimodal que combine:

- Una **CNN** (ResNet18 preentrenada) para extraer características visuales de la imagen del POI
- Una **DNN** para procesar los metadatos estructurados
- Una **capa de fusión** que combine ambos flujos y genere la predicción final

El modelo fue entrenado para clasificar si el nivel de engagement esperado es **alto (1)** o **bajo (0)**.

---

## Dataset

El dataset contiene un registro por POI con:

- Métricas de interacción: `visits`, `likes`, `dislikes`, `bookmarks`
- Atributos: `xps`, `tier`, `tags`, `categories`, coordenadas geográficas
- Imagen: ruta relativa en la columna `main_image_path`

La variable objetivo (`engagement_target`) fue construida a partir de una métrica sintética:

```python
engagement_score = log1p(Visits)*0.4 + log1p(Likes)*0.3 + log1p(Bookmarks)*0.3 - log1p(Dislikes)*0.4
engagement_target = 1 si engagement_score >= percentil 60, sino 0
```

---

## Modelo entrenado

Este repositorio incluye el modelo entrenado final con sus pesos:

- Arquitectura: `MultimodalNet` (CNN + DNN con capa de fusión)
- CNN basada en `ResNet18`, con pesos de ImageNet (congelada)
- Regularización: `Dropout`, `L2` y `EarlyStopping`
- Función de pérdida: `BCELoss`
- Optimización: `Adam`, batch size 32, hasta 50 épocas

### Pesos disponibles

El archivo `best_model.pth` contiene los pesos entrenados del mejor modelo.

Para usar el modelo entrenado:

```python
from modelo import MultimodalNet
import torch

# Número de features estructurados (ejemplo)
NUM_FEATS = 8  

model = MultimodalNet(num_meta_features=NUM_FEATS)
model.load_state_dict(torch.load("model/best_model.pth"))
model.eval()
```

---

## Reproducibilidad

El experimento puede ser reproducido utilizando:

- Archivo `notebook_practica_poi.ipynb`  
- Archivo de entorno `requirements.txt`  
- Semilla fija aplicada a Python, NumPy y PyTorch

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## Resultados

Evaluación en conjunto de test:

- Accuracy: **89.81%**
- F1-score: **87.87%**
- Precision: **84.06%**
- Recall: **92.06%**

Curvas de pérdida, matriz de confusión y análisis de errores están documentados en el notebook.

---

## Estructura del repositorio

```
├── model/
│   └── best_model.pth               # Pesos entrenados del mejor modelo
├── src/
│   ├── modelo.py                    # Clase MultimodalNet
│   └── dataset.py                   # Clase POIDataset
├── notebook_practica_poi.ipynb      # Desarrollo completo del proyecto
├── requirements.txt                 # Entorno reproducible
└── README.md                        # Este archivo
```

---

## Posibles mejoras

- Incorporar embeddings para texto libre (descripciones de los POIs)
- Aplicar técnicas de data augmentation en imágenes
- Ajuste fino de la red CNN (`fine-tuning`)
- Visualización con Grad-CAM para interpretar decisiones del modelo

---

## Autor

Este proyecto fue desarrollado como parte del módulo de Deep Learning del bootcamp de KeepCoding por el Ingeniero Darío F. Tomatis.
