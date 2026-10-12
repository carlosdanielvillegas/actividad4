# %% [markdown]
# ## Instalación de dependencias.

# %%
!pip install -U transformers datasets evaluate accelerate peft bitsandbytes sentencepiece huggingface_hub scikit-learn matplotlib pillow


# %%
!pip install numpy==2.0.0

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# %% [markdown]
# ## Configuración de GPU.

# %%
device = 0 if torch.cuda.is_available() else -1

if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Se utilizará CPU")

# %% [markdown]
# ## Autenticación en Hugging Face Hub.

# %% [markdown]
# Se manejo token api enviroment de variables de entorno

# %% [markdown]
# ## DataSet

# %%
dataset = load_dataset("uoft-cs/cifar10")
data_test = dataset["test"].select(range(1000))
imagenes = [row["img"].convert("RGB") for row in data_test]
y_true = np.asarray(data_test["label"], dtype=int)

# %%
set((data_test["label"]))

# %% [markdown]
# ## Modelos

# %%
#Lista de modelos 
modelos = [
     "nateraw/vit-base-patch16-224-cifar10",
    "ahsanjavid/convnext-tiny-finetuned-cifar10",
    "clr4takeoff/convnext-small-cifar10"
]


# %%
def etiquetado(etiqueta):
    """
    Convierte las etiquetas devueltas por los modelos
    en índices de las diez clases de CIFAR-10.
    """
    etiquetas =  [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"]
    etiqueta = str(etiqueta).strip().lower()

    # Algunas tarjetas usan nombres ligeramente distintos.
    alias = {
        "plane": "airplane",
        "airplane": "airplane",
        "car": "automobile",
        "automobile": "automobile",
        "bird": "bird",
        "cat": "cat",
        "deer": "deer",
        "dog": "dog",
        "frog": "frog",
        "horse": "horse",
        "ship": "ship",
        "truck": "truck"
    }

    if etiqueta in alias:
        return etiquetas.index(alias[etiqueta])

    if etiqueta.startswith("label_"):
        class_id = int(etiqueta.split("_")[-1])

        if 0 <= class_id < len(etiquetas):
            return class_id

    raise ValueError(
        f"Etiqueta no reconocida: {etiqueta}. "
        "Verifica id2label en la configuración del modelo."
    )


# %%
def  probar_modelos(modelo, imagenes, y_true):
    try:
        print(f"Ejecutando modelo:{modelo}")
        
        clf = pipeline(
        task="image-classification",
        model=modelo,
        device=device
        )
        predicciones=[]
        fin=0
        for i in range(0, len(imagenes), 16):
            inicio = time.perf_counter()
            imagenes_batch=imagenes[i:i+16]
            outputs = clf(
                imagenes_batch,
                batch_size=16,
                top_k=1
            )
            fin += time.perf_counter() - inicio
            for output in outputs:
                best = output[0]
                label=etiquetado(best["label"])
                predicciones.append(label)
        y_pred = np.asarray(predicciones, dtype=int)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred,average="weighted")
        recall = recall_score( y_true, y_pred,average="weighted")
        f1 = f1_score(y_true, y_pred,average="weighted")
        latencia = fin/1000
        result={"modelo":modelo,
            "y_pred":y_pred,
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1":f1,
            "latencia":latencia}
        return result
    except Exception as e:
        raise e

# %%
resultados=[]
for modelo in modelos:
    resultado=probar_modelos(modelo,imagenes,y_true)
    resultados.append(resultado)

# %% [markdown]
# ## Tabla de resultados

# %%
import pandas as pd
df = pd.DataFrame(resultados)

# %%
df[["modelo",
"accuracy",
"precision",
"recall",
"f1",
"latencia"]].head(5)

# %% [markdown]
# ## Conclusiones

# %% [markdown]
# El modelo que mas convendria en caso que no se necesite una gran presicion seria el modelo clr4takeoff/convnext-small-cifar10	 debido a que tiene menos latencia y un resultado similar al modelo nateraw/vit-base-patch16-224-cifar10.En cuestion de recursos es mas barato y tiempo.En caso que se necesite un modelo mucho más preciso, que se puedan utilizar más recursos computacionales y el tiempo no sea un factor relevante se tendria que utilizar vit-base-patch16-224-cifar10.


