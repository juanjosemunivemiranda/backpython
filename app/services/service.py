# service.py
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model("models/ctg_model.keras")

# Ejemplo de predicción con un input simulado (ajusta según tu input shape)
import numpy as np

dummy_input = np.random.rand(1, 100, 2)  # ej: 100 pasos, 2 features (FHR y contracción)
prediction = model.predict(dummy_input)

print("Predicción simulada:", prediction)
