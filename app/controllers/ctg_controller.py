import math, random
from tensorflow import keras
import numpy as np
from pathlib import Path
import joblib

# Definir la ubicación de los archivos del modelo y el escalador
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ctg_model_data_sheet_selected_features.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler_data_sheet_selected_features.pkl"

model = None
scaler = None

# Cargar el modelo y el escalador una sola vez al inicio del servicio
try:
    print("Cargando modelo y escalador...")
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Modelo y escalador cargados exitosamente.")
except (IOError, ValueError, FileNotFoundError) as e:
    print(f"❌ Error al cargar modelo o escalador: {e}")
    # Es crucial que la aplicación no falle si no se cargan
    # Se puede manejar el error en los endpoints.

def generate_ctg_data(t: int):
    fhr = 120 + random.randint(-10, 20)
    contraction = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
    return {"time": t, "fhr": fhr, "contraction": contraction}

def predict_ctg_data(t: int):
    """
    Genera un punto CTG usando el modelo.
    El input debe coincidir con cómo entrenaste tu modelo.
    """
    # Ejemplo: features = [time_normalizado]
    x = np.array([[t / 1000]])  # Normaliza si entrenaste así
    y_pred = model.predict(x, verbose=0)

    # Supongamos que el modelo devuelve 2 salidas: [fhr, contraction]
    fhr, contraction = y_pred[0]

    return {
        "time": t,
        "fhr": float(fhr),
        "contraction": float(contraction)
    }

def generate_ctg_from_model(t: int):
    # 1. GENERAR VALORES EN ESCALA REAL
    # Se debe generar un array de 21 características, NO 35.
    # Los valores deben estar en el rango típico de cada métrica CTG.
    # A continuación, un ejemplo con datos simulados más realistas:
    # Este array simula un registro real con 21 características
    # incluyendo las métricas de FHR y contracciones en posiciones específicas.
    
    # Simulación de un registro CTG de 21 características.
    # Esto es solo un ejemplo; lo ideal sería obtener datos reales de un sensor.
    fhr_value = np.random.uniform(110, 160)  # Valor normal para FHR
    contraction_value = np.random.uniform(5, 100) # Valor para contracciones
    
    # Crear un array de 21 características (las 19 restantes pueden ser aleatorias para la simulación)
    real_input_21 = np.random.rand(1, 21)
    
    # Asignar los valores simulados a las posiciones correctas (asegúrate de que estas son las posiciones reales)
    # Por ejemplo, si fhr y contraction son las primeras dos características.
    real_input_21[0][0] = fhr_value
    real_input_21[0][1] = contraction_value

    # 2. ESCALAR
    scaled_input = scaler.transform(real_input_21)

    # 3. PREDICCIÓN
    prediction = model.predict(scaled_input, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # 4. VOLVER A ESCALA REAL
    valores_reconstruidos = scaler.inverse_transform(scaled_input)

    return {
        "real_values": {
            "fhr": float(real_input_21[0][0]),
            "contraction": float(real_input_21[0][1])
        },
        "scaled_values": {
            "fhr": float(scaled_input[0][0]),
            "contraction": float(scaled_input[0][1])
        },
        "reconstructed_from_scaled": {
            "fhr": float(valores_reconstruidos[0][0]),
            "contraction": float(valores_reconstruidos[0][1])
        },
        "prediction": int(predicted_class),
        "time": t
    }

# --- New function to generate realistic CTG data ---
def generate_ctg_data_for_simulation(t: int, state: str = "normal"):
    """
    Generates a single, realistic CTG data point based on a specified state.
    This simulates a real-time trace for visualization.
    """
    fhr_base = 0
    fhr_noise = random.randint(-5, 5)
    
    if state == "normal":
        fhr_base = 135
        contraction_val = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
    elif state == "suspect":
        fhr_base = 105  # Slight bradycardia
        fhr_noise = random.randint(-15, 15)
        contraction_val = max(0, math.sin(t / 30) * 80 + random.uniform(0, 10))
    elif state == "pathologic":
        fhr_base = 90  # Clear bradycardia
        fhr_noise = random.randint(-10, 10)
        contraction_val = max(0, math.sin(t / 20) * 100 + random.uniform(0, 15))
    
    fhr_val = fhr_base + fhr_noise
    
    return {"time": t, "fhr": float(fhr_val), "contraction": float(contraction_val)}


# --- 2. Lógica para el diagnóstico final (HTTP POST) ---
def get_diagnosis_from_model(trace_data):
    """
    Procesa la traza completa de datos y usa el modelo para obtener un diagnóstico.
    """
    if model is None or scaler is None:
        return "Error: Modelo o escalador no disponibles."

    # Esta es la parte más compleja. Necesitas extraer las 35 características.
    # Por ahora, se simulará con valores de la traza para tener una demostración funcional.
    # En una aplicación real, esta función contendría una lógica compleja para
    # calcular todas las métricas del dataset CTG (variabilidad, aceleraciones, etc.).

    try:
        suspect_data = np.array([[
            140,  # LB (Moderate Baseline)
            0.002,  # AC.1 (Some Accelerations)
            0.1,  # FM.1 (Slightly Increased Fetal Movement)
            0.05,  # UC.1 (Moderate Uterine Contractions)
            0.05,  # DL.1 (Occasional Late Decelerations)
            0.0,  # DS.1 (No Severe Decelerations)
            0.0,  # DP.1 (No Prolonged Decelerations)
            60,   # ASTV (Moderate Short Term Variability)
            1.0,  # MSTV (Moderate Mean Short Term Variability)
            20,   # ALTV (Moderate Long Term Variability)
            0.5,  # MLTV (Moderate Mean Long Term Variability)
            80,   # Width (Moderate variability)
            100,  # Min (Moderate Minimum FHR)
            160,  # Max (Moderate Maximum FHR)
            10,    # Nmax (Moderate number of accelerations per minute)
            0,    # Nzeros (No zero crossings)
            140,  # Mode (Moderate Mode FHR)
            135,  # Mean (Moderate Mean FHR)
            138,  # Median (Moderate Median FHR)
            10,   # Variance (Moderate Variance)
            0    # Tendency (No tendency)
        ]])


        # Escalar los nuevos datos usando el scaler entrenado
        suspect_data_scaled = scaler.transform(suspect_data)

        # Realizar la predicción
        predictions = model.predict(suspect_data_scaled)

        # La salida del modelo softmax son probabilidades para cada clase (0, 1, 2)
        # Para obtener la clase predicha, se toma el índice de la probabilidad más alta
        predicted_class_index = np.argmax(predictions, axis=1)

        # Mapear el índice a la etiqueta original (1=Normal, 2=Sospechoso, 3=Patológico)
        # asumiendo que 0 -> 1, 1 -> 2, 2 -> 3
        class_mapping = {0: 'Normal', 1: 'Suspecto', 2: 'Patológico'}
        predicted_class_label = [class_mapping[i] for i in predicted_class_index]
        
        return {"prediction": predicted_class_label, "probability": predictions, "class_prediction": predicted_class_index}

    except Exception as e:
        print(f"Error en get_diagnosis_from_model: {e}")
        return {"prediction": "Error en el análisis", "details": str(e)}

def generate_ctg_features(t: int, state: str = "normal"):
    if state == "normal":
        baseline = random.uniform(120, 150)  # Línea basal normal
        contraction_val = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
        return np.array([[
            baseline,
            random.uniform(0.01, 0.05),
            random.uniform(0.05, 0.2),
            contraction_val,
            0.0,
            0.0,
            0.0,
            random.uniform(50, 70),
            random.uniform(0.8, 1.5),
            random.uniform(15, 25),
            random.uniform(0.4, 0.8),
            random.uniform(70, 90),
            random.uniform(100, 120),
            random.uniform(150, 160),
            random.uniform(5, 15),
            0,
            baseline,
            baseline - 5,
            baseline - 2,
            random.uniform(5, 15),
            0
        ]])

    elif state == "suspect":
        baseline = random.uniform(110, 130)  # Línea basal más baja
        contraction_val = max(0, math.sin(t / 30) * 80 + random.uniform(0, 10))  # más fuertes
        return np.array([[
            baseline, random.uniform(0.001, 0.01), random.uniform(0.05, 0.2),
            contraction_val, random.uniform(0.01, 0.05),
            0.0, 0.0,
            random.uniform(55, 65), random.uniform(0.8, 1.2),
            random.uniform(18, 25), random.uniform(0.4, 0.6),
            random.uniform(75, 90), random.uniform(90, 110),
            random.uniform(150, 165), random.uniform(5, 12),
            0, baseline, baseline - 5, baseline - 2,
            random.uniform(10, 20), 0
        ]])

    elif state == "pathologic":
        baseline = random.uniform(100, 120)  # Línea basal patológica
        contraction_val = max(0, math.sin(t / 20) * 100 + random.uniform(0, 15))
        return np.array([[
            baseline, random.uniform(0.0, 0.005), random.uniform(0.0, 0.05),
            contraction_val, random.uniform(0.05, 0.1),
            random.uniform(0.01, 0.05), random.uniform(0.01, 0.05),
            random.uniform(30, 50), random.uniform(0.3, 0.7),
            random.uniform(10, 15), random.uniform(0.2, 0.5),
            random.uniform(50, 70), random.uniform(80, 100),
            random.uniform(130, 150), random.uniform(0, 5),
            0, baseline, baseline - 5, baseline - 2,
            random.uniform(20, 40), 0
        ]])

    else:
        raise ValueError("Estado no reconocido")

async def stream_ctg_predictions(t: int, state: str = "normal"):
    try:
        while True:
            # Generar features simulados
            features = generate_ctg_features(t, state)
            print(features)
            # Predicción con tu modelo Keras
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled, verbose=0)
            pred_class = int(np.argmax(pred, axis=1)[0])
            pred_probs = pred.tolist()[0]  # probabilidades de cada clase

            # Armar payload
            payload = {
                "time": t,
                "features": features.tolist()[0],
                "prediction": {
                    "class": pred_class,          # 0=normal, 1=suspect, 2=pathologic
                    "probabilities": pred_probs   # lista con probs
                }
            }

            # Enviar al front
            return payload
    except Exception as e:
        print("Cliente desconectado ❌", e)