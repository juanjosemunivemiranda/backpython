import socketio
from fastapi import FastAPI
import uvicorn
import random, math, asyncio
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# === Carga modelo y scaler ===
# model = load_model("model.keras")
# scaler = joblib.load("scaler.pkl")

# === Configuración SocketIO ===
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app = FastAPI()
app.mount("/", socketio.ASGIApp(sio, other_asgi_app=app))

# === Estado global ===
rooms = {}   # {"room1": {"state": "normal", "t": 0}}
rooms_data = {}
clients = {} # {"sid": {"room": "room1", "role": "teacher"}}

# === Generador de features ===
def generate_ctg_features(t: int, state: str = "normal"):
    if state == "normal":
        baseline = random.uniform(120, 150)
        contraction_val = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
        return np.array([[
            baseline,
            random.uniform(0.01, 0.05),
            random.uniform(0.05, 0.2),
            contraction_val,
            0.0, 0.0, 0.0,
            random.uniform(50, 70),
            random.uniform(0.8, 1.5),
            random.uniform(15, 25),
            random.uniform(0.4, 0.8),
            random.uniform(70, 90),
            random.uniform(100, 120),
            random.uniform(150, 160),
            random.uniform(5, 15),
            0, baseline, baseline - 5, baseline - 2,
            random.uniform(5, 15), 0
        ]])
    elif state == "suspect":
        baseline = random.uniform(110, 130)
        contraction_val = max(0, math.sin(t / 30) * 80 + random.uniform(0, 10))
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
    else:  # pathologic
        baseline = random.uniform(100, 120)
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

# def generate_ctg_features_new(t: int, state: str, config: dict):

#     def get_range_value(key, default_low, default_high):
#         """Devuelve un número aleatorio dentro del rango definido en config o por defecto."""
#         if key in config and isinstance(config[key], list):
#             low = config[key][0]
#             high = 0 if len(config[key]) == 1 else config[key][1]
#             # low, high = config[key]
#         else:
#             low, high = default_low, default_high
#         return random.uniform(low, high)

#     contraction_val = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
#     if state == "normal":
#         baseline = random.uniform(120, 150)
#         contraction_val = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
#         return np.array([[
#             baseline,
#             random.uniform(0.01, 0.05),
#             random.uniform(0.05, 0.2),
#             contraction_val,
#             0.0, 0.0, 0.0,
#             random.uniform(50, 70),
#             random.uniform(0.8, 1.5),
#             random.uniform(15, 25),
#             random.uniform(0.4, 0.8),
#             random.uniform(70, 90),
#             random.uniform(100, 120),
#             random.uniform(150, 160),
#             random.uniform(5, 15),
#             0, baseline, baseline - 5, baseline - 2,
#             random.uniform(5, 15), 0
#         ]])
#     elif state == "suspect":
#         baseline = random.uniform(110, 130)
#         contraction_val = max(0, math.sin(t / 30) * 80 + random.uniform(0, 10))
#         return np.array([[
#             baseline, random.uniform(0.001, 0.01), random.uniform(0.05, 0.2),
#             contraction_val, random.uniform(0.01, 0.05),
#             0.0, 0.0,
#             random.uniform(55, 65), random.uniform(0.8, 1.2),
#             random.uniform(18, 25), random.uniform(0.4, 0.6),
#             random.uniform(75, 90), random.uniform(90, 110),
#             random.uniform(150, 165), random.uniform(5, 12),
#             0, baseline, baseline - 5, baseline - 2,
#             random.uniform(10, 20), 0
#         ]])
#     elif state == "pathologic":
#         baseline = random.uniform(100, 120)
#         contraction_val = max(0, math.sin(t / 20) * 100 + random.uniform(0, 15))
#         return np.array([[
#             baseline, random.uniform(0.0, 0.005), random.uniform(0.0, 0.05),
#             contraction_val, random.uniform(0.05, 0.1),
#             random.uniform(0.01, 0.05), random.uniform(0.01, 0.05),
#             random.uniform(30, 50), random.uniform(0.3, 0.7),
#             random.uniform(10, 15), random.uniform(0.2, 0.5),
#             random.uniform(50, 70), random.uniform(80, 100),
#             random.uniform(130, 150), random.uniform(0, 5),
#             0, baseline, baseline - 5, baseline - 2,
#             random.uniform(20, 40), 0
#         ]])
#     else:
#         baseline = get_range_value("LB", 120, 150)

#         if baseline >= 120 and baseline <= 150:
#             AC = get_range_value("AC", 0.02, 0.05)
#         else:
#             AC = get_range_value("AC", 0, 0.02)

#         FM = AC * get_range_value("FM", 2.5, 4)
#         UC = get_range_value("UC", 1, 5)

#         if UC > 2:
#             DP = get_range_value("DP", 5, 30)
#             DS = get_range_value("DS", 5, 60)
#             DL = get_range_value("DL", 0, 10)
#         else:
#             DP = DS = DL = 0

#         if baseline >= 160:
#             MSTV = get_range_value("MSTV", 0.4, 0.8)
#             ASTV = get_range_value("ASTV", 30, 50)
#         else:
#             MSTV = get_range_value("MSTV", 0.8, 1.5)
#             ASTV = get_range_value("ASTV", 50, 70)

#         ALTV = MSTV * get_range_value("ALTV", 10, 15)
#         MLTV = MSTV * get_range_value("MLTV", 0.3, 0.6)

#         Variance = (MSTV + MLTV) * get_range_value("Variance", 3, 7)

#         return np.array([[
#             baseline,
#             AC,
#             FM,
#             contraction_val,
#             UC,
#             DL,
#             DS,
#             DP,
#             ASTV,
#             MSTV,
#             ALTV,
#             MLTV,
#             get_range_value("Width", 70, 90),
#             baseline - 20,     # Min
#             baseline + 30,     # Max
#             get_range_value("Nmax", 5, 15),
#             0,                 # Nzeros
#             baseline,          # Mode
#             baseline - 5,      # Mean
#             baseline - 2,      # Median
#             Variance,
#             0                  # Tendency
#         ]])


def generate_ctg_features_new(t: int, state: str, config: dict):

    def get_range_value(key, default_low, default_high):
        """Devuelve un número aleatorio dentro del rango definido en config o por defecto."""
        if key in config and isinstance(config[key], list):
            low = config[key][0]
            high = 0 if len(config[key]) == 1 else config[key][1]
        else:
            low, high = default_low, default_high
        return random.uniform(low, high)

    # ---------------------------------------------------
    # CONTRACTION WAVES (USADO EN TODOS LOS ESTADOS)
    # ---------------------------------------------------
    contraction_val = max(0, 12 * math.sin(t / 25) + random.uniform(-1, 2))
    has_deceleration = contraction_val > 7

    # ===================================================
    # ESTADO NORMAL (corregido)
    # ===================================================
    if state == "normal":
        baseline = random.uniform(120, 150)

        # Movimiento fetal leve → AC pequeñas
        fetal_movement = max(0, np.random.normal(0.08, 0.03))
        has_acc = fetal_movement > 0.1
        acceleration = random.uniform(10, 25) if has_acc else 0

        # Desaceleración suave solo si hay contracción
        deceleration = random.uniform(5, 15) if has_deceleration else 0

        fhr = baseline + acceleration - deceleration
        fhr = np.clip(fhr, 100, 170)

        astv = random.uniform(55, 70)
        mstv = random.uniform(0.8, 1.4)

        altv = mstv * random.uniform(10, 15)
        mltv = mstv * random.uniform(0.4, 0.8)

        fhr_min = fhr - random.uniform(5, 12)
        fhr_max = fhr + random.uniform(5, 18)
        width = fhr_max - fhr_min

        return np.array([[
            fhr,                            # baseline dinámico
            acceleration / 100,             # AC normalizado
            fetal_movement,                 # FM
            contraction_val,                # UC
            # 0.0,                            # UC-derived (no aplica en normal)
            0.0, 0.0, 0.0,                   # DL DS DP
            astv, mstv,                     # ASTV, MSTV
            altv, mltv,                     # ALTV, MLTV
            width, fhr_min, fhr_max,        # Width, Min, Max
            random.uniform(5, 15),          # Nmax
            0,                              # Nzeros
            baseline,                       # Mode
            baseline - 5,                   # Mean
            baseline - 2,                   # Median
            random.uniform(5, 12),          # Variance
            0                               # Tendency
        ]])

    # ===================================================
    # ESTADO SUSPECT (corregido)
    # ===================================================
    elif state == "suspect":
        baseline = random.uniform(110, 130)

        fetal_movement = max(0, np.random.normal(0.03, 0.02))
        has_acc = fetal_movement > 0.05
        acceleration = random.uniform(5, 15) if has_acc else 0

        deceleration = random.uniform(10, 25) if has_deceleration else 0

        fhr = baseline + acceleration - deceleration
        fhr = np.clip(fhr, 90, 170)

        astv = random.uniform(45, 60)
        mstv = random.uniform(0.6, 1.1)

        altv = mstv * random.uniform(10, 15)
        mltv = mstv * random.uniform(0.3, 0.6)

        fhr_min = fhr - random.uniform(10, 20)
        fhr_max = fhr + random.uniform(10, 25)
        width = fhr_max - fhr_min

        return np.array([[
            fhr,
            acceleration / 100,
            fetal_movement,
            contraction_val,
            random.uniform(0.01, 0.05),
            0.0, 0.0, 0.0,
            astv, mstv,
            altv, mltv,
            width, fhr_min, fhr_max,
            random.uniform(5, 12),
            0,
            baseline,
            baseline - 5,
            baseline - 2,
            random.uniform(12, 22),
            0
        ]])

    # ===================================================
    # ESTADO PATHOLOGIC (corregido)
    # ===================================================
    elif state == "pathologic":
        baseline = random.uniform(100, 120)

        # muy poco movimiento fetal
        fetal_movement = max(0, np.random.normal(0.01, 0.01))
        acceleration = 0  # no AC en patológico

        # desaceleraciones profundas
        deceleration = random.uniform(15, 40) if has_deceleration else 0

        fhr = baseline + acceleration - deceleration
        fhr = np.clip(fhr, 70, 150)

        astv = random.uniform(20, 40)
        mstv = random.uniform(0.3, 0.7)

        altv = mstv * random.uniform(10, 15)
        mltv = mstv * random.uniform(0.2, 0.4)

        fhr_min = fhr - random.uniform(15, 30)
        fhr_max = fhr + random.uniform(5, 10)
        width = fhr_max - fhr_min

        return np.array([[
            fhr,
            0,
            fetal_movement,
            contraction_val,
            random.uniform(0.05, 0.1),
            random.uniform(0.01, 0.05),
            random.uniform(0.01, 0.05),
            random.uniform(0, 5),
            astv, mstv,
            altv, mltv,
            width, fhr_min, fhr_max,
            random.uniform(0, 5),
            0,
            baseline,
            baseline - 5,
            baseline - 2,
            random.uniform(20, 40),
            0
        ]])

    # ===================================================
    # MODO CONFIGURABLE (tu else original → corregido)
    # ===================================================
    else:
        baseline = get_range_value("LB", 120, 150)

        fetal_movement = get_range_value("FM", 0.05, 0.2)
        acceleration = fetal_movement * get_range_value("AC", 10, 25)

        deceleration = get_range_value("DP", 0, 20) if has_deceleration else 0

        fhr = baseline + acceleration - deceleration
        fhr = np.clip(fhr, 80, 180)

        MSTV = get_range_value("MSTV", 0.8, 1.5)
        ASTV = get_range_value("ASTV", 50, 70)
        ALTV = MSTV * get_range_value("ALTV", 10, 15)
        MLTV = MSTV * get_range_value("MLTV", 0.3, 0.6)

        fhr_min = fhr - random.uniform(10, 20)
        fhr_max = fhr + random.uniform(10, 25)
        width = fhr_max - fhr_min

        Variance = (MSTV + MLTV) * get_range_value("Variance", 3, 7)

        return np.array([[ 
            fhr,
            acceleration / 100,
            fetal_movement,
            contraction_val,
            get_range_value("UC", 1, 5),
            get_range_value("DL", 0, 10),
            get_range_value("DS", 0, 60),
            deceleration,
            ASTV,
            MSTV,
            ALTV,
            MLTV,
            width,
            fhr_min,
            fhr_max,
            get_range_value("Nmax", 5, 15),
            0,
            baseline,
            baseline - 5,
            baseline - 2,
            Variance,
            0
        ]])


async def start_room_stream(room_id: str):
    room_info = rooms_data[room_id]
    print(f"🚀 Iniciando transmisión de datos para sala {room_id} con la configuracion {room_info}")

    while rooms_data[room_id]["active"]:
        room_info["t"] += 1
        t = room_info["t"]
        state = room_info["state"]
        config = room_info["room"]["configuration"]
        features = generate_ctg_features_new(t, state, config)
        payload = {
            "time": t,
            "state": state,
            "features": features.tolist()[0]
        }

        await sio.emit("new_data", payload, room=room_id)
        await asyncio.sleep(1)

# === Tarea de emisión continua ===
async def data_broadcast():
    print("🚀 Iniciando transmisión de datos...")
    while True:
        if rooms:
            for room, info in rooms.items():
                info["t"] += 1
                t = info["t"]
                state = info["state"]

                features = generate_ctg_features(t, state)
                # features_scaled = scaler.transform(features)
                # pred = model.predict(features_scaled, verbose=0)
                # pred_class = int(np.argmax(pred, axis=1)[0])
                # pred_probs = pred.tolist()[0]

                payload = {
                    "time": t,
                    "state": state,
                    # "prediction": {
                    #     "class": pred_class,
                    #     "probabilities": pred_probs
                    # },
                    "features": features.tolist()[0]
                }

                await sio.emit("new_data", payload, room=room)
        await asyncio.sleep(1)

# === Eventos del socket ===
@sio.event
async def connect(sid, environ):
    print(f"✅ Cliente conectado: {sid}")

@sio.event
async def disconnect(sid):
    print(f"❌ Cliente desconectado: {sid}")
    client = clients.pop(sid, None)

    if client:
        room_id = client["roomId"]
        role = client["role"]
        participants = rooms_data[room_id]["participants"]
        rooms_data[room_id]["participants"] = [
            p for p in participants if p["sid"] != sid
        ]

        if not rooms_data[room_id]["participants"]:
            rooms_data[room_id]["active"] = False
            print(f"🛑 Sala {room_id} sin participantes, transmisión detenida.")

@sio.event
async def join_room(sid, data):
    room_data = data.get("room")
    role = data.get("role")
    print(f"Configuracion de la sala {room_data} es {data}")

    if not room_data or not role:
        await sio.emit("error", {"message": "room y role son requeridos"}, to=sid)
        return

    room_id = room_data.get("roomId")
    if not room_id:
        await sio.emit("error", {"message": "roomId no encontrado"}, to=sid)
        return

    # Registrar cliente
    clients[sid] = {"roomId": room_id, "role": role}

    # Guardar la sala si no existe
    if room_id not in rooms_data:
        rooms_data[room_id] = {
            "room": room_data,
            "participants": [],
            "t": 0,
            "state": "normal",
            "active": False,
            "configuration": []
        }

    # Agregar participante
    participants = rooms_data[room_id]["participants"]
    if role not in [p["role"] for p in participants]:
        participants.append({"sid": sid, "role": role})

    await sio.enter_room(sid, room_id)

    print(f"👥 Cliente {sid} ({role}) unido a sala {room_id}")
    await sio.emit("user_joined", { role: role }, room=room_id)

    # ✅ Si los tres roles están presentes, arrancamos la transmisión
    roles_presentes = {p["role"] for p in participants}
    if {"TEACHER", "STUDENT", "OBSERVER"} <= roles_presentes:
        if not rooms_data[room_id]["active"]:
            rooms_data[room_id]["active"] = True
            asyncio.create_task(start_room_stream(room_id))

    await sio.emit("joined_room", {"roomId": room_id, "role": role}, room=room_id)

@sio.event
async def change_state(sid, data):
    room = data.get("room")
    new_state = data.get("state")

    if not room or not new_state:
        await sio.emit("error", {"message": "room y state son requeridos"}, room=room)
        return

    if room in rooms_data:
        rooms_data[room]["state"] = new_state
        print(f"🔁 Estado de {room} cambiado a {new_state}")

        # Notificar a todos los clientes de la sala
        await sio.emit("state_changed", {"room": room, "state": new_state}, room=room)
        print(f"🔁 Estado de {sid} cambiado a {data}")
    else:
        print(f"La sala {room} no existe")
        await sio.emit("error", {"message": f"La sala {room} no existe"}, room=room)

@sio.event
async def change_configuration(sid, data):
    room = data.get("room")
    config = data.get("config")

    if not room or not config:
        await sio.emit("error", { "message": "Room y configuracion son requeridos"}, room=room )
        return

    if room in rooms_data:
        rooms_data[room]["room"]["configuration"] = config
        rooms_data[room]["state"] = "manual_mode"

        print(f"🔁 Configuracion de la sala {room} cambiado a {rooms_data[room]["configuration"]}")
        await sio.emit("config_changed", {"room": room, "config": config}, room=room)
    else:
        print(f"La sala {room} no existe")
        await sio.emit("error", {"message": f"La sala {room} no existe"}, room=room)

# === Iniciar servidor ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
