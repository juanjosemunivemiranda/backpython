from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from app.controllers.ctg_controller import generate_ctg_data, generate_ctg_from_model, generate_ctg_data_for_simulation, get_diagnosis_from_model, stream_ctg_predictions
from pydantic import BaseModel

router = APIRouter()
# Definir el esquema de los datos que se esperan para el diagnóstico
class CTGTraceRequest(BaseModel):
    ctg_trace: list

@router.websocket("/ws/ctg")
async def websocket_ctg(websocket: WebSocket):
    await websocket.accept()
    t = 0
    try:
        while True:
            data = generate_ctg_data(t)
            await websocket.send_json(data)
            t += 1
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        print("Cliente desconectado ❌")

@router.websocket("/ws/ctg/model")
async def websocket_ctg(websocket: WebSocket):
    await websocket.accept()
    t = 0
    try:
        while True:
            data = generate_ctg_from_model(t)
            await websocket.send_json(data)
            t += 1
            await asyncio.sleep(0.5)  # cada medio segundo manda un dato
    except WebSocketDisconnect:
        print("Cliente desconectado ❌")

@router.websocket("/ws/ctg/simulation")
async def websocket_ctg(websocket: WebSocket):
    await websocket.accept()
    t = 0
    # Estado inicial por defecto, se puede sobrescribir
    simulation_state = "normal"

    try:
        # Esperar el primer mensaje con el estado
        initial_message = await websocket.receive_json()
        if "state" in initial_message:
            simulation_state = initial_message["state"]

        while True:
            # Intentar recibir un nuevo mensaje del cliente sin bloquear el bucle
            try:
                # Usamos asyncio.wait_for para evitar un bloqueo infinito
                # Se espera 0.1 segundos para un nuevo mensaje.
                new_message = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                if "state" in new_message:
                    # Si se recibe un nuevo estado, se actualiza la variable
                    simulation_state = new_message["state"]
                    print(f"Estado de simulación cambiado a: {simulation_state}")
            except asyncio.TimeoutError:
                # No hay mensajes nuevos, el bucle continúa
                pass
            except Exception as e:
                # Manejar otros errores de recepción
                print(f"Error recibiendo mensaje: {e}")
            # Generar y enviar los datos con el estado actual
            data = await stream_ctg_predictions(t, simulation_state)
            await websocket.send_json(data)
            t += 1
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("Cliente desconectado ❌")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# --- Endpoint 2: HTTP POST para el diagnóstico final ---
@router.post("/diagnose")
async def diagnose_ctg(trace_data: CTGTraceRequest):
    diagnosis = get_diagnosis_from_model(trace_data.ctg_trace)
    return diagnosis