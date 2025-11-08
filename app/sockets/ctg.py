import random, math, asyncio

rooms = {}   # {"room1": {"state": "normal", "t": 0}}
clients = {} # {"sid1": {"room": "room1", "role": "teacher"}}

def generate_ctg(state, t):
    if state == "normal":
        fhr = 135 + random.randint(-10, 10)
        uc = max(0, math.sin(t / 50) * 50 + random.uniform(0, 5))
    elif state == "suspect":
        fhr = 115 + random.randint(-15, 15)
        uc = max(0, math.sin(t / 30) * 80 + random.uniform(0, 10))
    else:  # pathologic
        fhr = 90 + random.randint(-10, 10)
        uc = max(0, math.sin(t / 20) * 100 + random.uniform(0, 15))
    return {"fhr": fhr, "uc": uc, "t": t, "state": state}


def register_events(sio):
    @sio.on("join_room")
    async def join_room(sid, data):
        room = data.get("room")
        role = data.get("role")
        print("Join Room:", sid)

        if not room or not role:
            await sio.emit("error", {"message": "room y role son requeridos"}, room=sid)
            return

        # Guardar cliente y meterlo en la sala
        clients[sid] = {"room": room, "role": role}
        sio.enter_room(sid, room)

        if room not in rooms:
            rooms[room] = {"state": "normal", "t": 0}

        await sio.emit("room_joined", {
            "room": room,
            "role": role,
            "state": rooms[room]["state"]
        }, room=sid)

    @sio.on("change_state")
    async def change_state(sid, data):
        client = clients.get(sid)
        if not client:
            await sio.emit("error", {"message": "Cliente no registrado"}, room=sid)
            return

        if client["role"] != "teacher":
            await sio.emit("error", {"message": "No autorizado para cambiar estado"}, room=sid)
            return

        room = client["room"]
        new_state = data.get("state")
        if not new_state:
            await sio.emit("error", {"message": "Se requiere un nuevo estado"}, room=sid)
            return

        rooms[room]["state"] = new_state
        await sio.emit("state_changed", {"state": new_state}, room=room)

    @sio.on("disconnect")
    async def disconnect(sid):
        if sid in clients:
            del clients[sid]

    async def data_broadcast():
        while True:
            for room, info in rooms.items():
                info["t"] += 1
                sample = generate_ctg(info["state"], info["t"])
                await sio.emit("new_data", sample, room=room)
            await asyncio.sleep(1)

    # Usar start_background_task en vez de create_task
    sio.start_background_task(data_broadcast)
