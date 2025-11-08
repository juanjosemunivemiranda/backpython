from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import ctg_routes, user_routes  # importa tus routers
import asyncio

# --- Configuración del ciclo de vida (reemplaza on_event) ---
async def lifespan(app: FastAPI):
    print("🚀 Servidor inicializando...")
    # Aquí podrías lanzar tareas al inicio si quieres (por ejemplo, iniciar alguna simulación global)
    yield
    print("🛑 Servidor apagándose...")

# --- Crear app principal ---
app = FastAPI(
    title="Cardiotógrafo API",
    description="Backend para simulación y monitoreo de trazos CTG",
    version="1.0.0",
    lifespan=lifespan
)

# --- Configuración CORS (opcional pero recomendado) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta esto si vas a limitar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Registrar rutas ---
app.include_router(ctg_routes.router, prefix="/ctg", tags=["CTG"])
# app.include_router(user_routes.router, prefix="/users", tags=["Usuarios"])

# --- Ruta raíz ---
@app.get("/")
async def root():
    return {"message": "Cardiotógrafo API funcionando correctamente ✅"}

# --- Punto de entrada principal ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
