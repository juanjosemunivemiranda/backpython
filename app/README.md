# Instalacionnes

> Python 3 lastest

## Requedido

### 1. primer paso para correr el modelo, he instalado la libreria de ucimlrepo

https://archive.ics.uci.edu/dataset/193/cardiotocography

```bash
    pip install ucimlrepo
```

### 2. Se instala poetry para inicializar el servicio en un puerto en especifico

```bash
    pip install poetry
```

## Manual de instalación

Sigue estos pasos para preparar el entorno e instalar las dependencias en Windows (PowerShell). Ajusta las rutas si usas otra ubicación.

- **Crear entorno virtual:**

```powershell
python -m venv .venv
```

- **Activar entorno (PowerShell):**

```powershell
& env\Scripts\Activate.ps1
```

- **Actualizar pip, setuptools y wheel:**

```powershell
python.exe -m pip install --upgrade pip setuptools wheel
```

- **Instalar dependencias principales:**

```powershell
pip install fastapi "uvicorn[standard]" python-socketio numpy scikit-learn joblib pydantic
```

- **Instalar TensorFlow (si el proyecto lo requiere):**

```powershell
pip install tensorflow
```

- **Ejecutar el servidor en desarrollo:**

```powershell
uvicorn app.main:app --reload --host 0.0.0.0
```

Consejo: usa rutas relativas (`.venv\Scripts\Activate.ps1`) cuando compartas el proyecto para que otros puedan activarlo sin modificar rutas absolutas.
