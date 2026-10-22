# =============================================================================
# Traductor LSA — imagen del backend (que además sirve el frontend estático)
# -----------------------------------------------------------------------------
# Un solo servicio cubre API + WebSocket + frontend, tal como está previsto en
# docs/ARQUITECTURA.md §8: en las capas gratuitas de Render/Railway cada servicio
# ocupa un slot, así que conviene no gastar dos.
#
# Construir y correr localmente:
#     docker build -t traductor-lsa .
#     docker run --rm -p 8000:8000 traductor-lsa
# =============================================================================

FROM python:3.12-slim

# - libglib2.0-0: la requiere MediaPipe en tiempo de ejecución.
# - No hace falta libGL: se usa `opencv-python-headless`, que no abre ventanas.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Las dependencias van primero y en su propia capa: mientras requirements.txt no
# cambie, Docker reutiliza esta capa y no reinstala MediaPipe en cada build
# (que es lo más lento de todo el proceso).
#
# PyTorch se instala ANTES y desde el índice de ruedas CPU: la versión por
# defecto de PyPI arrastra las bibliotecas de CUDA (~2,5 GB) que el contenedor
# nunca usa, porque la BiLSTM infiere de a una ventana por vez en CPU en pocos
# milisegundos. Con torch ya instalado, el `torch>=2.2.0` de requirements.txt
# queda satisfecho y pip no baja la versión con CUDA. En la capa gratuita de
# Render/Railway el tamaño de la imagen pesa en el build y en el arranque en frío.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu     && pip install --no-cache-dir -r backend/requirements.txt

# El modelo de MediaPipe se descarga en tiempo de BUILD para que la imagen quede
# autocontenida: si se bajara al arrancar, cada despliegue dependería de que
# storage.googleapis.com esté accesible desde el entorno de producción.
COPY common/ ./common/
RUN python common/download_model.py

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Usuario sin privilegios: si algo se compromete, no corre como root.
RUN useradd --create-home --uid 1000 lsa && chown -R lsa:lsa /app
USER lsa

EXPOSE 8000

# Orígenes CORS permitidos (separados por coma). Vacío = cualquiera, que sirve
# en local; en el deploy definirlo en el panel de la plataforma con el dominio
# real, p. ej. LSA_CORS_ORIGINS=https://traductor-lsa.onrender.com
ENV LSA_CORS_ORIGINS=""

# Render/Railway inyectan el puerto en $PORT; en local cae a 8000.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
