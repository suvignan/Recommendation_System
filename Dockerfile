# The recommendation API (Day 16 Step 6). Build: docker build -t recsys-api .
# The ORDER is the lesson: Docker caches each step, so the rarely-changing steps come first.

# 1. A small Python
FROM python:3.13-slim

WORKDIR /app

# 2. Requirements FIRST: they change rarely, so a code change doesn't reinstall them (this step stays cached)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 3. The code. --no-deps: install ONLY our package (the API imports recsys/utils/shared.py), not the pipeline's
#    dependencies listed in requirements.txt
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .
COPY app/ app/

# 4. The serving files (the .npy arrays + manifest, from `dvc pull`), and where the app finds them
COPY artifacts/recommendation_table/serving/ serving/
ENV SERVING_DIR=/app/serving

# 5. Never run as root: if the app were ever broken into, the attacker isn't root inside the container
RUN useradd --create-home appuser
USER appuser

# 6. One worker (each worker loads its own copy of the arrays; memory is the tightest limit on the free tier).
#    Render sets $PORT; 8000 when running locally. `sh -c` is needed to read $PORT; `exec` then REPLACES the shell
#    with uvicorn, so Docker/Render's "stop" signal reaches uvicorn itself and it shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
