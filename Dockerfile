FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ src/
COPY artifacts/ artifacts/
COPY outputs/results/ outputs/results/
COPY frontend/ frontend/

RUN python -c "import json; from pathlib import Path, PurePosixPath; p=Path('/app/artifacts/lightgbm_deployment_metadata.json'); m=json.loads(p.read_text(encoding='utf-8')); m['model_paths']={sensor: str(PurePosixPath('/app/artifacts') / PurePosixPath(str(path).replace(chr(92), '/')).name) for sensor,path in m['model_paths'].items()}; p.write_text(json.dumps(m,indent=2),encoding='utf-8')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
