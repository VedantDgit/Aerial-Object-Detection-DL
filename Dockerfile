FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git wget ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy docker-specific requirements and install (includes CPU torch, tensorflow-cpu, ultralytics)
COPY requirements-docker.txt /app/requirements-docker.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements-docker.txt

# Copy project
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
