**Docker deployment (local & cloud)**

This project includes a `Dockerfile` and `requirements-docker.txt` to run the Streamlit app inside a reproducible Python 3.10 container (CPU-only). Use this when you want the full ML stack (TensorFlow + PyTorch + ultralytics) available.

Local build & run
1. Build the image:

```bash
docker build -t aerial-detection:latest .
```

2. Run the container (forward port 8501):

```bash
docker run --rm -p 8501:8501 aerial-detection:latest
```

Now open http://localhost:8501 to view the app.

Notes
- The Docker image installs CPU wheels for `torch` and `tensorflow-cpu` and may be large. The first build can take several minutes depending on network speed.
- If you deploy to a cloud container provider (Render, Railway, Heroku Container Registry, EC2, etc.), configure the platform to build from this repository and expose port 8501.
- Streamlit Community Cloud does not currently support custom Dockerfiles. Use a container host or Streamlit for Teams with custom images.

Troubleshooting
- If you see compatibility errors, confirm the host's Docker uses a compatible architecture (x86_64). For ARM hosts (e.g., Apple M1/M2), use appropriate wheels or a different base image.
