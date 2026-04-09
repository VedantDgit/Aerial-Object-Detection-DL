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

Automatic build & publish (GitHub Container Registry)

This repository includes a GitHub Actions workflow that builds the Docker image on each push to `main` and publishes it to GitHub Container Registry (GHCR) as `ghcr.io/<your-org-or-username>/aerial-detection:latest`.

How to use the published image on Render (example)
1. In Render, create a new "Docker" Web Service.
2. For the image, set the Registry to "GitHub Container Registry" and enter the image `ghcr.io/<your-username>/aerial-detection:latest`.
3. If Render asks for credentials, provide a GitHub personal access token (with `read:packages`) as the registry password and your GitHub username as the registry username.
4. Set the port to `8501`.

Once Render pulls the image and starts the container, the app will run with the full ML stack (TensorFlow + PyTorch + ultralytics) and both detection and classification will work.

Security note: If you don't want the image to be public, keep the GHCR package scoped to your account or organization and use a deploy service that can authenticate with your token.

Troubleshooting
- If you see compatibility errors, confirm the host's Docker uses a compatible architecture (x86_64). For ARM hosts (e.g., Apple M1/M2), use appropriate wheels or a different base image.
