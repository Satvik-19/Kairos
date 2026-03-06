# ── Stage: production backend image ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies needed to compile native extensions (numpy, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency installation (cached layer) ───────────────────────────────────
# Copy requirements first so Docker only re-runs pip when this file changes
COPY requirements-server.txt .

# Install PyTorch CPU-only FIRST to avoid pulling the 2 GB CUDA build
# The subsequent requirements install sees torch>=2.0 already satisfied and skips it
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining server + ML dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# ── Application source ────────────────────────────────────────────────────────
COPY pyproject.toml .
COPY kairos/ ./kairos/
COPY ml/     ./ml/

# Install the kairos package itself (editable, no-deps — already installed above)
RUN pip install --no-cache-dir -e . --no-deps

# ── Security: run as non-root ─────────────────────────────────────────────────
RUN useradd -m -u 1001 kairos && chown -R kairos:kairos /app
USER kairos

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# PORT env var is honoured at runtime (Render sets it automatically).
# Default 8000 for local docker run.
CMD ["sh", "-c", "uvicorn kairos.server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
