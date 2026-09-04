FROM python:3.11-slim

WORKDIR /app

# No apt packages needed: every dependency in requirements.txt ships a
# manylinux wheel. The previous build installed build-essential, curl and
# software-properties-common (~300MB) and then tried to clean up
# /var/lib/apt-get/lists/*, which is not a real path — so the apt lists stayed
# in the layer too.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/database \
    && chown -R appuser:appuser /app
USER appuser

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
