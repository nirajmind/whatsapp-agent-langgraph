# Use an appropriate base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1
# --- CRITICAL FIX: Set HF_HOME and HF_HUB_OFFLINE for runtime ---
#ENV HF_HOME="/app/hf_cache" 
# Must match download location
#ENV HF_HUB_OFFLINE=1 
# Tells Hugging Face libraries to not hit internet
# --- END CRITICAL FIX ---

# Install system dependencies for building libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/* 

# Copy the dependency management files (lock file and pyproject.toml) first
COPY uv.lock pyproject.toml README.md /app/

# Install the application dependencies
RUN uv sync --frozen --no-cache

# Copy your application code into the container
COPY src/ /app/

# Set the virtual environment environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install the package in editable mode
RUN uv pip install -e .
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz

RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
# Define volumes
VOLUME ["/app/data", "/app/hf_cache"]

# Expose the port
EXPOSE 8080

# Run the FastAPI app using uvicorn
CMD ["/app/.venv/bin/fastapi", "run", "ai_companion/interfaces/whatsapp/webhook_endpoint.py", "--port", "8080", "--host", "0.0.0.0"]
