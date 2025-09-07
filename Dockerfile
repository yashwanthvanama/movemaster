##### Multi-stage Dockerfile for MoveMaster #####
# Base builder stage
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# (Optional) install system build deps if later needed (commented for slimness)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --prefix /install -r requirements.txt

# Runtime stage
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Copy installed python packages
COPY --from=builder /install /usr/local

# Copy source (only what we need at runtime)
COPY src/ src/
# Optionally copy artifacts (if directory exists in build context)
ARG INCLUDE_ARTIFACTS=1
ONBUILD COPY artifacts/ artifacts/
COPY requirements.txt ./

EXPOSE 8000

# Default command uses the service entrypoint module
CMD ["uvicorn", "src.movemaster.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
