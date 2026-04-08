# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Production-grade Dockerfile for Nutrisync RL Environment.

FROM python:3.10-slim

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject configuration
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --no-install-project --no-dev

# Copy application code
COPY . /app/

# Install the project itself
RUN uv sync --no-dev

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Start Unified Server (API + UI)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
