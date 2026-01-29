FROM python:3.11-slim

# Install Chromium and dependencies for Kaleido
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libgbm1 \
    libgdk-pixbuf-2.0-0 \
    libgtk-3-0 \
    libpango-1.0-0 \
    make && \
    rm -rf /var/lib/apt/lists/*
