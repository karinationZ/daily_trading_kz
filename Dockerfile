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

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy project files
WORKDIR /app
COPY pyproject.toml poetry.lock Makefile ./
COPY . .

# Install Python dependencies (only main group)
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi

# Use build argument for the make command
ARG MAKE_COMMAND=run_order_block_sweep_bot

# Set start command
CMD ["make", "${MAKE_COMMAND}"]