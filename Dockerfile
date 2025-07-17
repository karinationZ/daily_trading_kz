FROM python:3.11-slim

# Install Chromium and dependencies for Kaleido
RUN apt-get update && apt-get install -y \
    chromium \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy project files
WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY Makefile ./
COPY . .

# Install Python dependencies
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Set start command
CMD ["make", "run_smc_bot"]