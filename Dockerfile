FROM python:3.11-slim

# Install Matplotlib dependencies and Chromium for Kaleido
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    make \
    chromium-browser \
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