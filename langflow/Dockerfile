# Use the official LangFlow image
FROM langflowai/langflow:latest

# Switch to root to install system dependencies
USER root

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && \
    apt-get install -y tzdata curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt /app

# Upgrade pip globally
RUN pip install --upgrade pip

# Create and activate virtual environment
RUN python -m venv /app/.venv

# Set venv path to be default
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN /app/.venv/bin/pip install --upgrade pip

# Install Python dependencies
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Force upgrade yfinance (to avoid LangFlow conflicts)
RUN /app/.venv/bin/pip install --upgrade yfinance

# Ensure compatible version of websockets
RUN /app/.venv/bin/pip install "websockets>=11.0"

# Copy project files
COPY scripts/ ./scripts/
COPY components/ ./components/
COPY logs/ ./logs/
COPY outputs/ ./outputs/
COPY data/ ./data/
COPY model/ ./model/
