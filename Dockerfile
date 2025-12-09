# Use a slim but modern Python base. Pick one that matches your mac host arch if needed.
FROM python:3.12-slim

# avoid line-buffering issues
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system deps (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    git \
    unzip \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    default-jre \
    ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# If you need torch with a specific wheel use pip install <wheel-url> before requirements
RUN python -m pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the repo
COPY . /app

# Create output folder and give permissions
RUN mkdir -p /app/output /app/data && chmod -R a+rwx /app/output /app/data

# Expose Streamlit default port
EXPOSE 8501

# If your streamlit app path is app/dashboard.py, command below will run it.
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
