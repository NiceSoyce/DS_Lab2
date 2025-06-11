FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib and pandas (fonts, libglib, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        fonts-dejavu-core \
        && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir numpy matplotlib scikit-learn pandas

# Copy the training script into the container
COPY ACP_train_2.py /app/ACP_train_2.py

# Use a non-interactive backend for matplotlib
ENV MPLBACKEND=Agg

# Run the training script by default
CMD ["python", "ACP_train_2.py"]
