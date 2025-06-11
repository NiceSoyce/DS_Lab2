FROM python:3.12-slim

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
RUN pip install --no-cache-dir numpy scikit-learn pandas

# Verify installation of all Python dependencies
RUN python -c "import numpy; print('Numpy:', numpy.__version__)" && \
    python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)" && \
    python -c "import pandas as pd; print('Pandas:', pd.__version__)"

# Copy the training script into the container
COPY ACP_train_2.py /app/ACP_train_2.py

WORKDIR /app

# Set output directory (can be overridden at runtime)
ENV OUTPUT_DIR=/output

# Create output directory in the container
RUN mkdir -p /output

# Run the training script by default, passing output directory as argument
CMD ["python", "ACP_train_2.py", "--output", "/output/digits_pca_svm.pkl"]
