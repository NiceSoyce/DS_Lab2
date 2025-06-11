FROM python:3.11-slim

WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir numpy matplotlib scikit-learn

# Copy the training script into the container
COPY ACP_train_2.py /app/ACP_train_2.py

# Use a non-interactive backend for matplotlib
ENV MPLBACKEND=Agg

# Run the training script by default
CMD ["python", "ACP_train_2.py"]
