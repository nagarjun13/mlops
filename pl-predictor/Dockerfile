# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ src/
COPY models/ models/

# Command to run prediction script by default
CMD ["python", "src/predict.py"]

