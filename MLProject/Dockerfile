# Gunakan base image Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy semua file dari folder MLProject ke dalam image
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn mlflow dagshub matplotlib seaborn

# Command default saat container dijalankan
CMD ["python", "modelling.py"]