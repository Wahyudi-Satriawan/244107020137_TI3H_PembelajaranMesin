# ==============================================================================
# Dockerfile Deployment RumahPoint - Hugging Face Spaces (Port 7860)
# ==============================================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/.local/bin:$PATH"

# Setup User Non-Root sesuai Standar Keamanan Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user

# Set direktori kerja aplikasi
WORKDIR /app

# Salin dependencies list dan instalasi paket Python
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi, template, dan artefak
COPY --chown=user . .

# Expose Port 7860 (Standar Wajib Hugging Face Spaces Docker SDK)
EXPOSE 7860

# Jalankan Web Service dengan Gunicorn WSGI Server Siap Produksi
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
