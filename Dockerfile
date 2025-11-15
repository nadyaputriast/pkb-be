# Gunakan base image Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /code

# Copy file requirements dulu dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sisa kode aplikasi kamu (hanya yang diperlukan untuk run)
COPY app.py generator.py ./

# Perintah untuk menjalankan aplikasi Flask dengan Gunicorn
# Hugging Face Spaces mengekspos port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]