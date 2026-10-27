FROM python:3.11-slim

WORKDIR /app

# Системні бібліотеки для роботи Pillow та matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dataset_sasmples.py dataset.py train.py app.py ./
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
