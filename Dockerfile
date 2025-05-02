# Alap kép választása
FROM python:3.11-slim

# Munkakönyvtár beállítása
WORKDIR /app

# Rendszerfüggőségek telepítése
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Alkalmazás fájlok másolása
COPY . .

# Port megnyitása
EXPOSE 5000

# Alkalmazás indítása
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]