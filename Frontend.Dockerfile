FROM python:3.12-slim
WORKDIR /app

# Build this image from the PROJECT ROOT, not from frontend/:
#   docker build -t hashimn/frontend-monitoring:v7 -f frontend/Dockerfile .

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY frontend/ .
COPY _feast/ ./_feast/

EXPOSE 5000
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
