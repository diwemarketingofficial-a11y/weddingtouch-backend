FROM python:3.11-slim

WORKDIR /app

# Runtime library required by our prebuilt dlib wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Install our prebuilt Linux dlib wheel
COPY wheels/dlib-19.24.6-cp311-cp311-linux_x86_64.whl /tmp/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir /tmp/dlib-19.24.6-cp311-cp311-linux_x86_64.whl && \
    rm /tmp/dlib-19.24.6-cp311-cp311-linux_x86_64.whl

# Install remaining Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]