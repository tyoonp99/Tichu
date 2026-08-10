FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/gym-tichu:/app

COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .

CMD ["pytest", "-q"]
