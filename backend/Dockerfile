FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# DigitalOcean App Platform injects PORT; default to 8080 for local Docker.
ENV PORT=8080
EXPOSE 8080

# Bind 0.0.0.0, not 127.0.0.1, or the platform health check never connects.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
