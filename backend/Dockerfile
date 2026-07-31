FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY swing_option ./swing_option

ENV PORT=8080 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
EXPOSE 8080

# Bind 0.0.0.0, not 127.0.0.1, or the platform health check never connects.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
