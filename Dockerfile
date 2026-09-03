FROM python:3.14-slim

WORKDIR /app

COPY --chown=1000:1000 main.py pyproject.toml README.md LICENSE ./
COPY --chown=1000:1000 src ./src

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 1000:1000

EXPOSE 8000

CMD ["python", "main.py"]
