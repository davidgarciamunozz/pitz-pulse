FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/pitz-pulse

RUN groupadd --gid 10001 pitz && \
    useradd --uid 10001 --gid pitz --create-home pitz && \
    mkdir /data && chown pitz:pitz /data

COPY pyproject.toml ./
COPY app ./app
RUN python -m pip install --no-cache-dir .

COPY prompts ./prompts
COPY migrations ./migrations

USER pitz
EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
