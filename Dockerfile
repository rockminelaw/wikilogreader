FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WRR_DB_PATH=/data/edits.sqlite

WORKDIR /app
COPY pyproject.toml ./
COPY wikirevert ./wikirevert
RUN pip install .

RUN useradd --create-home --uid 1000 collector && mkdir -p /data && chown collector /data
USER collector
VOLUME ["/data"]

HEALTHCHECK --interval=5m --timeout=30s --start-period=2m \
    CMD python -m wikirevert health || exit 1

CMD ["python", "-m", "wikirevert", "run"]
