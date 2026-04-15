FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY magi/ magi/
RUN python -m pip install --no-cache-dir uv \
    && uv sync --locked --no-dev --extra openai --extra google --extra dspy

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app .
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 magi \
    && mkdir -p /app/magi/storage \
    && chown -R magi:magi /app
USER magi
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -m magi --help >/dev/null || exit 1
ENTRYPOINT ["magi"]
