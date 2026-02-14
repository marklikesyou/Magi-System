FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml .
COPY magi/ magi/
RUN pip install --no-cache-dir ".[openai,google,dspy]"

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/magi /usr/local/bin/magi
COPY --from=builder /app .
ENTRYPOINT ["magi"]
