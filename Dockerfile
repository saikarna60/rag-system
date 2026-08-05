FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir sentence-transformers numpy fastapi uvicorn pydantic
COPY src/ ./src/
WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
