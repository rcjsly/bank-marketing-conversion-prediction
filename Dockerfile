FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY artifacts/model.joblib artifacts/model.joblib
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/artifacts/model.joblib

EXPOSE 8000
CMD ["uvicorn", "bank_campaign.api:app", "--host", "0.0.0.0", "--port", "8000"]
