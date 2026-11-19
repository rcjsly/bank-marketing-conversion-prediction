"""HTTP inference service. Run training before starting the server."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from bank_campaign.data import prepare_frame
from bank_campaign.storage import check_storage, create_storage, record_prediction
from bank_campaign.train import ARTIFACT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
REQUESTS = Counter("campaign_http_requests_total", "HTTP requests", ["route", "status"])
LATENCY = Histogram("campaign_http_request_seconds", "HTTP request duration", ["route"])
PREDICTIONS = Counter("campaign_predictions_total", "Persisted predictions", ["decision"])


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: int = Field(ge=18, le=120)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    poutcome: str
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")


class Prediction(BaseModel):
    conversion_probability: float
    threshold: float
    predicted_conversion: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", str(ARTIFACT_PATH)))
    app.state.bundle = joblib.load(path)
    app.state.engine = create_storage()
    logger.info("Model loaded from %s", path)
    try:
        yield
    finally:
        app.state.engine.dispose()


app = FastAPI(
    title="Bank Campaign Conversion Prediction",
    description="Pre-call conversion prediction demo trained on UCI Bank Marketing data.",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    label = route.path if route else "unmatched"
    REQUESTS.labels(label, str(response.status_code)).inc()
    LATENCY.labels(label).observe(time.perf_counter() - start)
    return response


@app.get("/health")
def health(request: Request) -> dict:
    if not hasattr(request.app.state, "bundle") or not hasattr(request.app.state, "engine"):
        raise HTTPException(status_code=503, detail="Model or database not loaded")
    try:
        check_storage(request.app.state.engine)
    except Exception:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "ok", "model_loaded": True, "database_connected": True}


@app.get("/live")
def live() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    frame = pd.DataFrame([customer.model_dump(by_alias=True)])
    features, _ = prepare_frame(frame, require_target=False)
    bundle = request.app.state.bundle
    probability = float(bundle["pipeline"].predict_proba(features)[0, 1])
    threshold = float(bundle["threshold"])
    model_name = bundle.get("model_name") or type(bundle["pipeline"]).__name__
    try:
        record_prediction(request.app.state.engine, model_name, probability, threshold)
    except Exception:
        logger.exception("Prediction audit write failed")
        raise HTTPException(status_code=503, detail="Prediction could not be audited")
    PREDICTIONS.labels(str(probability >= threshold).lower()).inc()
    logger.info("Prediction completed with model=%s", model_name)
    return Prediction(
        conversion_probability=round(probability, 6),
        threshold=threshold,
        predicted_conversion=probability >= threshold,
    )
