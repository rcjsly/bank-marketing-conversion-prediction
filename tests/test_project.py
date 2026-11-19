import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from bank_campaign.api import app
from bank_campaign.data import prepare_frame
from bank_campaign.storage import metadata, predictions
from bank_campaign.train import make_pipeline
from sklearn.linear_model import LogisticRegression


def row():
    return {
        "age": 35, "job": "technician", "marital": "single", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 300, "campaign": 2,
        "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
        "euribor3m": 4.857, "nr.employed": 5191.0, "y": "no",
    }


def test_duration_is_excluded_and_999_is_missing():
    features, target = prepare_frame(pd.DataFrame([row()]))
    assert "duration" not in features.columns
    assert features.loc[0, "previously_contacted"] == 0
    assert pd.isna(features.loc[0, "pdays"])
    assert target.iloc[0] == 0


def test_validation_rejects_missing_column():
    invalid = row()
    del invalid["age"]
    with pytest.raises(ValueError, match="age"):
        prepare_frame(pd.DataFrame([invalid]))


def test_pipeline_handles_unseen_category():
    first = row()
    second = {**row(), "age": 54, "job": "admin.", "pdays": 4, "y": "yes"}
    data, target = prepare_frame(pd.DataFrame([first, second]))
    model = make_pipeline(LogisticRegression(max_iter=100))
    model.fit(data, target)
    unseen = data.copy()
    unseen.loc[0, "job"] = "new occupation"
    assert 0 <= model.predict_proba(unseen)[0, 1] <= 1


def test_predict_api_uses_trained_bundle():
    first = row()
    second = {**row(), "age": 54, "job": "admin.", "pdays": 4, "y": "yes"}
    data, target = prepare_frame(pd.DataFrame([first, second]))
    model = make_pipeline(LogisticRegression(max_iter=100))
    model.fit(data, target)
    app.state.bundle = {"pipeline": model, "threshold": 0.5}
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    metadata.create_all(engine)
    app.state.engine = engine
    payload = {key: value for key, value in first.items() if key not in ("duration", "y")}
    # No lifespan: inject a small fitted model so this test does not need network/artifacts.
    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["conversion_probability"] <= 1
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(predictions)) == 1
    assert "campaign_predictions_total" in client.get("/metrics").text
    assert client.post("/predict", json={**payload, "duration": 300}).status_code == 422
    engine.dispose()


def test_transformer_forward_and_unknown_category():
    torch = pytest.importorskip("torch")
    from bank_campaign.transformer import TabularTransformer, TransformerPredictor, make_preprocessor

    first = row()
    second = {**row(), "age": 54, "job": "admin.", "pdays": 4, "y": "yes"}
    data, _ = prepare_frame(pd.DataFrame([first, second]))
    preprocessor = make_preprocessor().fit(data)
    categories = preprocessor.named_transformers_["categorical"].named_steps["encode"].categories_
    model = TabularTransformer(10, [len(values) for values in categories])
    predictor = TransformerPredictor(preprocessor, model)
    unseen = data.copy()
    unseen.loc[0, "job"] = "new occupation"
    assert predictor.predict_proba(unseen).shape == (2, 2)
    assert torch.isfinite(model(torch.ones(2, 10), torch.ones(2, 10, dtype=torch.long))).all()
