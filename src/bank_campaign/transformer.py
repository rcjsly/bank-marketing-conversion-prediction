"""Optional PyTorch tabular Transformer trained on the same UCI data and split."""

import argparse
import copy
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bank_campaign.data import CATEGORICAL, NUMERIC, download_data, load_data
from bank_campaign.train import DATA_PATH, ROOT, choose_threshold, scores

ARTIFACT_PATH = ROOT / "artifacts" / "transformer.joblib"
REPORT_PATH = ROOT / "artifacts" / "transformer_metrics.json"


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), [*NUMERIC, "previously_contacted"]),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), list(CATEGORICAL)),
    ])


class TabularTransformer(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int], dimension: int = 32):
        super().__init__()
        self.n_numeric = n_numeric
        self.numeric_weight = nn.Parameter(torch.randn(n_numeric, dimension) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, dimension))
        self.embeddings = nn.ModuleList([
            nn.Embedding(size + 1, dimension) for size in cardinalities
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=4, dim_feedforward=64,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        category_tokens = [
            embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        tokens = torch.cat([
            self.cls_token.expand(numeric.shape[0], -1, -1),
            numeric_tokens,
            torch.stack(category_tokens, dim=1),
        ], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(-1)


class TransformerPredictor:
    """Inference wrapper matching the predict_proba interface used by the API."""

    def __init__(self, preprocessor, model: TabularTransformer):
        self.preprocessor = preprocessor
        self.model = model.cpu().eval()

    def predict_proba(self, frame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame)
        n_numeric = self.model.n_numeric
        numeric = torch.tensor(matrix[:, :n_numeric], dtype=torch.float32)
        categorical = torch.tensor(matrix[:, n_numeric:] + 1, dtype=torch.long)
        with torch.no_grad():
            probability = torch.sigmoid(self.model(numeric, categorical)).numpy()
        return np.column_stack([1 - probability, probability])


def train(data_path: Path = DATA_PATH, artifact_path: Path = ARTIFACT_PATH,
          report_path: Path = REPORT_PATH, epochs: int = 12) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not data_path.exists():
        download_data(data_path)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    x, y = load_data(data_path)
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, random_state=42, stratify=y_holdout
    )
    preprocessor = make_preprocessor()
    train_matrix = preprocessor.fit_transform(x_train)
    val_matrix = preprocessor.transform(x_val)
    cardinalities = [
        len(values) for values in
        preprocessor.named_transformers_["categorical"].named_steps["encode"].categories_
    ]
    n_numeric = len(NUMERIC) + 1
    model = TabularTransformer(n_numeric, cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(y_train) - y_train.sum()) / y_train.sum())
    )

    def tensors(matrix):
        return (
            torch.tensor(matrix[:, :n_numeric], dtype=torch.float32),
            torch.tensor(matrix[:, n_numeric:] + 1, dtype=torch.long),
        )

    train_numeric, train_categorical = tensors(train_matrix)
    val_numeric, val_categorical = tensors(val_matrix)
    batches = DataLoader(
        TensorDataset(train_numeric, train_categorical,
                      torch.tensor(y_train.to_numpy(), dtype=torch.float32)),
        batch_size=512, shuffle=True,
    )
    best_ap, best_epoch, best_state, no_improvement = -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for numeric, categorical, labels in batches:
            optimizer.zero_grad()
            loss = loss_function(model(numeric, categorical), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(val_numeric, val_categorical)).numpy()
        ap = average_precision_score(y_val, probabilities)
        if ap > best_ap + 1e-4:
            best_ap, best_epoch = ap, epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= 3:
            break
    model.load_state_dict(best_state)
    predictor = TransformerPredictor(preprocessor, model)
    validation_probability = predictor.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, validation_probability)
    test_probability = predictor.predict_proba(x_test)[:, 1]
    report = {
        "dataset": "UCI Bank Marketing / bank-additional-full.csv",
        "model": "PyTorch tabular Transformer (feature tokens + self-attention)",
        "rows": len(x),
        "split": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "selection_metric": "validation average_precision",
        "best_epoch": best_epoch,
        "trained_epochs": epoch,
        "threshold": round(threshold, 6),
        "validation": scores(y_val, validation_probability, threshold),
        "test": scores(y_test, test_probability, threshold),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(".tmp")
    joblib.dump({"pipeline": predictor, "threshold": threshold,
                 "model_name": "tabular_transformer"}, temporary)
    temporary.replace(artifact_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.artifact, args.report, args.epochs), indent=2))
