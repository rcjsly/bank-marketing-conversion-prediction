"""Download and validate the UCI Bank Marketing dataset."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
NUMERIC = (
    "age", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
)
CATEGORICAL = (
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
)
FEATURES = NUMERIC + CATEGORICAL + ("previously_contacted",)
REQUIRED = set(NUMERIC + CATEGORICAL + ("duration", "y"))


def download_data(destination: Path) -> Path:
    """Extract only the full CSV from the two-level UCI ZIP archive."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATA_URL, timeout=60) as response:
        archive = BytesIO(response.read())
    with ZipFile(archive) as outer:
        with ZipFile(BytesIO(outer.read("bank-additional.zip"))) as inner:
            csv_bytes = inner.read("bank-additional/bank-additional-full.csv")
    # A temporary file avoids leaving a partially written dataset after interruption.
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(csv_bytes)
    temporary.replace(destination)
    return destination


def prepare_frame(frame: pd.DataFrame, *, require_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    required = REQUIRED if require_target else REQUIRED - {"duration", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty input data")
    data = frame.copy()
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="raise")
    # pdays=999 denotes no previous contact, not a meaningful elapsed time.
    data["previously_contacted"] = (data["pdays"] != 999).astype(int)
    data.loc[data["pdays"] == 999, "pdays"] = float("nan")
    for column in CATEGORICAL:
        data[column] = data[column].astype("string").fillna("unknown").astype(str)
    target = None
    if require_target:
        if not data["y"].isin(["yes", "no"]).all():
            raise ValueError("Target y must contain only yes/no")
        target = data["y"].eq("yes").astype(int)
    # duration is intentionally excluded: it is only known after the call ends.
    return data.loc[:, FEATURES], target


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, sep=";")
    features, target = prepare_frame(frame)
    return features, target
