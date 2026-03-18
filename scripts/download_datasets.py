from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    citation: str


SOURCES: dict[str, DatasetSource] = {
    "diabetes": DatasetSource(
        name="Early Stage Diabetes Risk Prediction Dataset",
        url="https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip",
        citation="UCI ML Repository: Early Stage Diabetes Risk Prediction Dataset (DOI: 10.24432/C5VG8H)",
    ),
    "heart": DatasetSource(
        name="Heart Disease (processed datasets: Cleveland/Hungary/Switzerland/VA)",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/",
        citation="UCI ML Repository: Heart Disease",
    ),
    "cancer": DatasetSource(
        name="Breast Cancer Wisconsin (Diagnostic) (WDBC)",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        citation="UCI ML Repository: Breast Cancer Wisconsin (Diagnostic)",
    ),
}


def _download_bytes(url: str) -> bytes:
    with urlopen(url) as resp:
        return resp.read()


def download_diabetes_uci() -> pd.DataFrame:
    """Early Stage Diabetes Risk Prediction -> normalized CSV."""
    raw_zip = _download_bytes(SOURCES["diabetes"].url)
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        # File is known as diabetes_data_upload.csv inside the zip.
        with zf.open("diabetes_data_upload.csv") as f:
            df = pd.read_csv(f)

    # Normalize column names and target
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    # Class column is "class" with values "Positive"/"Negative"
    if "class" in df.columns:
        df["target"] = (df["class"].astype(str).str.lower() == "positive").astype(int)
        df = df.drop(columns=["class"])

    return df


HEART_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def download_heart_uci_combined() -> pd.DataFrame:
    """Combine the 4 processed UCI heart datasets into one DataFrame."""
    base = SOURCES["heart"].url.rstrip("/") + "/"
    files = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data",
    ]

    frames: list[pd.DataFrame] = []
    for fname in files:
        data = _download_bytes(base + fname).decode("utf-8", errors="replace")
        df = pd.read_csv(
            io.StringIO(data),
            header=None,
            names=HEART_COLS,
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.replace("?", pd.NA)

    # Convert numeric columns
    for col in HEART_COLS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Binary target: 0 = no disease, 1 = any disease (1-4)
    combined["target"] = (combined["num"].fillna(0) > 0).astype(int)
    combined = combined.drop(columns=["num"])
    return combined


def download_cancer_wdbc() -> pd.DataFrame:
    raw = _download_bytes(SOURCES["cancer"].url).decode("utf-8", errors="replace")
    # wdbc.data has no header:
    # ID, diagnosis (M/B), then 30 features
    df = pd.read_csv(io.StringIO(raw), header=None)
    df = df.rename(columns={0: "id", 1: "diagnosis"})
    feature_cols = [f"f_{i:02d}" for i in range(30)]
    df.columns = ["id", "diagnosis", *feature_cols]

    df["target"] = (df["diagnosis"].astype(str).str.upper() == "M").astype(int)
    df = df.drop(columns=["diagnosis"])
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    diabetes = download_diabetes_uci()
    diabetes.to_csv(DATA_DIR / "diabetes.csv", index=False)

    heart = download_heart_uci_combined()
    heart.to_csv(DATA_DIR / "heart.csv", index=False)

    cancer = download_cancer_wdbc()
    cancer.to_csv(DATA_DIR / "breast_cancer_wdbc.csv", index=False)

    sources_md = "\n".join(
        [
            f"- **{key}**: {src.name} — `{src.url}` ({src.citation})"
            for key, src in SOURCES.items()
        ]
    )
    (DATA_DIR / "SOURCES.md").write_text(sources_md + "\n", encoding="utf-8")

    print("Downloaded and normalized datasets:")
    print(f"- {DATA_DIR / 'diabetes.csv'}  ({len(diabetes)} rows, {diabetes.shape[1]} cols)")
    print(f"- {DATA_DIR / 'heart.csv'}     ({len(heart)} rows, {heart.shape[1]} cols)")
    print(f"- {DATA_DIR / 'breast_cancer_wdbc.csv'} ({len(cancer)} rows, {cancer.shape[1]} cols)")
    print(f"- {DATA_DIR / 'SOURCES.md'}")


if __name__ == "__main__":
    main()

