from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

from sklearn.ensemble import RandomForestClassifier
import pickle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


@dataclass(frozen=True)
class TrainSpec:
    key: str
    csv_path: Path
    target_col: str
    drop_cols: tuple[str, ...] = ()


SPECS = [
    TrainSpec("diabetes", DATA_DIR / "diabetes.csv", target_col="target"),
    TrainSpec("heart", DATA_DIR / "heart.csv", target_col="target"),
    TrainSpec("cancer", DATA_DIR / "breast_cancer_wdbc.csv", target_col="target", drop_cols=("id",)),
]


def build_preprocess(df: pd.DataFrame, target_col: str, drop_cols: tuple[str, ...]) -> tuple[ColumnTransformer, list[str], list[str]]:
    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for c in X.columns:
        if is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def input_schema(df: pd.DataFrame, target_col: str, drop_cols: tuple[str, ...]) -> dict:
    """Beginner-friendly schema for Streamlit dynamic input generation."""
    X = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    schema: dict[str, dict] = {}

    for col in X.columns:
        s = X[col]
        if not is_numeric_dtype(s):
            values = sorted([str(v) for v in s.dropna().unique().tolist()])
            schema[col] = {"type": "categorical", "values": values[:50]}
        else:
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(s_num) == 0:
                continue
            lo = float(s_num.quantile(0.01))
            hi = float(s_num.quantile(0.99))
            med = float(s_num.median())
            schema[col] = {"type": "numeric", "min": lo, "max": hi, "default": med}

    return {"features": schema}


def train_one(spec: TrainSpec) -> dict:
    df = pd.read_csv(spec.csv_path)
    y = df[spec.target_col].astype(int).to_numpy()
    X = df.drop(columns=[spec.target_col, *spec.drop_cols], errors="ignore")

    pre, num_cols, cat_cols = build_preprocess(df, spec.target_col, spec.drop_cols)

    # Pick model family and tuning space
    candidates: list[tuple[str, object, dict]] = []

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=500,
            random_state=42,
            eval_metric="logloss",
            n_jobs=0,
        )
        candidates.append(
            (
                "xgboost",
                xgb,
                {
                    "model__max_depth": [3, 4, 5, 6],
                    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                    "model__min_child_weight": [1, 3, 5],
                    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                },
            )
        )

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    candidates.append(
        (
            "random_forest",
            rf,
            {
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best = None
    best_auc = -1.0
    best_name = ""
    best_params = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model, space in candidates:
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        search = RandomizedSearchCV(
            pipe,
            param_distributions=space,
            n_iter=20,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        if float(search.best_score_) > best_auc:
            best_auc = float(search.best_score_)
            best = search.best_estimator_
            best_name = name
            best_params = dict(search.best_params_)

    assert best is not None
    y_prob = best.predict_proba(X_test)[:, 1]
    m = metrics_dict(y_test, y_prob)

    MODEL_DIR.mkdir(exist_ok=True)
    # Save single pipeline per disease (preprocess + model)
    model_path = MODEL_DIR / f"{spec.key}_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best, f)

    schema = input_schema(df, spec.target_col, spec.drop_cols)
    schema_path = MODEL_DIR / f"{spec.key}_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    out = {
        "dataset": spec.csv_path.name,
        "n_rows": int(len(df)),
        "n_features_raw": int(df.drop(columns=[spec.target_col, *spec.drop_cols], errors="ignore").shape[1]),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "best_model": best_name,
        "best_cv_auc": best_auc,
        "best_params": best_params,
        "test_metrics": m,
        "model_path": str(model_path.relative_to(ROOT)),
        "schema_path": str(schema_path.relative_to(ROOT)),
    }

    metrics_path = MODEL_DIR / f"{spec.key}_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    results = []
    for spec in SPECS:
        print(f"\n=== Training: {spec.key} ({spec.csv_path}) ===")
        results.append(train_one(spec))

    print("\nDone. Summary (test metrics):")
    for r in results:
        tm = r["test_metrics"]
        print(
            f"- {r['best_model']:>12} | {r['dataset']:<24} "
            f"AUC={tm['auc_roc']:.3f} Acc={tm['accuracy']:.3f} F1={tm['f1']:.3f}"
        )


if __name__ == "__main__":
    main()

