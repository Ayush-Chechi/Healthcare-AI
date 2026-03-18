from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    nb.cells = [
        md(
            "# Healthcare AI — Updated Datasets & Models\n"
            "\n"
            "This notebook documents the upgraded datasets (UCI sources), EDA, and retrained models.\n"
        ),
        md("## 1) Setup"),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import classification_report, roc_auc_score, roc_curve\n"
            "\n"
            "ROOT = Path('.').resolve()\n"
            "sns.set_theme(style='whitegrid')\n"
        ),
        md("## 2) Dataset sources"),
        code(
            "print((ROOT/'data'/'SOURCES.md').read_text(encoding='utf-8'))\n"
        ),
        md("## 3) Load datasets"),
        code(
            "diabetes = pd.read_csv(ROOT/'data'/'diabetes.csv')\n"
            "heart = pd.read_csv(ROOT/'data'/'heart.csv')\n"
            "cancer = pd.read_csv(ROOT/'data'/'breast_cancer_wdbc.csv')\n"
            "\n"
            "display(diabetes.head())\n"
            "display(heart.head())\n"
            "display(cancer.head())\n"
            "\n"
            "print('Shapes:')\n"
            "print('diabetes', diabetes.shape)\n"
            "print('heart   ', heart.shape)\n"
            "print('cancer  ', cancer.shape)\n"
        ),
        md("## 4) Quick EDA (missingness, target balance)"),
        code(
            "def quick_eda(df, target='target'):\n"
            "    print('Missing values:', int(df.isna().sum().sum()))\n"
            "    print('Target balance:')\n"
            "    print(df[target].value_counts(normalize=True).rename('share'))\n"
            "\n"
            "print('--- Diabetes ---')\n"
            "quick_eda(diabetes)\n"
            "print('\\n--- Heart ---')\n"
            "quick_eda(heart)\n"
            "print('\\n--- Cancer ---')\n"
            "quick_eda(cancer)\n"
        ),
        md("## 5) Load trained pipelines + metrics"),
        code(
            "import pickle\n"
            "\n"
            "def load_artifacts(key):\n"
            "    pipe = pickle.load(open(ROOT/'model'/f'{key}_pipeline.pkl','rb'))\n"
            "    metrics = json.loads((ROOT/'model'/f'{key}_metrics.json').read_text(encoding='utf-8'))\n"
            "    return pipe, metrics\n"
            "\n"
            "pipes = {}\n"
            "metrics = {}\n"
            "for k in ['diabetes','heart','cancer']:\n"
            "    pipes[k], metrics[k] = load_artifacts(k)\n"
            "    print(k, metrics[k]['best_model'], metrics[k]['test_metrics'])\n"
        ),
        md("## 6) ROC curves (sanity-check on holdout split used in training script)"),
        code(
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "datasets = {'diabetes': diabetes, 'heart': heart, 'cancer': cancer}\n"
            "\n"
            "plt.figure(figsize=(8,6))\n"
            "for k, df in datasets.items():\n"
            "    y = df['target'].astype(int).to_numpy()\n"
            "    X = df.drop(columns=['target'], errors='ignore')\n"
            "    if k == 'cancer' and 'id' in X.columns:\n"
            "        X = X.drop(columns=['id'])\n"
            "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
            "    prob = pipes[k].predict_proba(X_test)[:,1]\n"
            "    fpr, tpr, _ = roc_curve(y_test, prob)\n"
            "    auc = roc_auc_score(y_test, prob)\n"
            "    plt.plot(fpr, tpr, label=f\"{k} (AUC={auc:.3f})\")\n"
            "plt.plot([0,1],[0,1],'k--',alpha=0.4)\n"
            "plt.xlabel('False Positive Rate')\n"
            "plt.ylabel('True Positive Rate')\n"
            "plt.title('ROC Curves (Holdout)')\n"
            "plt.legend()\n"
            "plt.show()\n"
        ),
        md("## 7) Summary table"),
        code(
            "rows = []\n"
            "for k in ['diabetes','heart','cancer']:\n"
            "    tm = metrics[k]['test_metrics']\n"
            "    rows.append({\n"
            "        'task': k,\n"
            "        'dataset': metrics[k]['dataset'],\n"
            "        'rows': metrics[k]['n_rows'],\n"
            "        'best_model': metrics[k]['best_model'],\n"
            "        **tm,\n"
            "    })\n"
            "summary = pd.DataFrame(rows)\n"
            "display(summary)\n"
        ),
    ]
    return nb


def main() -> None:
    nb = create_notebook()
    out = ROOT / "Healthcare_AI_Analysis.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote notebook: {out}")


if __name__ == "__main__":
    main()

