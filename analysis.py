"""Predict a human day-28 antibody endpoint from baseline RNA and covariates."""
import csv
import gzip
import zipfile

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS, download, plot_style, save_summary

SEED = 42
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207750/"
SOURCES = {
    "g207-tpm.gz": GEO + "suppl/GSE207750_UGA4_D0_tpm.csv.gz",
    "g207-meta.gz": GEO + "matrix/GSE207750_series_matrix.txt.gz",
    "CIVR-HRP_SequencingRuns.tsv": (
        "https://raw.githubusercontent.com/GreshamLab/CIVR_HRP_Day0/"
        "2c817746b8587a9ead06d7376caba0bb0a5faadc/CIVR-HRP_SequencingRuns.tsv"
    ),
    "forst-supp-alt.zip": (
        "https://mdpi-res.com/d_attachment/viruses/viruses-14-02446/"
        "article_deploy/viruses-14-02446-s001.zip"
    ),
}
CLINICAL = ["Age", "BMI_Value", "D0_Baseline_HAI_H1N1", "D0_Baseline_HAI_H3N2",
            "D0_Baseline_HAI_Yamagata", "D0_Baseline_HAI_Victoria"]


def read_inputs(paths):
    with gzip.open(paths["g207-meta.gz"], "rt", encoding="utf-8") as source:
        titles = next(line for line in source if line.startswith("!Sample_title\t"))
        public_ids = {int(label.rsplit(",", 1)[1]) for label in
                      next(csv.reader([titles], delimiter="\t"))[1:]}
    with zipfile.ZipFile(paths["forst-supp-alt.zip"]) as archive:
        with archive.open("Table S1.txt") as source:
            # The header omits the first field: the R row number.
            phenotype = pd.read_csv(source, sep="\t", index_col=0).set_index(
                "Participant ID"
            )
    expression = pd.read_csv(paths["g207-tpm.gz"], index_col=0).astype(np.float32).T
    run_map = pd.read_csv(paths["CIVR-HRP_SequencingRuns.tsv"], sep="\t")
    run_to_participant = run_map.set_index("File.Name")["UGA.Sample.ID"]
    if not run_to_participant.index.is_unique or not phenotype.index.is_unique:
        raise ValueError("Duplicated accession or phenotype identifier")
    expression.index = run_to_participant.loc[expression.index].astype(int).to_numpy()
    if not expression.index.is_unique or set(expression.index) != public_ids:
        raise ValueError("TPM column IDs do not match the GEO participant IDs")
    joined = phenotype.reindex(expression.index)
    endpoint = pd.to_numeric(joined["Average Seroconversion"], errors="coerce")
    keep = endpoint.notna() & (endpoint > 0)
    X = np.log2(expression.loc[keep] + 1)
    y = np.log2(endpoint.loc[keep]).to_numpy()
    clinical = joined.loc[keep, CLINICAL].apply(pd.to_numeric, errors="coerce")
    clinical[CLINICAL[2:]] = np.log2(clinical[CLINICAL[2:]].where(clinical[CLINICAL[2:]] > 0))
    return X, clinical, y


def model_set():
    rf = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=5, max_features=0.6, random_state=SEED, n_jobs=2
    )
    return {
        "Median baseline": ("expression", DummyRegressor(strategy="median")),
        "Clinical ridge": ("clinical", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)
        )),
        "Expression ridge": ("expression", make_pipeline(
            SelectKBest(f_regression, k=30), StandardScaler(), Ridge(alpha=35)
        )),
        "Expression random forest": ("expression", make_pipeline(
            SelectKBest(f_regression, k=30), rf
        )),
    }


def evaluate(X, clinical, y):
    features = {"expression": X, "clinical": clinical}
    development, holdout = train_test_split(np.arange(len(y)), test_size=0.25, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows, holdout_predictions, fitted = [], {}, {}
    for name, (input_kind, estimator) in model_set().items():
        data = features[input_kind]
        for fold, (left, right) in enumerate(cv.split(development), 1):
            train, validation = development[left], development[right]
            prediction = clone(estimator).fit(data.iloc[train], y[train]).predict(data.iloc[validation])
            rows.append({"model": name, "partition": "development_cv", "fold": fold,
                         "mae": mean_absolute_error(y[validation], prediction),
                         "r2": r2_score(y[validation], prediction)})
        fitted[name] = clone(estimator).fit(data.iloc[development], y[development])
        prediction = fitted[name].predict(data.iloc[holdout])
        holdout_predictions[name] = prediction
        rows.append({"model": name, "partition": "holdout", "fold": 0,
                     "mae": mean_absolute_error(y[holdout], prediction),
                     "r2": r2_score(y[holdout], prediction)})
    return pd.DataFrame(rows), development, holdout, holdout_predictions, fitted


def main():
    RESULTS.mkdir(exist_ok=True)
    X, clinical, y = read_inputs({name: download(name, url) for name, url in SOURCES.items()})
    metrics, development, holdout, predictions, models = evaluate(X, clinical, y)
    metrics.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.5f")
    rf = models["Expression random forest"]
    transformed = rf.steps[0][1].transform(X.iloc[holdout])
    names = X.columns[rf.steps[0][1].get_support()]
    values = shap.TreeExplainer(rf.steps[1][1])(transformed).values
    importances = pd.Series(np.abs(values).mean(axis=0), index=names, name="mean_abs_shap")
    importances.sort_values(ascending=False).rename_axis("ensembl_gene").to_csv(
        RESULTS / "rf_shap.csv", float_format="%.6f"
    )
    summary = metrics.loc[metrics.partition.eq("holdout")].sort_values("mae")
    plt = plot_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax.barh(summary.model[::-1], summary.mae[::-1], color="#268479")
    ax.set(xlabel="MAE (log₂ fold-change)", title="Held-out prediction error")
    top = importances.nlargest(8).sort_values()
    bx.barh(top.index, top.values, color="#435e8e")
    bx.set(xlabel="Mean |SHAP| (log₂ fold-change)", title="Random forest · held-out explanations")
    fig.savefig(RESULTS / "prediction_and_shap.png")
    plt.close(fig)
    save_summary({
        "participants": len(y), "development_n": len(development), "holdout_n": len(holdout),
        "genes_in_input": X.shape[1], "outcome": "log2 average HAI seroconversion at day 28",
        "best_holdout_model": summary.iloc[0].model,
        "holdout_mae": {row.model: round(row.mae, 4) for row in summary.itertuples()},
        "top_random_forest_feature": str(importances.idxmax()),
    })


if __name__ == "__main__":
    main()
