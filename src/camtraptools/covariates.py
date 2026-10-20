"""Occupancy models with site covariates, compared by AIC.

Occupancy (ψ) and the daily detection probability (q) of each site can depend on
site characteristics from the field sheet, such as habitat or camera height, through
logistic regression: logit ψ = β₀ + β·x and logit q = α₀ + α·z. Numeric covariates
are standardised (mean 0, standard deviation 1), so a coefficient is the change on
the logit scale per standard deviation; categorical ones get one coefficient per
level after the first. Four models are fitted for each species: no covariates,
covariates on occupancy, on detection, and on both. The one with the lowest AIC
(Akaike's information criterion) fits best for its number of parameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from camtraptools.crops import location_of
from camtraptools.ecology import (
    MIN_SITES_FOR_MODEL,
    maximize,
    numerical_derivatives,
    sigmoid,
    site_log_likelihood,
)

MODELS_FILE = "occupancy_covariates.csv"
FEW_SITES = 10
Z_95 = 1.959964


def site_covariates(deployments: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Return the chosen covariates for each site (camera location), from the first
    deployment at the site that has a value."""
    missing = [name for name in names if name not in deployments.columns]
    if missing:
        raise ValueError(f"no column {', '.join(missing)} in deployments; add it to the field sheet")
    table = deployments.assign(site=deployments["deploymentID"].map(location_of(deployments)))
    return table.groupby("site")[names].first()


def design(covariates: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return a design matrix with an intercept and its column names. Numeric
    covariates are standardised; others get a 0/1 column per level after the first."""
    columns, labels = [np.ones(len(covariates))], ["intercept"]
    for name in covariates.columns:
        values = covariates[name]
        numeric = pd.to_numeric(values, errors="coerce")
        if values.isna().any():
            sites = ", ".join(values.index[values.isna()].astype(str))
            raise ValueError(f"{name} is missing for {sites}; every site needs a value")
        if numeric.notna().all():
            spread = numeric.std(ddof=0)
            columns.append(((numeric - numeric.mean()) / (spread if spread > 0 else 1)).to_numpy(float))
            labels.append(name)
        else:
            levels = sorted(values.astype(str).unique())
            for level in levels[1:]:
                columns.append((values.astype(str) == level).to_numpy(float))
                labels.append(f"{name}={level}")
    return np.column_stack(columns), labels


def fit(detections: np.ndarray, days: np.ndarray, occupancy: np.ndarray, detection: np.ndarray, start: np.ndarray) -> dict:
    """Fit an occupancy model with the given design matrices for occupancy and daily
    detection, and return its log-likelihood, AIC, coefficients and their 95% Wald
    intervals (from the curvature of the likelihood)."""
    split = occupancy.shape[1]

    def function(parameters):
        return site_log_likelihood(
            sigmoid(occupancy @ parameters[:split]), sigmoid(detection @ parameters[split:]), detections, days
        )

    estimate = maximize(function, start)
    log_likelihood = function(estimate)
    _, hessian = numerical_derivatives(function, estimate)
    try:
        errors = np.sqrt(np.clip(np.diag(np.linalg.inv(-hessian)), 0, None))
    except np.linalg.LinAlgError:
        errors = np.full(len(estimate), np.nan)
    return {
        "logLik": log_likelihood,
        "parameters": len(estimate),
        "AIC": 2 * len(estimate) - 2 * log_likelihood,
        "estimate": estimate,
        "error": errors,
    }


def compare_models(
    history: pd.DataFrame, effort: pd.DataFrame, covariates: pd.DataFrame, occupancy_names: list[str], detection_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the four models for one species and return the model table (sorted by
    AIC, with ΔAIC) and the coefficients of every model."""
    detections = history.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    days = effort.loc[history.index, history.columns].to_numpy(dtype=float)
    days = np.where(np.nan_to_num(detections) == 1, np.maximum(days, 1), days)
    surveyed = ~np.isnan(detections).all(axis=1)
    detections, days = detections[surveyed], days[surveyed]
    sites = covariates.reindex(history.index[surveyed])

    occupancy_matrix, occupancy_labels = design(sites[occupancy_names]) if occupancy_names else design(sites[[]])
    detection_matrix, detection_labels = design(sites[detection_names]) if detection_names else design(sites[[]])
    constant = np.ones((len(sites), 1))
    variants = {
        "ψ(.) p(.)": (constant, ["intercept"], constant, ["intercept"]),
    }
    if occupancy_names:
        variants[f"ψ({', '.join(occupancy_names)}) p(.)"] = (occupancy_matrix, occupancy_labels, constant, ["intercept"])
    if detection_names:
        variants[f"ψ(.) p({', '.join(detection_names)})"] = (constant, ["intercept"], detection_matrix, detection_labels)
    if occupancy_names and detection_names:
        variants[f"ψ({', '.join(occupancy_names)}) p({', '.join(detection_names)})"] = (
            occupancy_matrix, occupancy_labels, detection_matrix, detection_labels,
        )

    null = fit(detections, days, constant, constant, np.array([0.0, -3.0]))
    rows, coefficients = [], []
    for name, (occupancy, occupancy_columns, detection, detection_columns) in variants.items():
        start = np.zeros(occupancy.shape[1] + detection.shape[1])
        start[0], start[occupancy.shape[1]] = null["estimate"]
        result = null if name == "ψ(.) p(.)" else fit(detections, days, occupancy, detection, start)
        rows.append({"model": name, "parameters": result["parameters"], "logLik": round(result["logLik"], 2),
                     "AIC": round(result["AIC"], 2)})
        labels = [("occupancy", label) for label in occupancy_columns] + [("detection", label) for label in detection_columns]
        for (part, label), value, error in zip(labels, result["estimate"], result["error"], strict=True):
            coefficients.append(
                {
                    "model": name, "part": part, "term": label, "estimate": round(value, 3),
                    "lower": round(value - Z_95 * error, 3), "upper": round(value + Z_95 * error, 3),
                }
            )
    table = pd.DataFrame(rows).sort_values("AIC", ignore_index=True)
    table["deltaAIC"] = (table["AIC"] - table["AIC"].min()).round(2)
    return table, pd.DataFrame(coefficients)


def level_occupancy(coefficients: pd.DataFrame, model: str) -> dict[str, float]:
    """Return occupancy at each level of the categorical covariates of a model (with
    numeric covariates at their mean): the intercept level and each other level."""
    rows = coefficients[(coefficients["model"] == model) & (coefficients["part"] == "occupancy")].set_index("term")
    intercept = rows.loc["intercept", "estimate"]
    levels = {"baseline": round(float(sigmoid(intercept)), 2)}
    for term, row in rows.iterrows():
        if "=" in term:
            levels[term] = round(float(sigmoid(intercept + row["estimate"])), 2)
    return levels


def run(
    histories: dict[str, pd.DataFrame],
    effort: pd.DataFrame,
    deployments: pd.DataFrame,
    occupancy_names: list[str],
    detection_names: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Compare covariate models for every species recorded at enough sites, write all
    models and coefficients to ``occupancy_covariates.csv``, and return one row per
    species with its best model, ΔAIC of the model without covariates and the
    coefficients of the best model."""
    covariates = site_covariates(deployments, list(dict.fromkeys(occupancy_names + detection_names)))
    rows, details = [], []
    for name, history in histories.items():
        if int(history.eq(1).any(axis=1).sum()) < MIN_SITES_FOR_MODEL:
            continue
        table, coefficients = compare_models(history, effort, covariates, occupancy_names, detection_names)
        best = table.iloc[0]["model"]
        null_delta = float(table.loc[table["model"] == "ψ(.) p(.)", "deltaAIC"].iloc[0])
        effects = coefficients[(coefficients["model"] == best) & (coefficients["term"] != "intercept")]
        rows.append(
            {
                "taxon": name,
                "sitesDetected": int(history.eq(1).any(axis=1).sum()),
                "bestModel": best,
                "nullDeltaAIC": null_delta,
                "effects": "; ".join(
                    f"{row.part} {row.term}: {row.estimate:+.2f} ({row.lower:+.2f} to {row.upper:+.2f})"
                    for row in effects.itertuples()
                ),
                "occupancyByLevel": level_occupancy(coefficients, best) if "=" in " ".join(effects["term"]) else None,
            }
        )
        details.append(coefficients.merge(table, on="model").assign(taxon=name))
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["taxon", "model", "parameters", "logLik", "AIC", "deltaAIC", "part", "term", "estimate", "lower", "upper"]
    all_models = pd.concat(details, ignore_index=True)[columns] if details else pd.DataFrame(columns=columns)
    all_models.to_csv(out_dir / MODELS_FILE, index=False)
    columns = ["taxon", "sitesDetected", "bestModel", "nullDeltaAIC", "effects", "occupancyByLevel"]
    return pd.DataFrame(rows, columns=columns).set_index("taxon")
