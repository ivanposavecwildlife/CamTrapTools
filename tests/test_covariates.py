import numpy as np
import pandas as pd
import pytest

from camtraptools import covariates
from camtraptools.ecology import sigmoid


def test_site_covariates_take_the_first_value_per_location():
    deployments = pd.DataFrame(
        {
            "deploymentID": ["n1", "n2", "s1"],
            "locationName": ["North", "North", "South"],
            "habitat": [None, "forest", "grassland"],
        }
    )

    assert covariates.site_covariates(deployments, ["habitat"])["habitat"].to_dict() == {
        "North": "forest",
        "South": "grassland",
    }
    with pytest.raises(ValueError, match="no column cameraHeight"):
        covariates.site_covariates(deployments, ["cameraHeight"])


def test_design_standardises_numbers_and_codes_levels():
    table = pd.DataFrame({"height": ["40", "50", "60"], "habitat": ["forest", "grassland", "scrub"]})

    matrix, labels = covariates.design(table)

    assert labels == ["intercept", "height", "habitat=grassland", "habitat=scrub"]
    assert matrix[:, 1] == pytest.approx([-1.2247, 0, 1.2247], abs=1e-4)
    assert matrix[:, 2:].tolist() == [[0, 0], [1, 0], [0, 1]]


def test_design_needs_a_value_for_every_site():
    with pytest.raises(ValueError, match="habitat is missing for b"):
        covariates.design(pd.DataFrame({"habitat": ["forest", None]}, index=["a", "b"]))


def simulated_survey(seed=0, sites=120, occasions=10):
    rng = np.random.default_rng(seed)
    habitat = rng.choice(["forest", "grassland"], sites)
    height = rng.normal(50, 15, sites)
    psi = sigmoid(-0.5 + 1.5 * (habitat == "grassland"))
    daily = sigmoid(-2.5 + 0.8 * (height - height.mean()) / height.std())
    occupied = rng.random(sites) < psi
    recorded = (rng.random((sites, occasions)) < 1 - (1 - daily[:, None]) ** 7) & occupied[:, None]
    names, columns = [f"s{i}" for i in range(sites)], [f"o{j}" for j in range(occasions)]
    history = pd.DataFrame(recorded.astype(float), index=names, columns=columns).astype("Int64")
    effort = pd.DataFrame(7.0, index=names, columns=columns)
    table = pd.DataFrame({"habitat": habitat, "cameraHeight": height.round(1).astype(str)}, index=names)
    return history, effort, table


def test_compare_models_finds_the_true_covariates():
    history, effort, table = simulated_survey()

    models, coefficients = covariates.compare_models(history, effort, table, ["habitat"], ["cameraHeight"])

    assert models["model"].tolist()[0] == "ψ(habitat) p(cameraHeight)"
    assert models["deltaAIC"].iloc[0] == 0
    assert len(models) == 4
    best = coefficients[coefficients["model"] == "ψ(habitat) p(cameraHeight)"].set_index("term")
    assert best.loc["habitat=grassland", "lower"] < 1.5 < best.loc["habitat=grassland", "upper"]
    assert best.loc["cameraHeight", "lower"] < 0.8 < best.loc["cameraHeight", "upper"]


def test_level_occupancy_converts_coefficients():
    coefficients = pd.DataFrame(
        {
            "model": ["m"] * 3,
            "part": ["occupancy", "occupancy", "detection"],
            "term": ["intercept", "habitat=grassland", "intercept"],
            "estimate": [0.0, np.log(3), -2.0],
        }
    )

    assert covariates.level_occupancy(coefficients, "m") == {"baseline": 0.5, "habitat=grassland": 0.75}


def test_run_writes_every_model(tmp_path):
    history, effort, table = simulated_survey(seed=1)
    deployments = table.rename_axis("deploymentID").reset_index()

    summary = covariates.run({"Vulpes chama": history}, effort, deployments, ["habitat"], [], tmp_path)

    assert summary.loc["Vulpes chama", "bestModel"] == "ψ(habitat) p(.)"
    assert summary.loc["Vulpes chama", "nullDeltaAIC"] > 2
    assert set(summary.loc["Vulpes chama", "occupancyByLevel"]) == {"baseline", "habitat=grassland"}
    saved = pd.read_csv(tmp_path / covariates.MODELS_FILE)
    assert set(saved["model"]) == {"ψ(.) p(.)", "ψ(habitat) p(.)"}
