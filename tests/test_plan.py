import pandas as pd
import pytest
from test_uncertainty import SURVEY, make_survey

from camtraptools import ecology, plan, review, reviews, uncertainty


def chances_for(package):
    deployments = pd.read_csv(package / "deployments.csv", dtype=str)
    return uncertainty.load(package, deployments), ecology.effort_matrix(deployments, 7)


def test_taxon_plan_reviews_doubtful_records_first(tmp_path):
    chances, effort = chances_for(make_survey(tmp_path, SURVEY))

    summary, cells = plan.taxon_plan(chances, "Oryx gazella", effort, 7)

    assert summary["records"] == 3
    assert summary["inDoubt"] > summary["leftInDoubt"]
    assert summary["leftInDoubt"] <= plan.MIN_DOUBT_LEFT
    assert cells["ids"].iloc[0] == ["a2"]
    assert cells["priority"].is_monotonic_decreasing


def test_taxon_plan_is_stable_when_records_are_reviewed(tmp_path):
    package = make_survey(tmp_path, SURVEY[-1:])
    reviews.save_review(package, "c2", "animal", "Vulpes chama", 1, "Tester")
    chances, effort = chances_for(package)

    summary, cells = plan.taxon_plan(chances, "Vulpes chama", effort, 7)

    assert (summary["records"], summary["inDoubt"], summary["toReview"]) == (1, 0.0, 0)
    assert cells.empty


def test_build_orders_events_once_with_the_taxa_they_matter_for(tmp_path):
    chances, effort = chances_for(make_survey(tmp_path, SURVEY))

    species, events = plan.build(chances, effort)

    assert set(species.index) == {"Oryx gazella", "Vulpes chama"}
    assert events["eventID"].is_unique
    assert events["order"].tolist() == list(range(1, len(events) + 1))
    assert events.loc[events["eventID"] == "a2", "taxa"].item() == "Oryx gazella"
    assert species.loc["Oryx gazella", "toReview"] >= 1


def test_build_lists_species_the_ai_probably_missed(tmp_path):
    survey = SURVEY + [(f"x{number}", "B", 5 + number, "blank", None, 0.9) for number in range(6)]
    package = make_survey(tmp_path, survey)
    for event_id in ["c1", "x0", "x1"]:
        reviews.save_review(package, event_id, "animal", "Sylvicapra grimmia", 1, "Tester")
    chances, effort = chances_for(package)

    species, _ = plan.build(chances, effort)

    assert species.loc["Sylvicapra grimmia", "expectedMissedSites"] >= plan.MISSED_SITES_TO_LIST


def test_run_writes_the_plan_files(tmp_path):
    package = make_survey(tmp_path, SURVEY)

    _, events, chances = plan.run(package)

    assert pd.read_csv(package / plan.PLAN_FILE)["eventID"].tolist() == events["eventID"].tolist()
    assert (package / plan.SPECIES_FILE).exists()
    assert chances.reviewed == 0


def test_review_app_plan_order_puts_planned_events_first(tmp_path):
    package = make_survey(tmp_path, SURVEY)
    _, planned, _ = plan.run(package)
    machine = pd.read_csv(package / "observations.csv", dtype=str)
    events = reviews.order_events(machine[machine["observationLevel"] == "event"], machine, "plan")

    ordered = review.planned_first(package, events)

    assert ordered["eventID"].tolist()[: len(planned)] == planned["eventID"].tolist()
    assert sorted(ordered["eventID"]) == sorted(events["eventID"])


def test_minutes_needed_uses_review_speed(tmp_path):
    package = make_survey(tmp_path, SURVEY)
    assert plan.minutes_needed(package, 10) is None
    pd.DataFrame(
        {"eventID": ["a1", "a2", "b1"], "reviewedAt": ["2024-03-01T10:00:00", "2024-03-01T10:01:00", "2024-03-01T10:02:00"]}
    ).to_csv(package / reviews.REVIEWS_FILE, index=False)

    assert plan.minutes_needed(package, 10) == pytest.approx(10.0)


def test_second_species_of_a_visit_counts_as_a_record(tmp_path):
    package = make_survey(tmp_path, SURVEY)
    reviews.save_review(package, "b1", "animal", "Oryx gazella", 2, "Tester", [("Connochaetes taurinus", 4)])
    chances, effort = chances_for(package)

    summary, _ = plan.taxon_plan(chances, "Connochaetes taurinus", effort, 7)

    assert summary["records"] == 1
    assert summary["expectedFalseSites"] == 0.0


def test_doubt_without_events_worth_reviewing_is_spread_thin(tmp_path):
    chances, effort = chances_for(make_survey(tmp_path, SURVEY))

    thin, cells = plan.taxon_plan(chances, "Oryx gazella", effort, 7, min_gain=0.99)
    planned, _ = plan.taxon_plan(chances, "Oryx gazella", effort, 7)

    assert cells.empty
    assert (thin["toReview"], thin["spreadThin"]) == (0, True)
    assert planned["spreadThin"] is False
