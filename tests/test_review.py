import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from camtraptools import review
from camtraptools.detect import OBSERVATION_COLUMNS


@pytest.fixture
def package(make_image, tmp_path):
    images = tmp_path / "images"
    make_image("images/camera_A/IMG_0001.JPG")
    make_image("images/camera_A/IMG_0002.JPG")
    make_image("images/camera_B/IMG_0001.JPG")
    (tmp_path / "secret.txt").write_text("not an image")
    package = tmp_path / "package"
    package.mkdir()
    pd.DataFrame(
        {
            "mediaID": ["camera_A/IMG_0001.JPG", "camera_A/IMG_0002.JPG", "camera_B/IMG_0001.JPG"],
            "deploymentID": ["camera_A", "camera_A", "camera_B"],
            "timestamp": ["2024-03-02T06:00:00", "2024-03-02T06:00:01", "2024-03-01T22:00:00"],
            "filePath": ["camera_A/IMG_0001.JPG", "camera_A/IMG_0002.JPG", "camera_B/IMG_0001.JPG"],
            "fileName": ["IMG_0001.JPG", "IMG_0002.JPG", "IMG_0001.JPG"],
        }
    ).to_csv(package / "media.csv", index=False)
    rows = [
        ("o1", "camera_A", "camera_A/IMG_0001.JPG", "camera_A-0001", "2024-03-02T06:00:00", "media", "animal", "Oryx gazella", "0.1", "0.99"),
        ("o2", "camera_A", "camera_A/IMG_0002.JPG", "camera_A-0001", "2024-03-02T06:00:01", "media", "blank", None, None, None),
        ("o3", "camera_B", "camera_B/IMG_0001.JPG", "camera_B-0001", "2024-03-01T22:00:00", "media", "blank", None, None, None),
        ("e1", "camera_A", None, "camera_A-0001", "2024-03-02T06:00:00", "event", "animal", "Oryx gazella", None, "1.0"),
        ("e2", "camera_B", None, "camera_B-0001", "2024-03-01T22:00:00", "event", "blank", None, None, None),
    ]
    columns = ["observationID", "deploymentID", "mediaID", "eventID", "eventStart", "observationLevel",
               "observationType", "scientificName", "bboxX", "classificationProbability"]
    observations = pd.DataFrame(rows, columns=columns).reindex(columns=OBSERVATION_COLUMNS).astype(object)
    observations["eventEnd"] = observations["eventStart"]
    observations.loc[0, ["bboxY", "bboxWidth", "bboxHeight", "count"]] = ["0.2", "0.3", "0.4", "1"]
    observations.loc[3, "count"] = "1"
    observations.to_csv(package / "observations.csv", index=False)
    label = "uuid;mammalia;artiodactyla;bovidae;oryx;gazella;gemsbok"
    results = {"predictions": [{"filepath": "camera_A/IMG_0001.JPG", "detections": [{"processed_classification": {"class": label}}]}]}
    (package / "speciesnet.json").write_text(json.dumps(results), encoding="utf-8")
    return package, images


@pytest.fixture
def client(package):
    return TestClient(review.create_app(*package, reviewer="Ivan Posavec"))


def form_token(page):
    return page.split('name="token" value="')[1].split('"')[0]


def test_start_page_opens_the_least_confident_event(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/events/camera_A-0001"


def test_time_order_starts_with_the_earliest_event(package):
    client = TestClient(review.create_app(*package, order="time"))

    assert client.get("/", follow_redirects=False).headers["location"] == "/events/camera_B-0001"


def test_event_page_shows_decision_boxes_and_navigation(client):
    page = client.get("/events/camera_A-0001").text

    assert "Oryx gazella</strong>" in page
    assert "(gemsbok)" in page
    assert "vote share 100%" in page
    assert 'src="/media/camera_A/IMG_0001.JPG"' in page
    assert "left: 10.0%; top: 20.0%; width: 30.0%; height: 40.0%;" in page
    assert "1 of 2 · least confident first" in page
    assert "AI confidence 0.99" in page
    assert "0 of 2 reviewed" in page
    assert 'id="previous"' not in page
    assert 'href="/events/camera_B-0001" id="next"' in page


def test_blank_event_page(client):
    page = client.get("/events/camera_B-0001").text

    assert "<strong>blank</strong>" in page
    assert 'class="box"' not in page


def test_unknown_event_is_not_found(client):
    assert client.get("/events/camera_Z-0001").status_code == 404


def test_images_of_the_package_are_served(client):
    response = client.get("/media/camera_A/IMG_0001.JPG")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("path", ["/media/../secret.txt", "/media/camera_A/../../secret.txt", "/media/secret.txt"])
def test_only_package_images_are_served(client, path):
    assert client.get(path).status_code == 404


def test_names_are_escaped(package):
    package_dir, images = package
    observations = pd.read_csv(package_dir / "observations.csv", dtype=str)
    observations.loc[observations["observationID"] == "e1", "scientificName"] = "<script>alert(1)</script>"
    observations.to_csv(package_dir / "observations.csv", index=False)

    page = TestClient(review.create_app(package_dir, images)).get("/events/camera_A-0001").text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_create_app_needs_events(package):
    package_dir, images = package
    observations = pd.read_csv(package_dir / "observations.csv", dtype=str)
    observations[observations["observationLevel"] == "media"].to_csv(package_dir / "observations.csv", index=False)

    with pytest.raises(ValueError, match="run `camtraptools events` first"):
        review.create_app(package_dir, images)


def test_event_page_has_review_form_prefilled_with_ai_decision(client):
    page = client.get("/events/camera_A-0001").text

    assert 'action="/events/camera_A-0001/review"' in page
    assert '<option value="animal" selected>' in page
    assert 'name="scientific_name" list="taxa" value="Oryx gazella"' in page
    assert 'name="count" type="number" min="1" value="1"' in page
    assert "Save review as Ivan Posavec" in page
    assert '<option value="Oryx gazella">gemsbok</option>' in page


def test_saving_a_review_moves_to_the_next_event(client):
    first = client.get("/events/camera_B-0001").text
    response = client.post(
        "/events/camera_B-0001/review",
        data={"token": form_token(first), "observation_type": "animal", "scientific_name": "Canis mesomelas", "count": "1"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/events/camera_A-0001"
    page = client.get("/events/camera_B-0001").text
    assert "Reviewed by Ivan Posavec: <strong>Canis mesomelas</strong> · count 1" in page


def test_after_the_last_review_the_event_list_opens(client):
    token = form_token(client.get("/events/camera_A-0001").text)
    for event_id in ("camera_A-0001", "camera_B-0001"):
        response = client.post(
            f"/events/{event_id}/review", data={"token": token, "observation_type": "blank"}, follow_redirects=False
        )

    assert response.headers["location"] == "/events"
    assert client.get("/", follow_redirects=False).headers["location"] == "/events"
    page = client.get("/events").text
    assert "2 of 2 reviewed" in page
    assert "✓ all done" in page


def test_event_list_shows_order_confidence_and_reviews(client):
    token = form_token(client.get("/events/camera_A-0001").text)
    client.post("/events/camera_A-0001/review", data={"token": token, "observation_type": "animal", "scientific_name": "Canis mesomelas", "count": "1"})

    page = client.get("/events").text

    assert page.index("camera_A-0001") < page.index("camera_B-0001")
    assert "<td class=\"num\">0.99</td>" in page
    assert "✓</span> Canis mesomelas" in page
    assert "1 of 2 reviewed" in page


def test_event_page_lists_keyboard_shortcuts(client):
    page = client.get("/events/camera_A-0001").text

    assert "<kbd>B</kbd> blank" in page
    assert 'form.observation_type.value = "blank"' in page


def media_rows(*rows):
    return pd.DataFrame(rows, columns=["observationType", "scientificName", "classificationProbability"])


@pytest.mark.parametrize(
    ("event", "media", "expected"),
    [
        ({"observationType": "unknown", "scientificName": None, "classificationProbability": None}, media_rows(), 0.0),
        (
            {"observationType": "animal", "scientificName": "Oryx gazella", "classificationProbability": "0.8"},
            media_rows(("animal", "Oryx gazella", "0.9"), ("animal", "Oryx gazella", "0.6"), ("animal", "Bovidae", "0.99")),
            0.72,
        ),
        (
            {"observationType": "animal", "scientificName": "Bovidae", "classificationProbability": "1.0"},
            media_rows(("animal", "Bovidae", "0.97")),
            0.5,
        ),
        ({"observationType": "blank", "scientificName": None, "classificationProbability": None}, media_rows(("blank", None, None)), 1.0),
        ({"observationType": "blank", "scientificName": None, "classificationProbability": None}, media_rows(("blank", None, "0.8")), 0.8),
    ],
)
def test_event_confidence(event, media, expected):
    assert review.event_confidence(pd.Series(event), media) == expected


def test_free_port_avoids_a_busy_port():
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]

        assert review.free_port(taken) != taken
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert review.free_port(free) == free


def test_videos_show_their_best_frame_and_a_player(package):
    package_dir, images = package
    media = pd.read_csv(package_dir / "media.csv")
    media.loc[1, ["mediaID", "filePath", "fileName"]] = ["camera_A/CLIP_0002.MP4", "camera_A/CLIP_0002.MP4", "CLIP_0002.MP4"]
    media["fileMediatype"] = ["image/jpeg", "video/mp4", "image/jpeg"]
    media.to_csv(package_dir / "media.csv", index=False)
    observations = pd.read_csv(package_dir / "observations.csv")
    observations.loc[1, "mediaID"] = "camera_A/CLIP_0002.MP4"
    observations.to_csv(package_dir / "observations.csv", index=False)
    (images / "camera_A/CLIP_0002.MP4").write_bytes(b"video")
    frame = package_dir / "frames/camera_A/CLIP_0002.MP4/frame_001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    results = json.loads((package_dir / "speciesnet.json").read_text())
    results["predictions"].append(
        {"filepath": "camera_A/CLIP_0002.MP4", "frame": "frames/camera_A/CLIP_0002.MP4/frame_001.jpg"}
    )
    (package_dir / "speciesnet.json").write_text(json.dumps(results))
    client = TestClient(review.create_app(package_dir, images))

    page = client.get("/events/camera_A-0001").text

    assert '<img src="/frame/camera_A/CLIP_0002.MP4"' in page
    assert '<video controls preload="none" src="/media/camera_A/CLIP_0002.MP4">' in page
    assert client.get("/frame/camera_A/CLIP_0002.MP4").content == b"frame"
    assert client.get("/frame/camera_A/IMG_0001.JPG").status_code == 404


def test_review_without_valid_token_is_refused(client):
    response = client.post("/events/camera_A-0001/review", data={"token": "guess", "observation_type": "blank"})

    assert response.status_code == 403


def test_invalid_review_is_rejected_with_reason(client):
    page = client.get("/events/camera_A-0001").text
    response = client.post(
        "/events/camera_A-0001/review",
        data={"token": form_token(page), "observation_type": "blank", "count": "3"},
    )

    assert response.status_code == 400
    assert "has no count" in response.text


def test_review_of_unknown_event_is_not_found(client):
    page = client.get("/events/camera_A-0001").text
    response = client.post("/events/camera_Z-0001/review", data={"token": form_token(page), "observation_type": "blank"})

    assert response.status_code == 404
