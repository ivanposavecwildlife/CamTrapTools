from camtraptools.ingest import MEDIA_COLUMNS, build_media, find_images, is_image


def test_is_image_accepts_jpeg_in_any_case(tmp_path):
    assert is_image(tmp_path / "IMG_0001.JPG")
    assert is_image(tmp_path / "IMG_0001.jpg")
    assert is_image(tmp_path / "IMG_0001.jpeg")


def test_is_image_rejects_hidden_and_non_image_files(tmp_path):
    assert not is_image(tmp_path / "._IMG_0001.JPG")
    assert not is_image(tmp_path / ".DS_Store")
    assert not is_image(tmp_path / "notes.txt")


def test_find_images_searches_subfolders(make_image, tmp_path):
    make_image("camera_A/DCIM/100MEDIA/IMG_0001.JPG")
    make_image("camera_A/DCIM/101MEDIA/IMG_0001.JPG")
    (tmp_path / "camera_A/notes.txt").write_text("battery low")

    images = find_images(tmp_path / "camera_A")

    assert [p.relative_to(tmp_path).as_posix() for p in images] == [
        "camera_A/DCIM/100MEDIA/IMG_0001.JPG",
        "camera_A/DCIM/101MEDIA/IMG_0001.JPG",
    ]


def test_build_media_has_one_row_per_image(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    make_image("camera_A/IMG_0002.JPG", date_time_original="2024:05:17 07:00:00")
    make_image("camera_B/IMG_0001.JPG", date_time_original="2024:05:18 22:15:00")

    media = build_media(tmp_path)

    assert list(media.columns) == MEDIA_COLUMNS
    assert list(media["deploymentID"]) == ["camera_A", "camera_A", "camera_B"]
    assert media["mediaID"].is_unique


def test_build_media_skips_loose_files_and_hidden_folders(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG")
    make_image(".Trashes/IMG_0001.JPG")
    (tmp_path / "README.txt").write_text("survey 1")

    media = build_media(tmp_path)

    assert list(media["deploymentID"]) == ["camera_A"]


def test_build_media_keeps_images_without_timestamp(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG")

    media = build_media(tmp_path)

    assert len(media) == 1
    assert media["timestamp"].isna().all()


def test_build_media_on_empty_folder_returns_empty_table(tmp_path):
    media = build_media(tmp_path)

    assert media.empty
    assert list(media.columns) == MEDIA_COLUMNS
