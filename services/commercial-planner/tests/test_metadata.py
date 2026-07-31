from metadata import JobFixture, build_planning_metadata


def _fixture(**overrides):
    base = dict(
        prompt="Compare these two screenshots for layout bugs",
        image_count=2,
        requested_domain="software_visual_qa",
        requested_tasks=["comparison", "ui_analysis"],
        ocr_mode="ensemble",
        ocr_language="en",
        detector_mode="auto",
        requested_labels=["forklift", "helmet"],
    )
    base.update(overrides)
    return JobFixture(**base)


def test_commercial_payload_contains_metadata_only():
    payload = build_planning_metadata(_fixture())
    serialized = payload.model_dump()

    assert set(serialized) == {
        "prompt_classifier_hints",
        "image_count",
        "requested_domain",
        "requested_tasks",
        "ocr_mode",
        "ocr_language",
        "detector_mode",
        "requested_label_count",
    }


def test_requested_labels_never_cross_the_boundary_only_their_count():
    payload = build_planning_metadata(
        _fixture(requested_labels=["forklift", "excavator", "crane"])
    )

    assert payload.requested_label_count == 3
    assert "requested_labels" not in payload.model_dump()
    assert "forklift" not in payload.model_dump_json()
    assert "excavator" not in payload.model_dump_json()


def test_raw_prompt_text_never_crosses_the_boundary():
    payload = build_planning_metadata(
        _fixture(prompt="the secret project codename is falcon")
    )

    assert "falcon" not in payload.model_dump_json()


def test_classifier_hints_are_derived_not_verbatim():
    payload = build_planning_metadata(
        _fixture(prompt="Check PPE compliance and count workers on site")
    )

    assert set(payload.prompt_classifier_hints) == {"safety", "counting"}
