from container_id.data.schemas import CanonicalSample
from container_id.data.splitting import assign_canonical_splits, build_split_groups


def create_sample(
    sample_id: str,
    exact_group: str | None = None,
    near_group: str | None = None,
    family_group: str | None = None,
    normalized_label: str | None = None,
) -> CanonicalSample:
    return CanonicalSample(
        sample_id=sample_id,
        source_dataset_id="test",
        source_split="train",
        source_image_id=1,
        exported_file_name=f"{sample_id}.jpg",
        original_file_name=f"{sample_id}.jpg",
        source_image_path=f"data/{sample_id}.jpg",
        image_sha256="abc",
        width=100,
        height=100,
        raw_filename_label="",
        normalized_label=normalized_label,
        label_status="no_candidate",
        iso_structure_valid=False,
        check_digit_valid=False,
        owner_prefix=None,
        equipment_category=None,
        serial_number=None,
        check_digit=None,
        annotations=[],
        exact_duplicate_group=exact_group,
        near_duplicate_group=near_group,
        source_family_group=family_group,
    )


def test_build_split_groups() -> None:
    # We want to test transitive grouping.
    # s1 and s2 share exact group
    s1 = create_sample("s1", exact_group="e1")
    s2 = create_sample("s2", exact_group="e1", near_group="n1")

    # s3 and s2 share near group
    s3 = create_sample("s3", near_group="n1", normalized_label="CSQU1234567")

    # s4 and s3 share normalized label
    s4 = create_sample("s4", normalized_label="CSQU1234567")

    # s5 is isolated
    s5 = create_sample("s5")

    groups = build_split_groups([s1, s2, s3, s4, s5])

    # s1, s2, s3, s4 should all share the same root group
    assert groups["s1"] == groups["s2"]
    assert groups["s2"] == groups["s3"]
    assert groups["s3"] == groups["s4"]

    # s5 should be isolated
    assert groups["s5"] != groups["s1"]


def test_assign_canonical_splits() -> None:
    samples = []
    # Create 100 isolated samples
    for i in range(100):
        samples.append(create_sample(f"iso_{i}"))

    # Create 1 large group of 50 samples
    for i in range(50):
        samples.append(create_sample(f"linked_{i}", exact_group="big_group"))

    assigned = assign_canonical_splits(samples, seed=42)

    # Verify every linked sample ended up in the same split
    linked_splits = {
        s.canonical_split for s in assigned if s.sample_id.startswith("linked_")
    }
    assert len(linked_splits) == 1

    # Count splits
    train = sum(1 for s in assigned if s.canonical_split == "train")
    valid = sum(1 for s in assigned if s.canonical_split == "valid")
    test = sum(1 for s in assigned if s.canonical_split == "test")

    # With 150 total samples, targets are train=120, valid=15, test=15.
    # Since there's a chunk of 50 that must stay together, exact balance is impossible,
    # but the greedy algorithm should get close or at least put them in *some* valid state
    # without failing.
    assert train + valid + test == 150
    assert (
        train >= 50
    )  # At least the big group or a bunch of isolated ones went to train
