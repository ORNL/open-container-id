import hashlib
import random
from collections import defaultdict

from container_id.data.schemas import CanonicalSample


def generate_sample_id(sample: CanonicalSample) -> str:
    """Generates a stable ID from immutable source identity."""
    s = (
        f"{sample.source_dataset_id}\0"
        f"{sample.source_split}\0"
        f"{sample.exported_file_name}\0"
        f"{sample.image_sha256}"
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def build_split_groups(samples: list[CanonicalSample]) -> dict[str, str]:
    """
    Builds a union-find graph connecting samples that share any linkage property
    and returns a mapping from sample_id to a unified split_group_id.
    """
    parent: dict[str, str] = {}

    def find(i: str) -> str:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: str, j: str) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Initialize each sample as its own root
    for s in samples:
        parent[s.sample_id] = s.sample_id

    # Grouping features
    exact_hash_map: dict[str, str] = {}
    near_dup_map: dict[str, str] = {}
    source_family_map: dict[str, str] = {}
    container_number_map: dict[str, str] = {}

    for s in samples:
        sid = s.sample_id

        if s.exact_duplicate_group:
            if s.exact_duplicate_group in exact_hash_map:
                union(sid, exact_hash_map[s.exact_duplicate_group])
            else:
                exact_hash_map[s.exact_duplicate_group] = sid

        if s.near_duplicate_group:
            if s.near_duplicate_group in near_dup_map:
                union(sid, near_dup_map[s.near_duplicate_group])
            else:
                near_dup_map[s.near_duplicate_group] = sid

        if s.source_family_group:
            if s.source_family_group in source_family_map:
                union(sid, source_family_map[s.source_family_group])
            else:
                source_family_map[s.source_family_group] = sid

        if s.normalized_label:
            # We group by normalized container number if one exists
            if s.normalized_label in container_number_map:
                union(sid, container_number_map[s.normalized_label])
            else:
                container_number_map[s.normalized_label] = sid

    # Map sample_id to its ultimate root
    split_group_mapping = {}
    for s in samples:
        root = find(s.sample_id)
        # Using the root sample ID as the split group ID
        split_group_mapping[s.sample_id] = f"group_{root}"

    return split_group_mapping


def assign_canonical_splits(
    samples: list[CanonicalSample],
    seed: int = 6346,
    train_pct: float = 0.8,
    valid_pct: float = 0.1,
) -> list[CanonicalSample]:
    """
    Assigns a canonical split (train, valid, test) to each sample based on its unified group.
    """
    # 1. First build the split groups
    group_mapping = build_split_groups(samples)

    # 2. Group samples together by their newly computed group ID
    grouped_samples: dict[str, list[CanonicalSample]] = defaultdict(list)
    for s in samples:
        s.split_group = group_mapping[s.sample_id]
        grouped_samples[s.split_group].append(s)

    # 3. We want deterministic splitting based on a seed.
    # To keep it deterministic, we sort the groups by their group ID.
    sorted_group_ids = sorted(grouped_samples.keys())

    rng = random.Random(seed)
    # Shuffle the group IDs deterministically
    rng.shuffle(sorted_group_ids)

    # Calculate target capacities based on total number of SAMPLES (not groups)
    total_samples = len(samples)
    train_target = int(total_samples * train_pct)
    valid_target = int(total_samples * valid_pct)

    train_count = 0
    valid_count = 0

    # We will iterate through the shuffled groups and assign them to the splits
    for gid in sorted_group_ids:
        group_size = len(grouped_samples[gid])

        # Decide which split to assign to
        if train_count + group_size <= train_target:
            split = "train"
            train_count += group_size
        elif valid_count + group_size <= valid_target:
            split = "valid"
            valid_count += group_size
        else:
            # If valid is full or adding to valid pushes it way over, it goes to test.
            # Alternatively, if we are short on train, we could force it into train even if over target slightly.
            # A simple greedy approach works best:
            if train_count < train_target:
                split = "train"
                train_count += group_size
            elif valid_count < valid_target:
                split = "valid"
                valid_count += group_size
            else:
                split = "test"

        # Assign split to all samples in the group
        for s in grouped_samples[gid]:
            s.canonical_split = split

    return samples
