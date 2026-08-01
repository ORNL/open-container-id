import uuid
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

from container_id.data.schemas import DuplicateGroup, DuplicateManifest
from container_id.util.hashing import calculate_file_sha256


def compute_phash(image_path: Path) -> imagehash.ImageHash:
    with Image.open(image_path) as img:
        return imagehash.phash(img, hash_size=8)

def build_union_find(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Simple union-find to group connected components."""
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

    # Initialize
    for u, v in edges:
        if u not in parent: parent[u] = u
        if v not in parent: parent[v] = v

    for u, v in edges:
        union(u, v)

    components = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)

    return list(components.values())


def find_exact_duplicates(image_paths: list[Path]) -> list[DuplicateGroup]:
    """Finds exact duplicates by SHA-256."""
    hash_map: dict[str, list[str]] = defaultdict(list)
    for path in image_paths:
        sha256 = calculate_file_sha256(path)
        hash_map[sha256].append(str(path))

    groups = []
    for h, members in hash_map.items():
        if len(members) > 1:
            groups.append(DuplicateGroup(
                group_id=f"exact_{h[:16]}",
                members=members,
                reason="exact_hash"
            ))
    return groups

def find_near_duplicates(image_paths: list[Path], threshold: int = 4) -> list[DuplicateGroup]:
    """Finds near duplicates using perceptual hashing (pHash)."""
    hashes = {}
    for path in image_paths:
        try:
            hashes[str(path)] = compute_phash(path)
        except Exception:  # noqa: BLE001, S110 # Ignore invalid images
            pass

    paths = list(hashes.keys())
    edges = []

    # O(N^2) pairwise comparison. This is fine for moderate datasets.
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] <= threshold:
                edges.append((paths[i], paths[j]))

    components = build_union_find(edges)

    groups = []
    for comp in components:
        if len(comp) > 1:
            groups.append(DuplicateGroup(
                group_id=f"phash_{uuid.uuid4().hex[:8]}",
                members=comp,
                reason="phash_near_duplicate"
            ))

    return groups

def deduplicate_dataset(dataset_id: str, image_paths: list[Path]) -> DuplicateManifest:
    """Finds both exact and near duplicates."""
    groups = find_exact_duplicates(image_paths)
    groups.extend(find_near_duplicates(image_paths))

    return DuplicateManifest(
        dataset_id=dataset_id,
        groups=groups
    )
