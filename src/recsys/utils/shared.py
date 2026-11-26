"""What BOTH the pipeline and the API need. Only Python built-ins here (no pandas, no numpy, no yaml), so the API
can import this file without installing the pipeline's packages.
"""
import hashlib
import math


def customer_id_to_uint64(customer_id: str) -> int:
    """H&M's 64-hex-character id -> a 64-bit number: '...00ff' -> 255 (the last 16 hex characters, read in base 16).
    10 MB for 1.37M customers instead of 94 MB of text. The pipeline checks no two customers get the same number."""
    return int(customer_id[-16:], 16)


def sha256_of(path) -> str:
    """The file's fingerprint (64 hex characters): change one byte and it changes completely. Read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)          # 1 MB at a time, so a 90 MB file never sits in memory at once
            if not chunk:                        # empty = end of the file
                break
            digest.update(chunk)
    return digest.hexdigest()


def age_bucket_for(age, edges) -> int:
    """ONE age -> its bucket number, exactly like the pipeline's age_to_bucket (proven by tests/test_shared.py).
    edges [16, 20, 25, ...]: 16 <= age < 20 -> 1, 20 <= age < 25 -> 2, ...; missing or outside the edges -> 0."""
    if age is None or math.isnan(age):
        return 0
    for number in range(len(edges) - 1):
        if edges[number] <= age < edges[number + 1]:
            return number + 1
    return 0
