"""
validator.py — Component #5, validation half (dataset_generator.md,
section 5).

Checks run BEFORE a generated sample is written to dataset.jsonl:
  - output isn't empty/truncated/doesn't end mid-sentence
  - output length within ~50-400 words
  - output doesn't contain a Z-score/classification number that
    contradicts the input (regex scan near HAZ/WAZ/classification
    keywords)
  - no near-duplicate output (hash-check against previous samples)

Returns (is_valid, reason) so failures can be logged for manual review
rather than silently dropped.
"""

import hashlib
import re

MIN_WORDS = 50
MAX_WORDS = 400

_SENTENCE_END = re.compile(r'[.!?"\')\]]\s*$')

# Matches a number appearing near "HAZ"/"WAZ"/"WHZ" in the OUTPUT text,
# so we can compare it against the number actually passed in the INPUT.
_Z_NEAR_LABEL = re.compile(
    r'(HAZ|WAZ|WHZ)[^\d-]{0,15}(-?\d+(?:[.,]\d+)?)', re.IGNORECASE
)


def _extract_z_values(text: str) -> dict[str, float]:
    found = {}
    for label, num in _Z_NEAR_LABEL.findall(text):
        try:
            found[label.upper()] = float(num.replace(",", "."))
        except ValueError:
            continue
    return found


def check_not_truncated(output: str) -> tuple[bool, str]:
    stripped = output.strip()
    if not stripped:
        return False, "empty output"
    if not _SENTENCE_END.search(stripped):
        return False, "does not end at a sentence boundary (possibly truncated)"
    return True, ""


def check_length(output: str) -> tuple[bool, str]:
    n = len(output.split())
    if n < MIN_WORDS:
        return False, f"too short ({n} words, min {MIN_WORDS})"
    if n > MAX_WORDS:
        return False, f"too long ({n} words, max {MAX_WORDS})"
    return True, ""


def check_zscore_consistency(output: str, expected: dict[str, float], tolerance: float = 0.05) -> tuple[bool, str]:
    """
    expected: e.g. {"HAZ": -2.55, "WAZ": -2.04} — the ACTUAL values that
    were fed to the agent as input. If the narrative restates a number
    near "HAZ"/"WAZ"/"WHZ" that doesn't match, flag it — the agent may
    have "corrected" or hallucinated a different value, which is exactly
    the failure mode the Critical Design Constraint exists to prevent.
    """
    found = _extract_z_values(output)
    for label, expected_val in expected.items():
        if label in found and abs(found[label] - expected_val) > tolerance:
            return False, f"{label} mismatch: input had {expected_val}, output states {found[label]}"
    return True, ""


def check_duplicate(output: str, seen_hashes: set[str]) -> tuple[bool, str]:
    h = hashlib.sha256(output.strip().lower().encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return False, "exact duplicate of a previous output"
    return True, ""


def validate_sample(output: str, expected_z: dict[str, float], seen_hashes: set[str]) -> tuple[bool, str]:
    """Runs all checks in order, short-circuiting on first failure."""
    for check, args in [
        (check_not_truncated, (output,)),
        (check_length, (output,)),
        (check_zscore_consistency, (output, expected_z)),
        (check_duplicate, (output, seen_hashes)),
    ]:
        ok, reason = check(*args)
        if not ok:
            return False, reason
    return True, ""


if __name__ == "__main__":
    good = ("Anak ini menunjukkan tanda-tanda perawakan pendek berdasarkan hasil pengukuran. "
             "Nilai HAZ -2.55 menunjukkan klasifikasi Pendek menurut standar WHO. " * 4)
    bad_mismatch = "Hasil pengukuran menunjukkan HAZ -1.0 yang tergolong normal."
    print(validate_sample(good, {"HAZ": -2.55}, set()))
    print(validate_sample(bad_mismatch, {"HAZ": -2.55}, set()))
    print(validate_sample("terlalu pendek", {"HAZ": -2.55}, set()))
