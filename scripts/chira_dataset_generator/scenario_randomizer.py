"""
scenario_randomizer.py — Component #1 of ChiRA's dataset generator
(dataset_generator.md, section 1: "Scenario Randomizer").

Samples age/sex/target-classification, then works BACKWARD via inverse
WHO LMS (who_zscore.invert_*) to a raw height/weight that actually lands
in the target Z-score band — rather than sampling height/weight freely
and classifying after, which would skew the dataset toward "Normal"
(population-realistic ratios) instead of the balanced split the spec
calls for.

Two independent tracks per sample, matching the spec's "(optionally
wasted, overweight)" note:
  - stunting track (~85% of samples): targets a HAZ class, derives
    height; WAZ is sampled correlated with HAZ severity (stunted kids
    are often also underweight in reality) and derives weight; WHZ is
    left to fall out as a *derived* value from that height+weight.
  - wasting/overweight track (~15% of samples): height sampled from an
    age-typical normal range, WHZ target sampled explicitly (wasted /
    severely wasted / overweight), weight derived from that. HAZ is
    sampled near-normal for these so the case isn't confounded with
    stunting.
"""

import random
from dataclasses import dataclass, field

from who_zscore import Sex, assess, invert_haz, invert_waz, invert_whz, GrowthAssessment

AGE_BINS = [(0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (48, 60)]

# HAZ target classification -> (weight in the split, Z-value sampling range)
HAZ_TARGETS = [
    ("Normal", 0.35, (-1.9, 2.0)),
    ("Stunted", 0.35, (-2.9, -2.0)),
    ("Severely Stunted", 0.20, (-5.0, -3.0)),
    ("Boundary", 0.10, None),  # handled specially: near -2 or -3 cutoff
]

WASTING_TRACK_SHARE = 0.15

WHZ_TARGETS = [
    ("Gizi Lebih (overweight)", 0.30, (2.1, 3.2)),
    ("Gizi Kurang (wasted)", 0.40, (-2.9, -2.0)),
    ("Gizi Buruk (severely wasted)", 0.30, (-5.0, -3.0)),
]


@dataclass
class RawCase:
    age_months: float
    sex: Sex
    height_cm: float
    weight_kg: float
    target_track: str  # "stunting" | "wasting"
    target_label: str  # the class we deliberately aimed for, for QC/logging
    assessment: GrowthAssessment = field(default=None)


def _sample_age_months(rng: random.Random) -> float:
    lo, hi = rng.choice(AGE_BINS)
    # keep to whole months, matching how the WHO age-based tables are indexed
    return float(rng.randint(lo, hi))


def _sample_haz_target(rng: random.Random) -> tuple[str, float]:
    labels, weights, _ = zip(*[(l, w, r) for l, w, r in HAZ_TARGETS])
    ranges = {l: r for l, w, r in HAZ_TARGETS}
    label = rng.choices(labels, weights=weights, k=1)[0]
    if label == "Boundary":
        cutoff = rng.choice([-2.0, -3.0])
        z = cutoff + rng.uniform(-0.1, 0.1)
        return f"Boundary (~{cutoff:.0f})", round(z, 2)
    lo, hi = ranges[label]
    return label, round(rng.uniform(lo, hi), 2)


def _sample_waz_target_correlated(rng: random.Random, haz_z: float) -> float:
    """
    WAZ tends to track HAZ severity in real stunting cases (chronic
    malnutrition affects both). Center WAZ near HAZ with noise rather
    than sampling fully independently, so cases read as clinically
    coherent rather than contradictory (e.g. severely stunted + very
    healthy weight would be an unusual, confusing training example).
    """
    z = haz_z + rng.uniform(-0.8, 0.8)
    return round(max(min(z, 3.5), -5.5), 2)


def _sample_whz_target(rng: random.Random) -> tuple[str, float]:
    labels, weights, _ = zip(*[(l, w, r) for l, w, r in WHZ_TARGETS])
    ranges = {l: r for l, w, r in WHZ_TARGETS}
    label = rng.choices(labels, weights=weights, k=1)[0]
    lo, hi = ranges[label]
    return label, round(rng.uniform(lo, hi), 2)


def generate_case(rng: random.Random) -> RawCase:
    age = _sample_age_months(rng)
    sex = rng.choice([Sex.LAKI_LAKI, Sex.PEREMPUAN])

    if rng.random() < WASTING_TRACK_SHARE:
        # Wasting/overweight track: near-normal HAZ, explicit WHZ target.
        _, haz_z = _sample_haz_target_normal_only(rng)
        height = invert_haz(age, sex, haz_z)
        whz_label, whz_z = _sample_whz_target(rng)
        weight = invert_whz(age, sex, height, whz_z)
        track, label = "wasting", whz_label
    else:
        haz_label, haz_z = _sample_haz_target(rng)
        height = invert_haz(age, sex, haz_z)
        waz_z = _sample_waz_target_correlated(rng, haz_z)
        weight = invert_waz(age, sex, waz_z)
        track, label = "stunting", haz_label

    # Guard against physically nonsensical values from extreme inversions
    # (e.g. negative or absurd weight at the tails of the Z range).
    height = max(height, 30.0)
    weight = max(weight, 1.0)

    case = RawCase(age_months=age, sex=sex, height_cm=height, weight_kg=weight,
                   target_track=track, target_label=label)
    # Recompute the REAL classification from the final height/weight pair
    # (source of truth for what goes into the dataset — the target above
    # is only a sampling aid, the actual HAZ/WAZ/WHZ come from who_zscore
    # applied to the final numbers, same as production would do).
    case.assessment = assess(age, sex, height, weight, include_whz=True)
    return case


def _sample_haz_target_normal_only(rng: random.Random) -> tuple[str, float]:
    return "Normal", round(rng.uniform(-1.5, 1.5), 2)


def generate_batch(n: int, seed: int | None = None) -> list[RawCase]:
    rng = random.Random(seed)
    return [generate_case(rng) for _ in range(n)]


if __name__ == "__main__":
    batch = generate_batch(20, seed=42)
    from collections import Counter
    haz_counts = Counter(c.assessment.haz_class for c in batch)
    whz_counts = Counter(c.assessment.whz_class for c in batch)
    for c in batch[:8]:
        print(c.target_track, "|", c.target_label, "->", c.assessment.to_structured_input(include_whz=True))
    print("\nHAZ class distribution (n=20):", dict(haz_counts))
    print("WHZ class distribution (n=20):", dict(whz_counts))
