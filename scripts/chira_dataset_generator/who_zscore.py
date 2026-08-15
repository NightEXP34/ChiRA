"""
who_zscore.py — Rule-based WHO Child Growth Standards Z-score calculator.

Part of ChiRA's synthetic dataset generator (see docs/dataset_generator.md,
section 2: "WHO Z-score Calculator (rule-based, code only)").

HARD RULE: this module is deterministic, pure arithmetic. No LLM involved.
The narrative-generation agent downstream only ever receives the OUTPUT of
this module (already-computed Z-score + classification) — never raw
height/weight — per the project's "no numeric reasoning by the LLM" design
constraint (chira.md, "Critical Design Constraint").

Reference data: official WHO Child Growth Standards (2006) LMS tables,
0-60 months, for:
  - HAZ (length/height-for-age)
  - WAZ (weight-for-age)
  - WHZ (weight-for-length 0-2y / weight-for-height 2-5y)
Tables stored as JSON in data/who_tables/, one row per month (age-based)
or per cm (length/height-based), each row giving L, M, S per WHO's
published values. Do not hand-edit these files.
"""

import json
import math
import os
from dataclasses import dataclass
from enum import Enum

TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "who_tables")


class Sex(str, Enum):
    LAKI_LAKI = "Laki-laki"
    PEREMPUAN = "Perempuan"


def _sex_key(sex: Sex | str) -> str:
    sex = Sex(sex)
    return "boys" if sex == Sex.LAKI_LAKI else "girls"


def _load_table(name: str) -> list[dict]:
    path = os.path.join(TABLES_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Cache so repeated calls (e.g. generating thousands of synthetic samples)
# don't re-read JSON off disk every time.
_TABLE_CACHE: dict[str, list[dict]] = {}


def _get_table(name: str) -> list[dict]:
    if name not in _TABLE_CACHE:
        _TABLE_CACHE[name] = _load_table(name)
    return _TABLE_CACHE[name]


def _lookup_lms(table: list[dict], key_field: str, key_value: float) -> tuple[float, float, float]:
    """
    Find L, M, S for the given key (age in whole months, or length/height
    rounded to nearest cm — matching how the WHO tables are indexed).
    Uses the nearest available row rather than interpolating, which is
    standard practice for age-in-months tables and an acceptable
    simplification for the length/height tables at this task's precision.
    """
    target = float(key_value)
    best_row = min(table, key=lambda row: abs(float(row[key_field]) - target))
    return float(best_row["L"]), float(best_row["M"]), float(best_row["S"])


def _lms_zscore(x: float, l: float, m: float, s: float) -> float:
    """Core WHO LMS formula (dataset_generator.md, section 2)."""
    if l != 0:
        z = (((x / m) ** l) - 1) / (l * s)
    else:
        z = math.log(x / m) / s
    return round(z, 2)


def _lms_inverse(z: float, l: float, m: float, s: float) -> float:
    """
    Inverse of the WHO LMS formula: given a target Z, return the measured
    value X (height or weight) that would produce it. Used by the
    Scenario Randomizer to work backward from a target classification to
    a raw measurement, so dataset classes can be balanced instead of
    sampled from a population-realistic (mostly-normal) distribution.
    """
    if l != 0:
        x = m * ((1 + l * s * z) ** (1 / l))
    else:
        x = m * math.exp(s * z)
    return x


def invert_haz(age_months: float, sex: Sex | str, target_z: float) -> float:
    """Given a target HAZ, return the height (cm) that produces it."""
    table = _get_table(f"haz_{_sex_key(sex)}_0_5")
    l, m, s = _lookup_lms(table, "Month", round(age_months))
    return round(_lms_inverse(target_z, l, m, s), 1)


def invert_waz(age_months: float, sex: Sex | str, target_z: float) -> float:
    """Given a target WAZ, return the weight (kg) that produces it."""
    table = _get_table(f"waz_{_sex_key(sex)}_0_5")
    l, m, s = _lookup_lms(table, "Month", round(age_months))
    return round(_lms_inverse(target_z, l, m, s), 1)


def invert_whz(age_months: float, sex: Sex | str, height_cm: float, target_z: float) -> float:
    """Given a height and a target WHZ, return the weight (kg) that produces it."""
    sex_key = _sex_key(sex)
    if age_months < 24:
        table = _get_table(f"whz_{sex_key}_0_2")
        key_field = "Length"
    else:
        table = _get_table(f"whz_{sex_key}_2_5")
        key_field = "Height"
    l, m, s = _lookup_lms(table, key_field, round(height_cm))
    return round(_lms_inverse(target_z, l, m, s), 1)


def calculate_haz(age_months: float, sex: Sex | str, height_cm: float) -> float:
    """Height/length-for-age Z-score."""
    table = _get_table(f"haz_{_sex_key(sex)}_0_5")
    l, m, s = _lookup_lms(table, "Month", round(age_months))
    return _lms_zscore(height_cm, l, m, s)


def calculate_waz(age_months: float, sex: Sex | str, weight_kg: float) -> float:
    """Weight-for-age Z-score."""
    table = _get_table(f"waz_{_sex_key(sex)}_0_5")
    l, m, s = _lookup_lms(table, "Month", round(age_months))
    return _lms_zscore(weight_kg, l, m, s)


def calculate_whz(age_months: float, sex: Sex | str, height_cm: float, weight_kg: float) -> float:
    """
    Weight-for-length/height Z-score. WHO switches reference table at 2
    years: weight-for-length (recumbent, 0-2y) vs weight-for-height
    (standing, 2-5y). We select by age, matching WHO's convention.
    """
    sex_key = _sex_key(sex)
    if age_months < 24:
        table = _get_table(f"whz_{sex_key}_0_2")
        key_field = "Length"
    else:
        table = _get_table(f"whz_{sex_key}_2_5")
        key_field = "Height"
    l, m, s = _lookup_lms(table, key_field, round(height_cm))
    return _lms_zscore(weight_kg, l, m, s)


def classify_haz(haz: float) -> str:
    if haz < -3:
        return "Sangat Pendek"  # Severely stunted
    if haz < -2:
        return "Pendek"  # Stunted
    if haz > 3:
        return "Sangat Tinggi"
    return "Normal"


def classify_waz(waz: float) -> str:
    if waz < -3:
        return "Berat Badan Sangat Kurang"  # Severely underweight
    if waz < -2:
        return "Berat Badan Kurang"  # Underweight
    if waz > 2:
        return "Berat Badan Lebih"
    return "Normal"


def classify_whz(whz: float) -> str:
    if whz < -3:
        return "Gizi Buruk"  # Severe wasting
    if whz < -2:
        return "Gizi Kurang"  # Wasting
    if whz > 3:
        return "Obesitas"
    if whz > 2:
        return "Gizi Lebih"  # Overweight
    return "Normal"


@dataclass
class GrowthAssessment:
    age_months: float
    sex: str
    height_cm: float
    weight_kg: float
    haz: float
    haz_class: str
    waz: float
    waz_class: str
    whz: float | None = None
    whz_class: str | None = None

    def to_structured_input(self, include_whz: bool = False) -> str:
        """Renders the format used in chira.md's data schema, e.g.:
        'Nama: Anak A, Usia: 28 bulan, Jenis kelamin: Laki-laki, ...'
        """
        parts = [
            f"Usia: {self.age_months:g} bulan",
            f"Jenis kelamin: {self.sex}",
            f"Tinggi: {self.height_cm:g} cm",
            f"Berat: {self.weight_kg:g} kg",
            f"HAZ: {self.haz:.2f} ({self.haz_class})",
            f"WAZ: {self.waz:.2f} ({self.waz_class})",
        ]
        if include_whz and self.whz is not None:
            parts.append(f"WHZ: {self.whz:.2f} ({self.whz_class})")
        return ", ".join(parts)


def assess(age_months: float, sex: Sex | str, height_cm: float, weight_kg: float,
           include_whz: bool = False) -> GrowthAssessment:
    """Single entry point: compute + classify everything for one case."""
    sex = Sex(sex)
    haz = calculate_haz(age_months, sex, height_cm)
    waz = calculate_waz(age_months, sex, weight_kg)
    result = GrowthAssessment(
        age_months=age_months, sex=sex.value, height_cm=height_cm, weight_kg=weight_kg,
        haz=haz, haz_class=classify_haz(haz),
        waz=waz, waz_class=classify_waz(waz),
    )
    if include_whz:
        whz = calculate_whz(age_months, sex, height_cm, weight_kg)
        result.whz = whz
        result.whz_class = classify_whz(whz)
    return result


if __name__ == "__main__":
    # Sanity check against the example row in chira.md's Data Format section:
    # "Usia: 28 bulan, Laki-laki, Tinggi: 82 cm, Berat: 10.2 kg, HAZ: -2.8, WAZ: -2.1"
    r = assess(28, Sex.LAKI_LAKI, 82, 10.2, include_whz=True)
    print(r.to_structured_input(include_whz=True))
    print(f"Expected ~HAZ -2.8, WAZ -2.1 -> got HAZ {r.haz}, WAZ {r.waz}")
