# ChiRA — Synthetic Dataset Generator Spec

## Purpose
Generate 1500–3000 `(instruction, input, output)` pairs for QLoRA fine-tuning
of Qwen3-4B-Instruct on the ChiRA task: structured anthropometric data →
natural-language clinical growth narrative (Bahasa Indonesia).

## Architecture
```
Python script (orchestrator)
  ├── 1. Scenario Randomizer      → raw case (age, sex, height, weight)
  ├── 2. WHO Z-score Calculator   → HAZ, WAZ, WHZ + classification  [RULE-BASED, no LLM]
  ├── 3. Prompt Template Selector → picks instruction phrasing + style
  ├── 4. Local Agent API Call     → Qwen3.6-35B-A3B generates narrative ONLY
  └── 5. Validator + Writer       → schema check → append to dataset.jsonl
```

**Hard rule:** the local agent NEVER sees raw height/weight and computes
Z-scores itself. It only ever receives already-computed HAZ/WAZ/WHZ +
classification label as input, and its job is narrating that — never
recalculating or contradicting it.

---

## 1. Scenario Randomizer

Sample uniformly across these strata so the dataset isn't skewed toward
"normal" cases (which dominate in real population distributions but need
to be intentionally balanced for training):

| Dimension | Range / Options |
|---|---|
| Age | 0–60 months, stratified into bins: 0–6, 6–12, 12–24, 24–36, 36–48, 48–60 |
| Sex | Laki-laki / Perempuan (50/50) |
| Target classification | Normal / Stunted / Severely Stunted / (optionally wasted, overweight) — sample ~evenly, NOT population-realistic ratios |
| Height, Weight | Sampled to land in the target Z-score band for the target classification (see §2) — don't sample height/weight freely then classify after; work backward from target Z so classes are balanced |

Suggested class split for ~2500 samples:
- Normal (HAZ ≥ -2): ~35%
- Stunted (-3 ≤ HAZ < -2): ~35%
- Severely stunted (HAZ < -3): ~20%
- Edge/boundary cases (HAZ within 0.1 of a cutoff, tests report robustness near thresholds): ~10%

## 2. WHO Z-score Calculator (rule-based, code only)

Use official WHO Child Growth Standards LMS tables (2006, 0–60 months) for:
- Height-for-age (HAZ)
- Weight-for-age (WAZ)
- Weight-for-height (WHZ)

Formula (LMS method):
```
if L != 0:
    Z = (((X / M) ** L) - 1) / (L * S)
else:
    Z = ln(X / M) / S
```
Where X = measured value, and L/M/S come from the WHO reference table for
the child's exact age (months) and sex.

Classification thresholds (standard WHO cutoffs):
- HAZ ≥ -2: Normal
- -3 ≤ HAZ < -2: Stunted (Pendek)
- HAZ < -3: Severely Stunted (Sangat Pendek)
(mirror same cutoff logic for WAZ/WHZ if included)

**Implementation note:** don't hand-roll LMS tables from memory — pull the
official WHO Anthro tables (CSV, publicly published by WHO) and load them
in the script. Round Z-scores to 2 decimals before passing downstream.

## 3. Prompt Template Selector (variation, to avoid monotonous output)

Rotate across style axes so the dataset isn't repetitive — this matters
because a single generator model tends to converge on similar phrasing if
prompted the same way every time.

**Style A — Formal medis** (for clinician-facing report):
```
Buat interpretasi hasil pengukuran pertumbuhan anak berikut dalam format
laporan klinis formal, sesuai standar WHO Child Growth Standards. Gunakan
istilah medis yang tepat.
```

**Style B — Bahasa awam untuk orang tua**:
```
Jelaskan hasil pengukuran pertumbuhan anak berikut dengan bahasa yang
mudah dipahami orang tua, tanpa istilah medis yang membingungkan, tapi
tetap akurat.
```

**Style C — Ringkas / poin-poin** (SOAP-lite):
```
Ringkas hasil pengukuran ini dalam format singkat: temuan utama,
interpretasi, dan rekomendasi tindak lanjut.
```

Vary further with minor phrasing rewrites per batch (e.g. synonyms for
"buat", "jelaskan", "interpretasikan") — generate 5–10 instruction
paraphrases per style and rotate randomly, rather than reusing one exact
string 2500 times.

Target rough split: 40% Style A, 40% Style B, 20% Style C.

## 4. Local Agent Call (narrative generation only)

**Input passed to agent** (never raw height/weight alone — always with
classification already attached):
```
Nama: {nama}, Usia: {usia} bulan, Jenis kelamin: {sex},
Tinggi: {height} cm, Berat: {weight} kg,
HAZ: {haz} ({haz_class}), WAZ: {waz} ({waz_class})
```

**System prompt for the local agent** (constrain it to narration only):
```
Kamu adalah asisten yang menulis narasi laporan pertumbuhan anak dari
data yang SUDAH diklasifikasikan. JANGAN menghitung ulang atau mengubah
angka Z-score atau klasifikasi yang diberikan — gunakan apa adanya.
Tugasmu murni menulis narasi yang natural, sesuai gaya yang diminta.
```

Call via local OpenAI-compatible endpoint (adjust host/port to your
local server):
```python
import requests

def generate_narrative(structured_input, instruction, system_prompt):
    resp = requests.post(
        "http://localhost:PORT/v1/chat/completions",
        json={
            "model": "qwen3.6-35b-a3b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{instruction}\n\n{structured_input}"},
            ],
            "temperature": 0.8,   # some variation across samples
            "max_tokens": 400,
        },
    )
    return resp.json()["choices"][0]["message"]["content"]
```

Use `temperature` ~0.7–0.9 for narrative diversity (this is a case where
some randomness is desirable — unlike the Z-score step, which must be
deterministic).

## 5. Validator + Writer

Before writing each sample to `dataset.jsonl`, check:
- [ ] Output text doesn't contain a numeric Z-score/classification that
      contradicts the input (regex scan for numbers near "HAZ"/"WAZ"/
      classification keywords, flag mismatches for manual review)
- [ ] Output isn't empty / truncated / doesn't end mid-sentence
- [ ] Output length within reasonable bounds (~50–400 words)
- [ ] No duplicate output text (hash-check against previous samples,
      flag near-duplicates for regeneration with a different template)

Write each accepted sample in the schema already defined in chira.md:
```json
{
  "instruction": "...",
  "input": "Nama: ..., Usia: ..., ...",
  "output": "<narrative>"
}
```

## Batch & QC Plan
1. Run generation in batches of ~100–200 (checkpoint to disk each batch —
   don't lose progress on a multi-hour run)
2. After first batch, manually spot-check ~20 samples before scaling to
   full run — catch systemic issues (agent ignoring classification,
   template producing weird output) early
3. Final QC: random spot-check 50–100 samples across the full dataset
   before training (per chiro.md's existing QC plan)
4. Split 90% train / 10% eval only after QC passes

## Open Parameters to Decide Before Running
- [ ] Exact target dataset size (1500 vs 3000 — start smaller, evaluate,
      scale up if needed rather than committing to 3000 upfront)
- [ ] Include WHZ (weight-for-height) cases or keep to HAZ/WAZ only for v1
- [ ] Local agent endpoint/port + confirm OpenAI-compatible API is exposed
- [ ] Source of WHO LMS reference tables (exact file/version to pull)