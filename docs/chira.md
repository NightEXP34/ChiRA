# ChiRA — Children Growth Assistance

## Overview
ChiRA (Children Growth Assistance) is a QLoRA fine-tuning project that converts structured
child anthropometric measurements into natural-language clinical growth
assessment narratives. This is Candra's first hands-on LLM fine-tuning
project — prior work was vision/CNN-based (stunting detection, growth
monitoring research).

Fits into the broader research pipeline as a downstream stage:

```
vision model (existing research) → structured measurement data → ChiRA (LLM) → clinical narrative report
```

## Task Definition
- **Type**: Format/style adaptation (NOT knowledge injection)
- **Input**: Structured data — age, sex, height, weight, and pre-computed
  Z-scores (HAZ, WAZ, WHZ)
- **Output**: Natural-language clinical report narrating the growth
  assessment, following a WHO-style growth report format

## Critical Design Constraint
**Z-score calculation and classification (normal / stunted / severely
stunted) must be done by rule-based code using WHO Child Growth Standards
formulas — never by the LLM.** The LLM's job is strictly narrative
generation from already-classified data, not numeric reasoning. This keeps
the fine-tuning task narrow (format-following) and avoids hallucinated
numbers.

## Reference Formats
- SOAP note (Subjective, Objective, Assessment, Plan)
- Findings + Impression (radiology-style)
- Anthropometric/growth assessment report — primary target format,
  interpreting HAZ/WAZ/WHZ against WHO Child Growth Standards

## Model & Method
- **Base model**: Qwen3-4B-Instruct — repo:
  `unsloth/Qwen3-4B-unsloth-bnb-4bit` (pre-quantized bnb-4bit, official
  Qwen weights repacked by Unsloth, not a third-party finetune). Chosen
  over 8B because: (1) task is format/style adaptation, not
  knowledge-heavy — 4B and 8B share the same multilingual pretraining
  corpus (119 languages, Indonesian included), so base-language capability
  is expected to be comparable; the main gap is reasoning depth, which
  this narrow narrative-generation task doesn't stress much; (2) fits
  edge deployment target (Raspberry Pi 5, see below) far better than 8B.
  Fallback candidate: `unsloth/Qwen3-8B-unsloth-bnb-4bit`, only if 4B
  eval quality (Indonesian narrative coherence, clinical phrasing) turns
  out insufficient after spot-check.
- **Method**: QLoRA, 4-bit quantization
- **Framework**: Unsloth
- **Epochs**: 1–3 (LLM fine-tuning overfits fast on small instruction
  datasets — unlike vision/YOLO training, the base model already knows
  language; fine-tuning only nudges format, not from-scratch learning)

## Hardware
- Laptop RTX 4060 Mobile, 8GB VRAM, 256GB/s bandwidth (see
  [[laptop-thermal-tuning]] for chassis/thermal considerations —
  sustained multi-hour training load has different thermal behavior than
  gaming bursts)
- Estimated training time: ~1–3 hours (Qwen3-4B QLoRA, ~2500 samples,
  seq len ~512, 3 epochs) — depends on actual token length, batch size,
  and thermal throttling
- ~7–7.5GB VRAM footprint expected at 4-bit (base model + LoRA adapter +
  activations/gradients + optimizer state), leaves headroom vs 8B on the
  8GB card

## Edge Deployment (target, post-training)
- **Target device**: Raspberry Pi 5 (8GB RAM), CPU-only inference via
  llama.cpp
- **Export path**: after QLoRA training, merge/export fine-tuned model to
  GGUF (Unsloth `save_pretrained_gguf`), quantize to Q4_K_M
- **Expected throughput**: no official Qwen benchmark for Pi 5; community
  numbers for similarly-sized (3–4B) Q4 GGUF models on Pi 5 land around
  ~4–7 tok/s CPU-only — usable for non-realtime report generation, not
  interactive chat speed. 8B-class models drop to ~1–3 tok/s, which is
  the practical reason 4B was chosen as primary over 8B.
- **Rationale**: vision model (existing research) already targets
  edge/field deployment — keeping ChiRA's LLM stage edge-viable keeps the
  full pipeline deployable on the same class of hardware

## Dataset Plan
- **No existing off-the-shelf dataset** for this exact task (pediatric
  anthropometric data → growth narrative). Closest public analogues:
  MIMIC-IV-Note (general clinical notes), MTS-Dialog (doctor-patient
  conversation → note) — not directly usable.
- **Approach**: Synthetic dataset generation
  1. Randomly generate realistic age/height/weight/sex combinations
     (grounded in WHO growth standard distributions, not arbitrary)
  2. Compute Z-scores and classification via rule-based WHO formula (code,
     not LLM)
  3. Feed structured data to a large LLM (local agent: Qwen3.6-35B-A3B) to
     generate narrative variations
  4. Vary prompt style: formal medical vs. plain-language for parents, to
     avoid monotonous/generic-sounding output
- **Target size**: ~1500–3000 instruction-response pairs
- **Split**: ~90% train / ~10% held-out eval
- **Quality control**: spot-check ~50–100 samples manually for clinical
  hallucination before training

## Data Format
```json
{
  "instruction": "Buat interpretasi hasil pengukuran pertumbuhan anak berikut dalam format laporan klinis.",
  "input": "Nama: Anak A, Usia: 28 bulan, Jenis kelamin: Laki-laki, Tinggi: 82 cm, Berat: 10.2 kg, HAZ: -2.8, WAZ: -2.1",
  "output": "<narrative report following WHO growth assessment format>"
}
```

## Status
- [x] Concept + target format decided
- [x] Model + method decided (Qwen3-4B-Instruct, unsloth bnb-4bit, QLoRA,
      Unsloth — 8B kept as fallback if eval quality insufficient)
- [x] Edge deployment target decided (Raspberry Pi 5, GGUF Q4_K_M export)
- [ ] Base model downloaded (`unsloth/Qwen3-4B-unsloth-bnb-4bit`)
- [ ] Synthetic dataset generator (scenario randomizer + WHO Z-score
      calculator + prompt templates)
- [ ] Dataset generation run (via local Qwen3.6-35B-A3B agent)
- [ ] Training script setup
- [ ] Training run + eval
- [ ] GGUF export + Raspberry Pi 5 throughput test
