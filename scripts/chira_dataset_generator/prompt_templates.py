"""
prompt_templates.py — Component #3 (dataset_generator.md, section 3).

Rotates instruction phrasing across three styles so the dataset doesn't
converge on repetitive phrasing (a known failure mode when one generator
model is prompted identically thousands of times).

Target split: 40% Style A (formal medis), 40% Style B (bahasa awam),
20% Style C (ringkas/poin-poin).
"""

import random

STYLE_WEIGHTS = [("A", 0.40), ("B", 0.40), ("C", 0.20)]

SYSTEM_PROMPT = (
    "Kamu adalah asisten yang menulis narasi laporan pertumbuhan anak dari "
    "data yang SUDAH diklasifikasikan. JANGAN menghitung ulang atau mengubah "
    "angka Z-score atau klasifikasi yang diberikan — gunakan apa adanya. "
    "Tugasmu murni menulis narasi yang natural, sesuai gaya yang diminta."
)

# 5-10 paraphrases per style, rotated randomly rather than reusing one
# exact string every time (spec: "generate 5-10 instruction paraphrases
# per style and rotate randomly").
STYLE_A_PARAPHRASES = [
    "Buat interpretasi hasil pengukuran pertumbuhan anak berikut dalam format laporan klinis formal, sesuai standar WHO Child Growth Standards. Gunakan istilah medis yang tepat.",
    "Susun laporan klinis formal atas hasil pengukuran pertumbuhan anak berikut, mengacu pada standar WHO Child Growth Standards, dengan terminologi medis yang akurat.",
    "Tuliskan interpretasi klinis dari data antropometri anak berikut, dalam format laporan formal sesuai WHO Child Growth Standards.",
    "Interpretasikan hasil pengukuran antropometri anak berikut secara formal, layaknya laporan klinis yang mengacu pada WHO Child Growth Standards.",
    "Buatkan asesmen pertumbuhan formal untuk anak berikut berdasarkan standar WHO, dengan bahasa dan istilah medis yang tepat.",
    "Rumuskan laporan klinis mengenai status pertumbuhan anak berikut, sesuai kaidah WHO Child Growth Standards.",
    "Susun narasi hasil pemeriksaan antropometri anak berikut dalam gaya laporan klinis formal berbasis standar WHO.",
]

STYLE_B_PARAPHRASES = [
    "Jelaskan hasil pengukuran pertumbuhan anak berikut dengan bahasa yang mudah dipahami orang tua, tanpa istilah medis yang membingungkan, tapi tetap akurat.",
    "Ceritakan kondisi pertumbuhan anak berikut kepada orang tua dengan bahasa sehari-hari yang mudah dimengerti, namun tetap akurat secara medis.",
    "Sampaikan hasil pengukuran anak berikut dengan kalimat sederhana untuk orang tua, hindari istilah teknis yang rumit.",
    "Jelaskan secara awam dan ramah kepada orang tua bagaimana kondisi pertumbuhan anak berikut, tetap berdasarkan data yang akurat.",
    "Buat penjelasan santai namun tetap tepat untuk orang tua mengenai hasil pengukuran pertumbuhan anak berikut.",
    "Uraikan hasil pemeriksaan pertumbuhan anak berikut dengan bahasa yang gampang dicerna orang awam.",
]

STYLE_C_PARAPHRASES = [
    "Ringkas hasil pengukuran ini dalam format singkat: temuan utama, interpretasi, dan rekomendasi tindak lanjut.",
    "Buat ringkasan poin-poin dari hasil pengukuran ini: temuan, interpretasi, rekomendasi.",
    "Sajikan hasil pengukuran ini secara ringkas dalam bentuk poin: temuan utama, interpretasi klinis, dan langkah lanjutan.",
    "Rangkum data pengukuran ini secara singkat: apa yang ditemukan, apa artinya, dan apa yang perlu dilakukan selanjutnya.",
]

_STYLE_POOL = {"A": STYLE_A_PARAPHRASES, "B": STYLE_B_PARAPHRASES, "C": STYLE_C_PARAPHRASES}


def pick_style(rng: random.Random) -> str:
    labels, weights = zip(*STYLE_WEIGHTS)
    return rng.choices(labels, weights=weights, k=1)[0]


def pick_instruction(rng: random.Random, style: str | None = None) -> tuple[str, str]:
    """Returns (style, instruction_text)."""
    style = style or pick_style(rng)
    return style, rng.choice(_STYLE_POOL[style])


if __name__ == "__main__":
    rng = random.Random(1)
    from collections import Counter
    counts = Counter()
    for _ in range(2000):
        style, _ = pick_instruction(rng)
        counts[style] += 1
    print("Style distribution over 2000 draws:", dict(counts))
    print()
    for style in ["A", "B", "C"]:
        s, instr = pick_instruction(rng, style=style)
        print(f"[{s}] {instr}")
