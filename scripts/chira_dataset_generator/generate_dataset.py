"""
generate_dataset.py — Orchestrator for ChiRA's synthetic dataset
(dataset_generator.md, "Batch & QC Plan").

Pipeline per sample:
  1. scenario_randomizer -> raw case (age, sex, height, weight)
  2. who_zscore (already applied inside scenario_randomizer.assess)
     -> HAZ/WAZ/WHZ + classification  [RULE-BASED]
  3. prompt_templates -> instruction style + phrasing
  4. agent_client -> narrative from local Qwen3.5-9B (llama.cpp, thinking disabled)
  5. validator -> accept/reject, append to dataset.jsonl

Run in batches of ~100-200 with a checkpoint file, so a multi-hour run
survives interruption:
  python3 generate_dataset.py --n 2500 --batch-size 150 --out dataset.jsonl

Resumes automatically if dataset.jsonl + .progress already exist.

Terminal output: a single live progress bar (ETA, rate, accept %) that
updates in place, so you don't need to tail the llama.cpp server log to
know the run is alive. Rejects/errors print as normal scrolling lines
above the bar so you can spot systemic issues without losing the bar.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time

from scenario_randomizer import generate_case
from prompt_templates import pick_instruction, SYSTEM_PROMPT
from agent_client import generate_narrative, AgentError, check_connection
from validator import validate_sample


def load_existing(out_path: str) -> tuple[list[dict], set[str]]:
    """Resume support: read what's already in dataset.jsonl."""
    samples, hashes = [], set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(obj)
                hashes.add(hashlib.sha256(obj["output"].strip().lower().encode("utf-8")).hexdigest())
    return samples, hashes


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 24)).columns
    except Exception:
        return default


def _fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds <= 0:  # NaN or non-positive
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_progress(n_written: int, n_target: int, n_rejected: int, run_start: float, n_have: int) -> None:
    """Redraw the single-line live progress bar in place (carriage return, no newline).

    Rate/ETA are computed from samples produced THIS session only
    (n_written - n_have) — otherwise resumed samples from a prior run
    inflate the rate early on and the ETA drifts upward as the session
    goes on instead of converging.
    """
    width = _term_width()
    bar_width = max(10, min(30, width - 72))
    frac = min(1.0, n_written / n_target) if n_target else 1.0
    filled = int(bar_width * frac)
    bar = "#" * filled + "-" * (bar_width - filled)

    elapsed = time.time() - run_start
    produced_this_session = max(0, n_written - n_have)
    # Guard against div-by-near-zero on the very first draw (elapsed ~0s).
    rate = produced_this_session / elapsed if elapsed > 1.0 else 0.0
    remaining = max(0, n_target - n_written)
    eta = remaining / rate if rate > 0 else float("nan")
    total_tries = n_written + n_rejected
    accept_pct = (n_written / total_tries * 100) if total_tries else 100.0

    line = (f"\r[{bar}] {frac*100:5.1f}%  {n_written}/{n_target}  "
            f"rejected={n_rejected} ({accept_pct:.0f}% accept)  "
            f"{rate:.2f} samp/s  ETA {_fmt_eta(eta)}")
    sys.stdout.write(line[:width].ljust(width))
    sys.stdout.flush()


def _print_above_bar(msg: str, n_written: int, n_target: int, n_rejected: int, run_start: float, n_have: int) -> None:
    """Clear the current bar line, print a normal scrolling message, then redraw the bar below it."""
    sys.stdout.write("\r" + " " * _term_width() + "\r")
    print(msg)
    _draw_progress(n_written, n_target, n_rejected, run_start, n_have)


def run(n_target: int, batch_size: int, out_path: str, seed: int, log_rejects_path: str):
    existing, seen_hashes = load_existing(out_path)
    n_have = len(existing)
    if n_have >= n_target:
        print(f"Already have {n_have} samples in {out_path} (target {n_target}). Nothing to do.")
        return

    print(f"Resuming: {n_have}/{n_target} samples already in {out_path}.")
    print("Checking connection to local agent...")
    if not check_connection():
        print("ABORTING: cannot reach local agent. Check agent_client.AGENT_ENDPOINT "
              "and that your llama.cpp server is running.")
        sys.exit(1)

    rng = random.Random(seed + n_have)  # offset so resumes don't repeat the same sequence
    out_f = open(out_path, "a", encoding="utf-8")
    reject_f = open(log_rejects_path, "a", encoding="utf-8")

    n_written = n_have
    n_rejected = 0
    n_consecutive_rejects = 0
    run_start = time.time()

    # Circuit breaker: if the agent is systematically misbehaving (e.g.
    # ignoring the "don't recompute the classification" instruction, or
    # the endpoint is silently returning garbage), fail loudly instead of
    # spinning forever rejecting every sample. This many consecutive
    # rejects strongly suggests a systemic issue, not sampling bad luck.
    MAX_CONSECUTIVE_REJECTS = 50

    _draw_progress(n_written, n_target, n_rejected, run_start, n_have)

    try:
        while n_written < n_target:
            for _ in range(min(batch_size, n_target - n_written)):
                if n_consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    print()
                    print(f"ABORTING: {MAX_CONSECUTIVE_REJECTS} consecutive rejects. "
                          f"This usually means the agent is ignoring the system prompt "
                          f"(e.g. recomputing/contradicting the given Z-score) or the "
                          f"endpoint is misconfigured. Check {log_rejects_path} for the "
                          f"pattern before re-running.")
                    sys.exit(1)

                case = generate_case(rng)
                style, instruction = pick_instruction(rng)
                structured_input = case.assessment.to_structured_input(include_whz=True)

                expected_z = {"HAZ": case.assessment.haz, "WAZ": case.assessment.waz}
                if case.assessment.whz is not None:
                    expected_z["WHZ"] = case.assessment.whz

                try:
                    narrative = generate_narrative(structured_input, instruction, SYSTEM_PROMPT)
                except AgentError as e:
                    n_rejected += 1
                    n_consecutive_rejects += 1
                    reject_f.write(json.dumps({"reason": f"agent error: {e}", "input": structured_input},
                                               ensure_ascii=False) + "\n")
                    reject_f.flush()
                    _print_above_bar(f"  ! agent error: {e}", n_written, n_target, n_rejected, run_start, n_have)
                    continue

                ok, reason = validate_sample(narrative, expected_z, seen_hashes)
                if not ok:
                    n_rejected += 1
                    n_consecutive_rejects += 1
                    reject_f.write(json.dumps({"reason": reason, "input": structured_input, "output": narrative},
                                               ensure_ascii=False) + "\n")
                    reject_f.flush()
                    _print_above_bar(f"  ! rejected: {reason}", n_written, n_target, n_rejected, run_start, n_have)
                    continue

                n_consecutive_rejects = 0
                seen_hashes.add(hashlib.sha256(narrative.strip().lower().encode("utf-8")).hexdigest())
                record = {"instruction": instruction, "input": structured_input, "output": narrative}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()  # checkpoint every sample, not just every batch — cheap and crash-safe
                n_written += 1
                _draw_progress(n_written, n_target, n_rejected, run_start, n_have)
    finally:
        out_f.close()
        reject_f.close()

    print()
    print(f"Done. {n_written} samples in {out_path}, {n_rejected} rejected "
          f"(see {log_rejects_path} for reasons — spot-check these for systemic issues).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500, help="target dataset size")
    p.add_argument("--batch-size", type=int, default=150)
    p.add_argument("--out", type=str, default="dataset.jsonl")
    p.add_argument("--reject-log", type=str, default="dataset_rejects.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run(n_target=args.n, batch_size=args.batch_size, out_path=args.out,
        seed=args.seed, log_rejects_path=args.reject_log)