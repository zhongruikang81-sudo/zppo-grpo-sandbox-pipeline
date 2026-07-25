"""
Data overlap / duplication audit for the ZPPO-GRPO sandbox pipeline.

Re-runnable verification of the figures disclosed in README § Known Issues:
  1. Exact-match leakage between the 500-question test set and the training pool.
  2. Near-duplicate test questions (SequenceMatcher similarity vs training segment).
  3. Near-duplication inside the training pool.

Usage:
    python data/audit_data_overlap.py
"""
import json
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SIMILARITY_HIGH = 0.90   # template-variant threshold
SIMILARITY_EXACTISH = 0.99  # near-exact twin threshold
MAX_BOILERPLATE_GROUP = 50  # prefix groups larger than this are instruction boilerplate


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).strip().lower())


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    numina = load_jsonl(DATA_DIR / "numina_gsm_mix_numeric.jsonl")
    hendrycks = load_jsonl(DATA_DIR / "hendrycks_math_grpo_numeric.jsonl")
    test = json.load(open(DATA_DIR / "numina_500_test.json", encoding="utf-8"))

    pool_qs = [norm(r["prompt"]) for r in numina]
    train_qs = pool_qs[:1980]          # segment trainers can actually consume
    test_qs = [norm(r["prompt"]) for r in test]
    hendr_qs = [norm(r.get("question", r.get("prompt", ""))) for r in hendrycks]

    print("== 1. Exact-match leakage ==")
    # The test set is a slice (indices 1980-2479) of the same source file, so
    # overlap against the FULL pool is 500 by construction. The meaningful
    # leakage metric is overlap against the segment trainers actually consume
    # (indices 0-1979 for GRPO step-indexed sampling and the sequential ZPPO
    # replay buffer) plus the Hendrycks MATH training file.
    consumed_exact = set(train_qs) | set(hendr_qs)
    leak = [i for i, q in enumerate(test_qs) if q in consumed_exact]
    print(f"test(500) exact overlap with CONSUMED training segment + hendrycks: {len(leak)}")
    print("(test set itself = pool indices 1980-2479; holdout relies on training")
    print(" consumption staying below index 1980 — no code-level guard enforces this)")

    print("\n== 2. Near-duplicate test questions (vs training segment 0-1979) ==")
    sig_index = {}
    for i, q in enumerate(train_qs):
        sig_index.setdefault(q[:60], []).append(i)
    pairs = []
    for ti, q in enumerate(test_qs):
        group = sig_index.get(q[:60], [])
        if len(group) > MAX_BOILERPLATE_GROUP:
            continue
        for tr in group:
            r = SequenceMatcher(None, q, train_qs[tr]).ratio()
            if r >= 0.80:
                pairs.append((ti, tr, round(r, 3)))
    pairs.sort(key=lambda x: -x[2])
    for ti, tr, r in pairs:
        print(f"  test#{ti} <-> train#{tr}: {r}")
    print(f"  >=0.99: {sum(1 for p in pairs if p[2] >= SIMILARITY_EXACTISH)}"
          f" | >=0.90: {sum(1 for p in pairs if p[2] >= SIMILARITY_HIGH)}"
          f" | >=0.80: {len(pairs)}")

    print("\n== 3. Training-pool duplication ==")
    print(f"  exact duplicates: {len(pool_qs) - len(set(pool_qs))} / {len(pool_qs)}")
    groups = {}
    for i, q in enumerate(pool_qs):
        groups.setdefault(q[:60], []).append(i)
    flagged = set()
    n_pairs = 0
    for idxs in groups.values():
        if 2 <= len(idxs) <= MAX_BOILERPLATE_GROUP:
            for a, b in combinations(idxs, 2):
                if SequenceMatcher(None, pool_qs[a], pool_qs[b]).ratio() >= SIMILARITY_HIGH:
                    flagged.add(a)
                    flagged.add(b)
                    n_pairs += 1
    print(f"  near-duplicates (>={SIMILARITY_HIGH}): {len(flagged)} rows "
          f"({len(flagged) / len(pool_qs) * 100:.1f}%), {n_pairs} pairs")


if __name__ == "__main__":
    main()
