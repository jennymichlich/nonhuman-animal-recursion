"""
Given a call-type sequence for a single recording, run all five recursion
tests used in this analysis and return their results as one dict.

This is the per-file entry point. It assumes you already have:
  - a list of (onset, offset) times for each detected call in the recording
    (see call_detection.py for one way to get these from raw audio), and
  - a call-type label for each call, either from expert annotation or from
    unsupervised clustering (see recursion_pipeline.extract_features +
    cluster_types for the clustering approach used in this study).

Usage:
    from run_statistical_tests import run_all_tests
    results = run_all_tests(['A','B','A','C','B','A', ...])
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from recursion_pipeline import (
    aba_surrogate_test,
    bigram_surrogate_test,
    boundary_surrogate_test,
)
from new_expansion_tests import (
    external_prefix_test,
    internal_expansion_test,
)


def run_all_tests(sequence, n_shuffles=2000, seed=0):
    """
    Run all five recursion tests on a single call-type sequence.

    Parameters
    ----------
    sequence : list
        Ordered list of call-type labels (any hashable type: str, int, etc.)
        for one recording/bout, in chronological order.
    n_shuffles : int
        Number of random permutations used to build each null distribution.
    seed : int
        RNG seed, for reproducibility.

    Returns
    -------
    dict with keys: aba, tail_bigram, boundary, external_prefix, internal_expansion
    Each value is itself a dict with at least 'z' and 'p' keys (boundary may
    be None if the sequence has too few occurrences of its most frequent
    symbol to compute a trend).
    """
    if len(sequence) < 4:
        raise ValueError(
            f"Sequence has only {len(sequence)} calls; all five tests need "
            "at least ~15 calls per file to be meaningful (see Methods)."
        )

    return {
        "aba": aba_surrogate_test(sequence, n_shuffles=n_shuffles, seed=seed),
        "tail_bigram": bigram_surrogate_test(sequence, n_shuffles=n_shuffles, seed=seed),
        "boundary": boundary_surrogate_test(sequence, n_shuffles=n_shuffles, seed=seed),
        "external_prefix": external_prefix_test(sequence, n_shuffles=n_shuffles, seed=seed),
        "internal_expansion": internal_expansion_test(sequence, n_shuffles=n_shuffles, seed=seed),
    }


def validate_on_synthetic_data(n_trials=5, seed=0):
    """
    Sanity check used throughout this study: every test should show an
    appropriately null (non-significant) result on purely random sequences,
    and should have real detection power on sequences engineered to contain
    the pattern it's designed to find. Run this before trusting results on
    a new species or a modified test.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    print("=== Null check: random sequences (should mostly be non-significant) ===")
    for trial in range(n_trials):
        seq = list(rng.integers(0, 3, size=60))
        results = run_all_tests(seq, seed=trial)
        flags = {k: round(v["p"], 3) if v else None for k, v in results.items()}
        print(f"trial {trial}: {flags}")

    print("\n=== Power check: engineered positive-control sequences ===")
    # Center-embedded: A B A B A B ... with a genuine ABA pattern threaded through
    aba_seq = list(np.tile([0, 1, 0], 20))
    r = aba_surrogate_test(aba_seq, seed=seed)
    print(f"ABA-engineered sequence: z={r['z']:.2f}, p={r['p']:.4f} (should be significant)")

    # Tail-end: a repeated bigram unit
    tail_seq = list(np.tile([0, 1], 20))
    r = bigram_surrogate_test(tail_seq, seed=seed)
    print(f"Tail-bigram-engineered sequence: z={r['z']:.2f}, p={r['p']:.4f} (should be significant)")

    # Front-loaded repetition
    front_seq = list(np.array([0, 1] * 8)) + list(rng.integers(0, 3, size=54))
    r = external_prefix_test(front_seq, seed=seed)
    print(f"Front-loaded sequence: z={r['z']:.2f}, p={r['p']:.4f} (should be significant)")

    # Middle-loaded repetition
    mid_seq = list(rng.integers(0, 3, size=25)) + list(np.array([0, 1] * 10)) + list(rng.integers(0, 3, size=25))
    r = internal_expansion_test(mid_seq, seed=seed)
    print(f"Middle-loaded sequence: z={r['z']:.2f}, p={r['p']:.4f} (should be significant)")


if __name__ == "__main__":
    validate_on_synthetic_data()
