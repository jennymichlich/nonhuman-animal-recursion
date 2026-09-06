"""
Two new recursion-typology tests, motivated by Rothstein (1989) / Koch (1983)'s
musical phrase-expansion typology as cited in Schreuder's "Recursion in Phonology"
chapter:

- External (prefix) phrase expansion: repeated subordinate material added BEFORE
  the main phrase. Operationalized as excess repeated-bigram structure concentrated
  in the front third of a sequence relative to a permuted null.
- Internal phrase expansion: repeated/varied material added WITHIN the phrase itself
  (distinct from center-embedding's specific single-instance symmetric return).
  Operationalized as excess repeated-bigram structure concentrated in the middle
  third of a sequence relative to a permuted null.

Both reuse the same repeated-bigram detector as the existing tail-end embedding
test, differing only in WHERE in the sequence the count is taken.
"""
import numpy as np

def count_repeated_bigrams_in_region(sequence, start_frac, end_frac):
    n = len(sequence)
    lo = int(n * start_frac)
    hi = int(n * end_frac)
    count = 0
    for i in range(lo, min(hi, n-3)):
        if sequence[i] == sequence[i+2] and sequence[i+1] == sequence[i+3]:
            count += 1
    return count

def region_surrogate_test(sequence, start_frac, end_frac, n_shuffles=2000, seed=0):
    rng = np.random.default_rng(seed)
    seq = np.array(sequence)
    observed = count_repeated_bigrams_in_region(seq, start_frac, end_frac)
    null_counts = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(seq)
        null_counts.append(count_repeated_bigrams_in_region(shuffled, start_frac, end_frac))
    null_counts = np.array(null_counts)
    z = (observed - null_counts.mean()) / (null_counts.std() + 1e-9)
    p = (np.sum(null_counts >= observed) + 1) / (n_shuffles + 1)
    return dict(observed=observed, null_mean=null_counts.mean(), null_std=null_counts.std(), z=z, p=p)

def external_prefix_test(sequence, n_shuffles=2000, seed=0):
    """Front-third repeated-bigram excess: external (prefix) phrase expansion."""
    return region_surrogate_test(sequence, 0.0, 1/3, n_shuffles, seed)

def internal_expansion_test(sequence, n_shuffles=2000, seed=0):
    """Middle-third repeated-bigram excess: internal phrase expansion."""
    return region_surrogate_test(sequence, 1/3, 2/3, n_shuffles, seed)
