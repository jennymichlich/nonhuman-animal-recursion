"""
Test for center-embedded (ABA) structure in real call-type sequences, following
the recursion typology in the draft paper (center-embedded, tail-end embedded, etc.)
and directly analogous to Rey, Perruchet & Fagot (2012)'s test for center-embedding
in baboon (Papio papio) sequence processing -- but here testing for SPONTANEOUS
occurrence in natural vocal bouts rather than a trained operant task.

Method:
1. Extract acoustic features per call (duration, spectral centroid, bandwidth, rolloff).
2. Cluster into k discrete "call types" via KMeans (a data-driven proxy for the
   labeled call categories, since we don't have hand-verified per-call type labels).
3. Build the ordered symbolic sequence of call types per file.
4. Count immediate ABA triplets: positions (i, i+1, i+2) where type(i)==type(i+2)
   and type(i+1) is different -- the simplest, most standard operationalization of
   center-embedding used in animal sequence-learning literature.
5. Compare the observed ABA count against a null distribution built from many
   random permutations of the SAME symbol sequence (preserving type frequencies,
   randomizing order) to test whether ABA structure exceeds chance.
"""
import numpy as np
import pandas as pd
import librosa
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def extract_features(audio_path, calls_df, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    feats = []
    for _, row in calls_df.iterrows():
        s, e = row.onset, row.offset
        seg = y[int(s*sr):int(e*sr)]
        if len(seg) < 32:
            feats.append([0,0,0,0]); continue
        dur = e - s
        cent = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
        bw = librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean()
        rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr).mean()
        feats.append([dur, cent, bw, rolloff])
    feats = np.array(feats)
    return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)

def cluster_types(feats, k=3, seed=0):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(feats)
    return labels

def count_aba(sequence):
    count = 0
    for i in range(len(sequence)-2):
        if sequence[i] == sequence[i+2] and sequence[i] != sequence[i+1]:
            count += 1
    return count

def aba_surrogate_test(sequence, n_shuffles=2000, seed=0):
    rng = np.random.default_rng(seed)
    observed = count_aba(sequence)
    seq = np.array(sequence)
    null_counts = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(seq)
        null_counts.append(count_aba(shuffled))
    null_counts = np.array(null_counts)
    z = (observed - null_counts.mean()) / (null_counts.std() + 1e-9)
    p = (np.sum(null_counts >= observed) + 1) / (n_shuffles + 1)  # one-tailed empirical p
    return dict(observed=observed, null_mean=null_counts.mean(), null_std=null_counts.std(),
                z=z, p=p, n=len(sequence))

def analyze_file(name, audio_path, calls_df, k=3):
    calls_df = calls_df.sort_values("onset").reset_index(drop=True)
    if len(calls_df) < 10:
        return None
    feats = extract_features(audio_path, calls_df)
    labels = cluster_types(feats, k=k)
    result = aba_surrogate_test(labels)
    result['species'] = name
    result['k'] = k
    print(f"{name}: n={result['n']}, k={k}, observed_ABA={result['observed']}, "
          f"null_mean={result['null_mean']:.2f}, z={result['z']:.2f}, p={result['p']:.4f}")
    return result

def count_repeated_bigrams(sequence):
    """Tail-end embedding proxy: count of immediately-adjacent repeated bigrams,
    i.e., positions (i,i+1) forming the same ordered pair as (i+2,i+3) -- e.g. 'XY XY',
    analogous to the I-(V-I)-(V-I) repeated-unit tail structure in the reference figure."""
    count = 0
    for i in range(len(sequence)-3):
        if sequence[i] == sequence[i+2] and sequence[i+1] == sequence[i+3]:
            count += 1
    return count

def bigram_surrogate_test(sequence, n_shuffles=2000, seed=0):
    rng = np.random.default_rng(seed)
    observed = count_repeated_bigrams(sequence)
    seq = np.array(sequence)
    null_counts = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(seq)
        null_counts.append(count_repeated_bigrams(shuffled))
    null_counts = np.array(null_counts)
    z = (observed - null_counts.mean()) / (null_counts.std() + 1e-9)
    p = (np.sum(null_counts >= observed) + 1) / (n_shuffles + 1)
    return dict(observed=observed, null_mean=null_counts.mean(), null_std=null_counts.std(),
                z=z, p=p, n=len(sequence))

def analyze_file_full(name, audio_path, calls_df, k=3):
    calls_df = calls_df.sort_values("onset").reset_index(drop=True)
    if len(calls_df) < 10:
        return None
    feats = extract_features(audio_path, calls_df)
    labels = cluster_types(feats, k=k)
    aba = aba_surrogate_test(labels)
    tail = bigram_surrogate_test(labels)
    print(f"{name}: n={aba['n']}, k={k} | ABA: obs={aba['observed']}, z={aba['z']:.2f}, p={aba['p']:.4f} | "
          f"TailBigram: obs={tail['observed']}, z={tail['z']:.2f}, p={tail['p']:.4f}")
    return dict(species=name, k=k, n=aba['n'],
                aba_observed=aba['observed'], aba_z=aba['z'], aba_p=aba['p'],
                tail_observed=tail['observed'], tail_z=tail['z'], tail_p=tail['p'])

def anchor_gap_trend(sequence):
    """Left-boundary-embedding proxy: find the most frequent symbol (the 'anchor'/
    head-like element), compute the gaps between its successive occurrences, and
    test whether those gaps show a monotonically INCREASING trend -- the signature
    of progressively deeper nesting being built up before returning to the anchor,
    analogous to each new modifier pushing the head further away in
    [A [B [C D]]]-style left-embedding."""
    seq = np.array(sequence)
    vals, counts = np.unique(seq, return_counts=True)
    anchor = vals[np.argmax(counts)]
    positions = np.where(seq == anchor)[0]
    if len(positions) < 4:
        return None
    gaps = np.diff(positions)
    # Spearman correlation between gap order (1st, 2nd, 3rd... gap) and gap size
    from scipy.stats import spearmanr
    order = np.arange(len(gaps))
    rho, _ = spearmanr(order, gaps)
    return rho, len(gaps), anchor

def boundary_surrogate_test(sequence, n_shuffles=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = anchor_gap_trend(sequence)
    if obs is None:
        return None
    observed_rho, n_gaps, anchor = obs
    seq = np.array(sequence)
    null_rhos = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(seq)
        r = anchor_gap_trend(shuffled)
        if r is not None:
            null_rhos.append(r[0])
    null_rhos = np.array(null_rhos)
    z = (observed_rho - null_rhos.mean()) / (null_rhos.std() + 1e-9)
    p = (np.sum(null_rhos >= observed_rho) + 1) / (len(null_rhos) + 1)
    return dict(observed_rho=observed_rho, null_mean=null_rhos.mean(), null_std=null_rhos.std(),
                z=z, p=p, n_gaps=n_gaps)
