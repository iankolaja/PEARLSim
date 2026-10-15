"""Family-grouped splits: a sequence family lies entirely in one of train, validation or test."""

import numpy as np


def choose_test_families(families, fraction=0.2, seed=0):
    """A reproducible random set of test families (record the result in the experiment config)."""
    unique = np.unique(families)
    count = max(1, int(round(fraction * len(unique))))
    return sorted(np.random.default_rng(seed).choice(unique, count, replace=False).tolist())


def grouped_folds(families, k=5, seed=0):
    """k (fit, validation) family lists covering the given families, each family validated once."""
    unique = np.unique(families)
    order = np.random.default_rng(seed).permutation(unique)
    folds = [order[j::k].tolist() for j in range(k)]
    return [(sorted(set(unique) - set(val)), sorted(val)) for val in folds if val]


def masks(window_families, *family_lists):
    """Boolean window masks, one per family list; raises if the lists overlap."""
    sets = [set(f) for f in family_lists]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"families in more than one split: {sorted(overlap)[:5]}")
    return [np.isin(window_families, list(s)) for s in sets]
