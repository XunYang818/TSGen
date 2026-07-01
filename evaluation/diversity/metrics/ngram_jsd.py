"""Semantic pattern divergence (N-gram JSD)."""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def extract_ngrams(sequence: List[str], n: int) -> List[Tuple]:
    """
    Extract N-grams from sequence.

    Args:
        sequence: Furniture ID sequence
        n: N value for N-gram

    Returns:
        List of N-grams
    """
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


def ngram_distribution(sequence: List[str], n: int) -> Dict[Tuple, float]:
    """
    Compute N-gram probability distribution of sequence.

    Args:
        sequence: Furniture ID sequence
        n: N value for N-gram

    Returns:
        Dictionary mapping N-gram -> normalized probability
    """
    ngrams = extract_ngrams(sequence, n)
    if not ngrams:
        return {}

    counts = defaultdict(int)
    for ng in ngrams:
        counts[ng] += 1

    total = sum(counts.values())
    return {ng: count / total for ng, count in counts.items()}


def kl_divergence(p: Dict, q: Dict, epsilon: float = 1e-10) -> float:
    """Compute KL divergence D_KL(P || Q)."""
    all_keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for key in all_keys:
        p_val = p.get(key, epsilon)
        q_val = q.get(key, epsilon)
        if p_val > epsilon:
            kl += p_val * np.log(p_val / q_val)
    return kl


def js_divergence(p: Dict, q: Dict) -> float:
    """Compute Jensen-Shannon divergence."""
    if not p and not q:
        return 0.0
    if not p or not q:
        return np.log(2)

    all_keys = set(p.keys()) | set(q.keys())
    m = {key: 0.5 * (p.get(key, 0) + q.get(key, 0)) for key in all_keys}

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_ngram_jsd(seq1: List[str], seq2: List[str], n: int) -> float:
    """
    Compute N-gram JSD between two furniture sequences.

    Args:
        seq1: Furniture sequence 1
        seq2: Furniture sequence 2
        n: N value for N-gram (2, 3, 4)

    Returns:
        JSD value, higher means greater semantic pattern difference
    """
    dist1 = ngram_distribution(seq1, n)
    dist2 = ngram_distribution(seq2, n)
    return js_divergence(dist1, dist2)
