"""Sequence edit distance (Levenshtein Distance)."""
from typing import List, Tuple


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two sequences.

    Args:
        seq1: Sequence 1
        seq2: Sequence 2

    Returns:
        Edit distance (minimum number of operations)
    """
    m, n = len(seq1), len(seq2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1]   # replace
                )

    return dp[m][n]


def compute_edit_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, float]:
    """
    Compute edit distance (raw and normalized).

    Args:
        seq1: Furniture sequence 1
        seq2: Furniture sequence 2

    Returns:
        (raw edit distance, normalized edit distance)
        Normalized edit distance range [0, 1], higher means greater sequence difference
    """
    if not seq1 and not seq2:
        return 0, 0.0

    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0, 0.0

    raw_dist = levenshtein_distance(seq1, seq2)
    norm_dist = raw_dist / max_len
    return raw_dist, norm_dist
