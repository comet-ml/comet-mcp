#!/usr/bin/env python3
"""
Example metrics file for ez-mcp-optimize.

This file shows how to create custom metrics that work with the optimizer.
Copy this file and modify it to create your own metrics.
"""

from opik.evaluation.metrics.score_result import ScoreResult
from Levenshtein import distance as levenshtein_distance

def levenshtein_ratio_metric(dataset_item, output):
    # Based on dataset:
    reference = dataset_item["answer"]

    distance = levenshtein_distance(reference, output)
    max_len = max(len(reference), len(output))
    if max_len == 0:
        ratio = 1.0
        reason = "The output is identical to what is expected"
    else:
        ratio = 1.0 - (distance / max_len)
        reason = "There are some differences in what is expected"

    return ScoreResult(
        name="levenshtein_ratio_metric",
        reason=reason,
        value=ratio,
        metadata=None,
        scoring_failed=False
    )
