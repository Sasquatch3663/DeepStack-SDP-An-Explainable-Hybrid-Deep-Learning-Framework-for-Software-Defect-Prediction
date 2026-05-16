"""
statistical_tests.py

Performs statistical validation.
"""

from scipy.stats import ttest_rel, wilcoxon


def statistical_comparison(scores1, scores2):
    results = {}

    try:
        _, t_p = ttest_rel(scores1, scores2)
        results["t_test_p_value"] = t_p
    except:
        results["t_test_p_value"] = None

    try:
        _, w_p = wilcoxon(scores1, scores2)
        results["wilcoxon_p_value"] = w_p
    except:
        results["wilcoxon_p_value"] = None

    return results