import math

import pytest

from experiments.analyze_adaptation import (
    analyze_pairs,
    effect_size_bin,
    minimum_detectable_effect,
)


def _pairs(adapted, retrained, old=None):
    old = old or [0.4] * len(adapted)
    return [
        {
            "seed": seed,
            "old_policy_accuracy": old_value,
            "adapted_policy_accuracy": adapted_value,
            "full_retrain_policy_accuracy": retrained_value,
        }
        for seed, (old_value, adapted_value, retrained_value) in enumerate(
            zip(old, adapted, retrained)
        )
    ]


def test_paired_rule_detects_clear_full_retrain_advantage():
    analysis = analyze_pairs(
        _pairs(
            [0.60, 0.62, 0.58, 0.61, 0.59],
            [0.80, 0.79, 0.82, 0.81, 0.78],
        )
    )
    assert analysis["decision"] == "full retrain meaningfully better"
    assert analysis["mean_paired_difference"] > analysis["tau"]
    assert analysis["absolute_effect_size_bin"] == "large"


def test_identical_pairs_are_equivalent_and_zero_recovery_is_guarded():
    analysis = analyze_pairs(
        _pairs([0.8, 0.8, 0.8], [0.8, 0.8, 0.8], old=[0.8, 0.8, 0.8])
    )
    assert analysis["decision"] == "adaptation statistically equivalent to full retrain"
    assert analysis["recovery_ratio"] is None
    assert analysis["cohens_d_paired"] is None


def test_recovery_ratio_and_effect_bins():
    analysis = analyze_pairs(
        _pairs([0.6, 0.7, 0.8], [0.8, 0.9, 1.0], old=[0.4, 0.5, 0.6])
    )
    assert analysis["recovery_ratio"] == pytest.approx(0.5)
    assert effect_size_bin(0.19) == "negligible"
    assert effect_size_bin(0.2) == "small"
    assert effect_size_bin(0.5) == "medium"
    assert effect_size_bin(0.8) == "large"


def test_paired_t_mde_decreases_with_more_seeds():
    small = minimum_detectable_effect(10, observed_std=0.1)
    large = minimum_detectable_effect(30, observed_std=0.1)
    assert math.isfinite(small["standardized"])
    assert large["standardized"] < small["standardized"]
    assert large["accuracy"] < small["accuracy"]
