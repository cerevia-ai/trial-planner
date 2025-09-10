# utils/stats_utils.py
import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower

# -----------------------------------------
# Sample Size & Power
# -----------------------------------------

def cohen_h(p1, p2):
    """
    Compute Cohen's h for two proportions.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def sample_size_proportions(p1, p2, alpha=0.05, power=0.8, ratio=1.0):
    """
    Two-arm sample size calculation for proportions (per group).
    p1: control event rate
    p2: treatment event rate
    """
    analysis = NormalIndPower()
    h = cohen_h(p1, p2)
    n = analysis.solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio, alternative="two-sided")
    return int(np.ceil(n))  # per group sample size


def hazard_to_event_rate(hazard, follow_up_years, dropout_rate=0.0):
    """
    Convert hazard rate to cumulative event probability over time.
    """
    t_eff = follow_up_years * (1 - dropout_rate)
    return float(1 - np.exp(-hazard * t_eff))


def required_follow_up(event_rate, required_events, sample_size):
    """
    Estimate follow-up duration needed to accumulate required events.
    """
    if sample_size * event_rate == 0:
        return np.inf
    return required_events / (sample_size * event_rate)


# -----------------------------------------
# Statistical Tests
# -----------------------------------------

def z_test_proportions(p1, p2, n1, n2, alpha=0.05):
    """
    Z-test for two proportions.
    Returns: z-statistic, p-value
    """
    p_pool = (p1*n1 + p2*n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p
