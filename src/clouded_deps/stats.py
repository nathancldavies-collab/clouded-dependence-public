"""
Bootstrap confidence intervals shared by the analyses.

References:
  Efron, B. (1987). Better Bootstrap Confidence Intervals. JASA 82, 171-185.
  DiCiccio, T. J. & Efron, B. (1996). Bootstrap Confidence Intervals.
    Statistical Science 11, 189-228.
"""

import warnings

import numpy as np
from scipy.stats import norm

# Fewest bootstrap draws beyond a BCa endpoint before it is flagged as unstable.
MIN_TAIL_DRAWS = 10


def bca_interval(
    estimate: float, draws: np.ndarray, jackknife: np.ndarray, level: float
) -> tuple[float, float]:
    """
    BCa interval (Efron 1987; DiCiccio & Efron 1996, eqs. 2.3-2.8).

    z0 corrects median bias: the share of draws below the estimate, on the
    normal scale (ties count half). a corrects for the standard error changing
    with the parameter, estimated from the jackknife's skewness. The interval is
    then read off the bootstrap draws at the adjusted percentiles.
    """
    below = np.mean(draws < estimate) + 0.5 * np.mean(draws == estimate)
    # A share of exactly 0 or 1 would make z0 infinite; clip to one draw's worth.
    below = np.clip(below, 1 / len(draws), 1 - 1 / len(draws))
    z0 = norm.ppf(below)

    deviations = jackknife.mean() - jackknife
    spread = (deviations**2).sum()
    a = (deviations**3).sum() / (6 * spread**1.5) if spread > 0 else 0.0

    tail = (1 - level) / 2
    percentiles = []
    for z_alpha in norm.ppf([tail, 1 - tail]):
        shifted = z0 + z_alpha
        percentiles.append(norm.cdf(z0 + shifted / (1 - a * shifted)) * 100)
    # With a large z0 the adjusted percentiles run far into the tails, where an
    # endpoint rests on a handful of draws (DiCiccio & Efron 1996, sec. 7).
    tail_draws = min(percentiles[0], 100 - percentiles[1]) / 100 * len(draws)
    if tail_draws < MIN_TAIL_DRAWS:
        warnings.warn(
            f"BCa endpoint rests on {tail_draws:.1f} draws (z0={z0:+.2f}, a={a:+.3f});"
            " raise --n-boot for a stable interval.",
            stacklevel=2,
        )
    low, high = np.percentile(draws, percentiles)
    return float(low), float(high)
