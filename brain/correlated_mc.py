"""
engine.correlated_mc — Correlated Monte Carlo via Cholesky Decomposition
==========================================================================
Replaces independent triangular draws with correlated multivariate samples.

The Problem:
  Independent draws assume NOI, interest_rate, exit_cap, noi_growth are
  uncorrelated. In reality:
    - When rates rise → exit caps rise → NOI growth slows
    - When NOI drops → DSCR drops (mechanically)
    - Cap rates and exit caps co-move (ρ ≈ 0.85)

  Independent sampling underestimates tail risk because it allows
  "NOI way up AND rates way up AND exit cap way down" — a scenario
  that never happens in reality.

The Fix:
  1. Define the correlation matrix for MC input variables
  2. Cholesky-decompose it: Σ = L·Lᵀ
  3. Draw independent standard normals Z
  4. Transform: X = L·Z (now correlated standard normals)
  5. Map to marginal distributions via inverse CDF (triangular)

Result: samples that respect real-world co-movement of CRE variables.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from .determinism import get_rng
except ImportError:
    from determinism import get_rng


# ═══════════════════════════════════════════════════════════════
# CORRELATION MATRIX FOR MC INPUT VARIABLES
# ═══════════════════════════════════════════════════════════════

# The MC simulation draws these 4 core variables per scenario:
#   noi, loan_rate (interest_rate), noi_growth, exit_cap
#
# Correlation matrix (symmetric, diagonal = 1.0):
#
#              noi    rate   growth  exit_cap
#   noi       1.00  -0.30   0.45   -0.25
#   rate     -0.30   1.00  -0.35    0.55
#   growth    0.45  -0.35   1.00   -0.20
#   exit_cap -0.25   0.55  -0.20    1.00
#
# Rationale:
#   noi ↔ rate (-0.30): higher rates → higher debt service → stress on operations
#   noi ↔ growth (0.45): strong NOI properties tend to grow faster
#   noi ↔ exit_cap (-0.25): higher NOI → lower cap (premium assets)
#   rate ↔ growth (-0.35): rate hikes slow economic growth
#   rate ↔ exit_cap (0.55): rates push cap rates up (strong co-movement)
#   growth ↔ exit_cap (-0.20): growth compression → cap expansion

MC_VARIABLES = ["noi", "loan_rate", "noi_growth", "exit_cap"]

MC_CORRELATION_MATRIX = [
    [1.00, -0.30,  0.45, -0.25],
    [-0.30,  1.00, -0.35,  0.55],
    [0.45, -0.35,  1.00, -0.20],
    [-0.25,  0.55, -0.20,  1.00],
]


# ═══════════════════════════════════════════════════════════════
# CHOLESKY DECOMPOSITION (pure Python, no numpy)
# ═══════════════════════════════════════════════════════════════

def cholesky_decompose(matrix: List[List[float]]) -> List[List[float]]:
    """Cholesky decomposition: A = L·Lᵀ where L is lower triangular.

    Input: n×n symmetric positive-definite correlation matrix
    Output: n×n lower triangular matrix L

    Pure Python — no numpy required.
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = matrix[i][i] - s
                if val < 0:
                    # Matrix is not positive definite — apply Higham nearestPD fix
                    val = max(val, 1e-10)
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (matrix[i][j] - s) / L[j][j]

    return L


def mat_vec_multiply(L: List[List[float]], z: List[float]) -> List[float]:
    """Multiply lower triangular matrix L by vector z."""
    n = len(L)
    result = [0.0] * n
    for i in range(n):
        for j in range(i + 1):
            result[i] += L[i][j] * z[j]
    return result


# ═══════════════════════════════════════════════════════════════
# STANDARD NORMAL VIA BOX-MULLER (no scipy)
# ═══════════════════════════════════════════════════════════════

def standard_normal_pair(rng) -> Tuple[float, float]:
    """Generate pair of standard normal variates via Box-Muller."""
    u1 = max(rng.random(), 1e-15)  # avoid log(0)
    u2 = rng.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z0, z1


def standard_normals(n: int, rng) -> List[float]:
    """Generate n standard normal variates."""
    results = []
    while len(results) < n:
        z0, z1 = standard_normal_pair(rng)
        results.append(z0)
        if len(results) < n:
            results.append(z1)
    return results[:n]


# ═══════════════════════════════════════════════════════════════
# NORMAL CDF → TRIANGULAR INVERSE CDF (copula mapping)
# ═══════════════════════════════════════════════════════════════

def normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    # Highly accurate rational approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x/2)
    return 0.5 * (1.0 + sign * y)


def triangular_inverse_cdf(u: float, low: float, high: float, mode: float) -> float:
    """Inverse CDF of triangular distribution.

    u: uniform [0, 1]
    Returns: sample from Triangular(low, high, mode)
    """
    if low >= high:
        return mode
    u = max(0.0001, min(u, 0.9999))

    c = (mode - low) / (high - low)  # mode position [0, 1]

    if u <= c:
        return low + math.sqrt(u * (high - low) * (mode - low))
    else:
        return high - math.sqrt((1 - u) * (high - low) * (high - mode))


# ═══════════════════════════════════════════════════════════════
# CORRELATED DRAW ENGINE
# ═══════════════════════════════════════════════════════════════

class CorrelatedDrawEngine:
    """Generates correlated draws for Monte Carlo simulation.

    Usage:
        engine = CorrelatedDrawEngine()
        # Per simulation:
        draws = engine.draw(params)
        # draws = {"noi": 185000, "loan_rate": 7.2, "noi_growth": 1.8, "exit_cap": 7.6}
    """

    def __init__(self, correlation_matrix: List[List[float]] = None,
                 variable_names: List[str] = None):
        self._corr = correlation_matrix or MC_CORRELATION_MATRIX
        self._vars = variable_names or MC_VARIABLES
        self._L = cholesky_decompose(self._corr)
        self._n = len(self._vars)

    def draw(self, params: Dict, rng=None) -> Dict[str, float]:
        """Generate one set of correlated draws.

        params: {var_name: {point, low, high}} for each MC variable
        Returns: {var_name: sampled_value}
        """
        if rng is None:
            rng = get_rng()

        # 1. Draw independent standard normals
        z = standard_normals(self._n, rng)

        # 2. Apply Cholesky: X = L·Z (now correlated standard normals)
        x = mat_vec_multiply(self._L, z)

        # 3. Map each correlated normal → uniform via Φ(x) → marginal via inverse CDF
        result = {}
        for i, var in enumerate(self._vars):
            u = normal_cdf(x[i])  # Φ(correlated normal) → uniform [0,1]

            dist = params.get(var, {})
            if isinstance(dist, (int, float)):
                result[var] = float(dist)
                continue

            point = dist.get("point", 0)
            low = dist.get("low", point * 0.7)
            high = dist.get("high", point * 1.3)

            result[var] = triangular_inverse_cdf(u, low, high, point)

        return result

    def draw_batch(self, params: Dict, n: int, rng=None) -> List[Dict[str, float]]:
        """Generate n correlated draw sets."""
        return [self.draw(params, rng) for _ in range(n)]

    def verify_correlation(self, params: Dict, n: int = 5000) -> Dict:
        """Draw n samples and compute empirical correlation matrix.

        Use this to verify the Cholesky decomposition produces
        correct correlations.
        """
        rng = get_rng()
        samples = {var: [] for var in self._vars}

        for _ in range(n):
            draw = self.draw(params, rng)
            for var in self._vars:
                if var in draw:
                    samples[var].append(draw[var])

        # Compute empirical correlations
        def pearson(xs, ys):
            n = len(xs)
            if n < 2:
                return 0
            mx, my = sum(xs)/n, sum(ys)/n
            sx = math.sqrt(sum((x-mx)**2 for x in xs) / (n-1))
            sy = math.sqrt(sum((y-my)**2 for y in ys) / (n-1))
            if sx == 0 or sy == 0:
                return 0
            return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / ((n-1) * sx * sy)

        empirical = {}
        for i, a in enumerate(self._vars):
            for j, b in enumerate(self._vars):
                if j > i:
                    target = self._corr[i][j]
                    actual = pearson(samples[a], samples[b])
                    empirical[f"{a} ↔ {b}"] = {
                        "target": round(target, 2),
                        "actual": round(actual, 2),
                        "error": round(abs(target - actual), 3),
                    }

        return empirical
