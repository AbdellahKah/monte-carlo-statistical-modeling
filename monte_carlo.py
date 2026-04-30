"""
monte_carlo.py
--------------
Vectorized Monte Carlo simulation engine for analyzing complex probability
distributions. Supports Normal, Log-Normal, Exponential, and Poisson
distributions with configurable number of paths and seeds.

Author: Abdellah Kahlaoui
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class SimulationResult:
    """Holds the output of a Monte Carlo simulation run."""
    distribution : str
    params       : dict
    n_paths      : int
    samples      : np.ndarray          # shape (n_paths,)
    mean         : float
    std          : float
    variance     : float
    skewness     : float
    kurtosis     : float
    ci_95        : tuple[float, float]  # 95% confidence interval on the mean


# --------------------------------------------------------------------------- #
# Core engine                                                                  #
# --------------------------------------------------------------------------- #

class MonteCarloEngine:
    """
    Vectorized Monte Carlo engine for probability distribution analysis.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths (default: 1_000_000).
    seed : int | None
        Random seed for reproducibility (default: 42).
    """

    def __init__(self, n_paths: int = 1_000_000, seed: int | None = 42):
        self.n_paths = n_paths
        self.rng     = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Samplers                                                             #
    # ------------------------------------------------------------------ #

    def simulate_normal(self, mu: float = 0.0, sigma: float = 1.0) -> SimulationResult:
        """
        Simulate a Normal distribution N(mu, sigma^2).

        Parameters
        ----------
        mu    : mean
        sigma : standard deviation (> 0)
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        samples = self.rng.normal(loc=mu, scale=sigma, size=self.n_paths)
        return self._build_result("Normal", {"mu": mu, "sigma": sigma}, samples)

    def simulate_lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> SimulationResult:
        """
        Simulate a Log-Normal distribution where log(X) ~ N(mu, sigma^2).
        Commonly used for asset price modelling.

        Parameters
        ----------
        mu    : mean of the underlying normal (log-space)
        sigma : std  of the underlying normal (log-space, > 0)
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        samples = self.rng.lognormal(mean=mu, sigma=sigma, size=self.n_paths)
        return self._build_result("Log-Normal", {"mu": mu, "sigma": sigma}, samples)

    def simulate_exponential(self, lam: float = 1.0) -> SimulationResult:
        """
        Simulate an Exponential distribution Exp(lambda).
        Models waiting times between independent events.

        Parameters
        ----------
        lam : rate parameter lambda (> 0)
        """
        if lam <= 0:
            raise ValueError("lam must be positive.")
        samples = self.rng.exponential(scale=1.0 / lam, size=self.n_paths)
        return self._build_result("Exponential", {"lambda": lam}, samples)

    def simulate_poisson(self, lam: float = 5.0) -> SimulationResult:
        """
        Simulate a Poisson distribution Poisson(lambda).
        Models counts of events in a fixed interval.

        Parameters
        ----------
        lam : expected number of events (> 0)
        """
        if lam <= 0:
            raise ValueError("lam must be positive.")
        samples = self.rng.poisson(lam=lam, size=self.n_paths).astype(float)
        return self._build_result("Poisson", {"lambda": lam}, samples)

    # ------------------------------------------------------------------ #
    # Option pricing via MC (Black-Scholes sanity check)                  #
    # ------------------------------------------------------------------ #

    def price_european_call(
        self,
        S0    : float = 100.0,
        K     : float = 100.0,
        T     : float = 1.0,
        r     : float = 0.05,
        sigma : float = 0.2,
    ) -> dict:
        """
        Price a European call option using Monte Carlo under Black-Scholes.

        Parameters
        ----------
        S0    : current stock price
        K     : strike price
        T     : time to maturity (years)
        r     : risk-free rate
        sigma : volatility

        Returns
        -------
        dict with keys: price, std_error, ci_95
        """
        Z       = self.rng.standard_normal(self.n_paths)
        ST      = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoffs = np.maximum(ST - K, 0.0)
        disc    = np.exp(-r * T)

        price    = disc * payoffs.mean()
        std_err  = disc * payoffs.std() / np.sqrt(self.n_paths)
        ci_95    = (price - 1.96 * std_err, price + 1.96 * std_err)

        return {"price": price, "std_error": std_err, "ci_95": ci_95}

    # ------------------------------------------------------------------ #
    # Convergence study                                                    #
    # ------------------------------------------------------------------ #

    def convergence_study(
        self,
        dist  : Literal["normal", "lognormal", "exponential", "poisson"] = "normal",
        steps : int = 50,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyse how the sample mean converges as n_paths grows.

        Returns
        -------
        ns       : array of path counts  (log-spaced from 100 to n_paths)
        means    : sample mean at each n
        std_errs : standard error of the mean at each n
        """
        ns       = np.logspace(2, np.log10(self.n_paths), steps, dtype=int)
        means    = np.empty(steps)
        std_errs = np.empty(steps)

        # Draw one large batch; slice it for each n
        full_result = getattr(self, f"simulate_{dist}")(**kwargs)
        samples     = full_result.samples

        for i, n in enumerate(ns):
            s            = samples[:n]
            means[i]     = s.mean()
            std_errs[i]  = s.std() / np.sqrt(n)

        return ns, means, std_errs

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_result(
        self, name: str, params: dict, samples: np.ndarray
    ) -> SimulationResult:
        """Compute descriptive statistics and return a SimulationResult."""
        n       = len(samples)
        mean    = samples.mean()
        std     = samples.std()
        var     = samples.var()

        # Skewness and excess kurtosis (Fisher)
        centred  = samples - mean
        skew     = (centred**3).mean() / (std**3 + 1e-12)
        kurt     = (centred**4).mean() / (std**4 + 1e-12) - 3.0

        # 95% CI on the mean (CLT)
        se       = std / np.sqrt(n)
        ci_95    = (mean - 1.96 * se, mean + 1.96 * se)

        return SimulationResult(
            distribution = name,
            params       = params,
            n_paths      = n,
            samples      = samples,
            mean         = mean,
            std          = std,
            variance     = var,
            skewness     = skew,
            kurtosis     = kurt,
            ci_95        = ci_95,
        )


# --------------------------------------------------------------------------- #
# Quick demo                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    engine = MonteCarloEngine(n_paths=1_000_000, seed=42)

    print("=" * 55)
    print(f"{'Monte Carlo Simulation Engine':^55}")
    print(f"{'Paths: 1,000,000  |  Seed: 42':^55}")
    print("=" * 55)

    for sim in [
        engine.simulate_normal(mu=0, sigma=1),
        engine.simulate_lognormal(mu=0, sigma=0.5),
        engine.simulate_exponential(lam=2.0),
        engine.simulate_poisson(lam=5.0),
    ]:
        print(f"\n{sim.distribution}  {sim.params}")
        print(f"  Mean      : {sim.mean:.6f}")
        print(f"  Std       : {sim.std:.6f}")
        print(f"  Skewness  : {sim.skewness:.4f}")
        print(f"  Kurtosis  : {sim.kurtosis:.4f}")
        print(f"  95% CI    : ({sim.ci_95[0]:.6f}, {sim.ci_95[1]:.6f})")

    print("\nEuropean Call (BS Monte Carlo):")
    call = engine.price_european_call(S0=100, K=100, T=1, r=0.05, sigma=0.2)
    print(f"  Price     : {call['price']:.4f}")
    print(f"  Std Error : {call['std_error']:.6f}")
    print(f"  95% CI    : ({call['ci_95'][0]:.4f}, {call['ci_95'][1]:.4f})")
    print("\nBS analytical price ≈ 10.4506  (Black-Scholes formula)")
