"""
analysis.py
-----------
Convergence analysis and statistical validation for Monte Carlo simulations.
Tests the Law of Large Numbers, Central Limit Theorem, and compares
empirical moments against theoretical values.

Author: Abdellah Kahlaoui
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from monte_carlo import MonteCarloEngine, SimulationResult


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ConvergenceResult:
    """Holds convergence study output for a single distribution."""
    distribution : str
    ns           : np.ndarray   # path counts
    means        : np.ndarray   # sample mean at each n
    std_errs     : np.ndarray   # standard error at each n
    true_mean    : float        # theoretical mean
    rate         : float        # empirical convergence rate (slope in log-log)


@dataclass
class ValidationResult:
    """Holds statistical validation output against theoretical moments."""
    distribution  : str
    n_paths       : int
    # Empirical vs theoretical
    empirical_mean     : float
    theoretical_mean   : float
    empirical_std      : float
    theoretical_std    : float
    empirical_skew     : float
    theoretical_skew   : float
    # Goodness-of-fit
    ks_statistic  : float
    ks_pvalue     : float
    ks_passed     : bool        # True if p-value > 0.05
    # CI coverage
    ci_covers_true: bool        # True if theoretical mean falls in 95% CI


# --------------------------------------------------------------------------- #
# Theoretical moments                                                          #
# --------------------------------------------------------------------------- #

THEORETICAL = {
    "Normal": {
        "mean"     : lambda p: p["mu"],
        "std"      : lambda p: p["sigma"],
        "skewness" : lambda p: 0.0,
        "scipy_dist": lambda p: stats.norm(loc=p["mu"], scale=p["sigma"]),
    },
    "Log-Normal": {
        "mean"     : lambda p: np.exp(p["mu"] + 0.5 * p["sigma"]**2),
        "std"      : lambda p: np.sqrt(
            (np.exp(p["sigma"]**2) - 1) * np.exp(2*p["mu"] + p["sigma"]**2)
        ),
        "skewness" : lambda p: (np.exp(p["sigma"]**2) + 2)
                               * np.sqrt(np.exp(p["sigma"]**2) - 1),
        "scipy_dist": lambda p: stats.lognorm(s=p["sigma"],
                                               scale=np.exp(p["mu"])),
    },
    "Exponential": {
        "mean"     : lambda p: 1.0 / p["lambda"],
        "std"      : lambda p: 1.0 / p["lambda"],
        "skewness" : lambda p: 2.0,
        "scipy_dist": lambda p: stats.expon(scale=1.0 / p["lambda"]),
    },
    "Poisson": {
        "mean"     : lambda p: p["lambda"],
        "std"      : lambda p: np.sqrt(p["lambda"]),
        "skewness" : lambda p: 1.0 / np.sqrt(p["lambda"]),
        "scipy_dist": lambda p: stats.poisson(mu=p["lambda"]),
    },
}


# --------------------------------------------------------------------------- #
# Analyser                                                                     #
# --------------------------------------------------------------------------- #

class MCAnalyser:
    """
    Statistical analyser for Monte Carlo simulation results.

    Parameters
    ----------
    engine : MonteCarloEngine
        A configured simulation engine instance.
    """

    def __init__(self, engine: MonteCarloEngine):
        self.engine = engine

    # ------------------------------------------------------------------ #
    # Convergence                                                          #
    # ------------------------------------------------------------------ #

    def convergence(
        self,
        dist    : str   = "normal",
        steps   : int   = 60,
        **kwargs,
    ) -> ConvergenceResult:
        """
        Study how the sample mean converges to the true mean as n grows.
        Fits a log-log slope to measure empirical convergence rate
        (should be close to -0.5 by the CLT).

        Parameters
        ----------
        dist   : one of 'normal', 'lognormal', 'exponential', 'poisson'
        steps  : number of log-spaced checkpoints
        kwargs : distribution parameters (e.g. mu=0, sigma=1)
        """
        ns, means, std_errs = self.engine.convergence_study(
            dist=dist, steps=steps, **kwargs
        )

        name_map  = {
            "normal": "Normal", "lognormal": "Log-Normal",
            "exponential": "Exponential", "poisson": "Poisson",
        }
        full_name = name_map[dist.lower()]
        # Use the final (largest n) mean as the true mean reference
        true_mean = float(means[-1])

        # Empirical convergence rate via log-log linear regression on std_errs
        log_ns   = np.log10(ns.astype(float))
        log_errs = np.log10(std_errs + 1e-12)
        slope, _ = np.polyfit(log_ns, log_errs, 1)

        return ConvergenceResult(
            distribution = full_name,
            ns           = ns,
            means        = means,
            std_errs     = std_errs,
            true_mean    = true_mean,
            rate         = slope,
        )

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self, result: SimulationResult) -> ValidationResult:
        """
        Validate a SimulationResult against theoretical moments and
        perform a Kolmogorov-Smirnov goodness-of-fit test.

        Parameters
        ----------
        result : SimulationResult from MonteCarloEngine
        """
        name    = result.distribution
        params  = result.params
        theory  = THEORETICAL[name]

        th_mean = theory["mean"](params)
        th_std  = theory["std"](params)
        th_skew = theory["skewness"](params)

        # Goodness-of-fit test
        # Poisson is discrete → chi-squared test; continuous → KS test
        rng       = np.random.default_rng(0)
        subsample = rng.choice(result.samples, size=10_000, replace=False)
        if name == "Poisson":
            # Chi-squared test on observed vs expected counts
            lam      = params["lambda"]
            max_k    = int(stats.poisson.ppf(0.9999, lam))
            bins     = np.arange(0, max_k + 2)
            observed, _ = np.histogram(subsample, bins=bins)
            expected = len(subsample) * stats.poisson.pmf(
                np.arange(0, max_k + 1), lam
            )
            # Merge bins with expected < 5
            mask     = expected >= 5
            observed = observed[mask]
            expected = expected[mask]
            # Normalize expected to match observed sum exactly
            expected = expected * (observed.sum() / expected.sum())
            chi2_stat, ks_pval = stats.chisquare(observed, expected)
            ks_stat  = chi2_stat
        else:
            scipy_dist       = theory["scipy_dist"](params)
            ks_stat, ks_pval = stats.kstest(subsample, scipy_dist.cdf)

        ci_covers = result.ci_95[0] <= th_mean <= result.ci_95[1]

        return ValidationResult(
            distribution       = name,
            n_paths            = result.n_paths,
            empirical_mean     = result.mean,
            theoretical_mean   = th_mean,
            empirical_std      = result.std,
            theoretical_std    = th_std,
            empirical_skew     = result.skewness,
            theoretical_skew   = th_skew,
            ks_statistic       = ks_stat,
            ks_pvalue          = ks_pval,
            ks_passed          = ks_pval > 0.05,
            ci_covers_true     = ci_covers,
        )

    # ------------------------------------------------------------------ #
    # Full report                                                          #
    # ------------------------------------------------------------------ #

    def full_report(self) -> list[ValidationResult]:
        """
        Run validation on all four distributions with default parameters
        and print a formatted summary table.
        """
        sims = [
            self.engine.simulate_normal(mu=0.0, sigma=1.0),
            self.engine.simulate_lognormal(mu=0.0, sigma=0.5),
            self.engine.simulate_exponential(lam=2.0),
            self.engine.simulate_poisson(lam=5.0),
        ]

        results = [self.validate(s) for s in sims]

        # Print table
        w = 62
        print("=" * w)
        print(f"{'STATISTICAL VALIDATION REPORT':^{w}}")
        print(f"{'n = {:,} paths per distribution'.format(self.engine.n_paths):^{w}}")
        print("=" * w)

        for r in results:
            print(f"\n  {r.distribution}")
            print(f"  {'Metric':<22} {'Empirical':>12} {'Theoretical':>12}")
            print(f"  {'-'*46}")
            print(f"  {'Mean':<22} {r.empirical_mean:>12.6f} {r.theoretical_mean:>12.6f}")
            print(f"  {'Std Dev':<22} {r.empirical_std:>12.6f} {r.theoretical_std:>12.6f}")
            print(f"  {'Skewness':<22} {r.empirical_skew:>12.4f} {r.theoretical_skew:>12.4f}")
            print(f"  {'KS statistic':<22} {r.ks_statistic:>12.6f}")
            print(f"  {'KS p-value':<22} {r.ks_pvalue:>12.4f}  "
                  f"{'PASS ✓' if r.ks_passed else 'FAIL ✗'}")
            print(f"  {'95% CI covers true mean':<22} "
                  f"{'Yes ✓' if r.ci_covers_true else 'No ✗':>12}")

        print("\n" + "=" * w)
        all_pass = all(r.ks_passed and r.ci_covers_true for r in results)
        print(f"  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
        print("=" * w)

        return results


# --------------------------------------------------------------------------- #
# Convergence rate summary                                                     #
# --------------------------------------------------------------------------- #

def convergence_summary(analyser: MCAnalyser) -> None:
    """Print convergence rates for all distributions."""
    configs = [
        ("normal",      {"mu": 0.0,  "sigma": 1.0}),
        ("lognormal",   {"mu": 0.0,  "sigma": 0.5}),
        ("exponential", {"lam": 2.0}),
        ("poisson",     {"lam": 5.0}),
    ]

    # Theoretical mean lookup uses "lambda" key; simulate uses "lam"
    theory_params = [
        {"mu": 0.0, "sigma": 1.0},
        {"mu": 0.0, "sigma": 0.5},
        {"lambda": 2.0},
        {"lambda": 5.0},
    ]

    print("\n" + "=" * 50)
    print(f"{'CONVERGENCE RATE ANALYSIS':^50}")
    print(f"{'(CLT predicts slope ≈ -0.50)':^50}")
    print("=" * 50)
    print(f"  {'Distribution':<18} {'True Mean':>10} {'Slope':>8}")
    print(f"  {'-'*38}")

    for (dist, kwargs), tparams in zip(configs, theory_params):
        cr = analyser.convergence(dist=dist, steps=60, **kwargs)
        name_map = {
            "normal": "Normal", "lognormal": "Log-Normal",
            "exponential": "Exponential", "poisson": "Poisson",
        }
        true_mean = THEORETICAL[name_map[dist]]["mean"](tparams)
        print(f"  {name_map[dist]:<18} {true_mean:>10.4f} {cr.rate:>8.4f}")

    print("=" * 50)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    engine   = MonteCarloEngine(n_paths=1_000_000, seed=42)
    analyser = MCAnalyser(engine)

    analyser.full_report()
    convergence_summary(analyser)
