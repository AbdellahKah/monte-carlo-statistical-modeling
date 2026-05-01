"""
plots.py
--------
Interactive Plotly visualizations for Monte Carlo simulation results.
Generates distribution histograms, convergence plots, option pricing
convergence, and a full HTML report dashboard.

Author: Abdellah Kahlaoui
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

from monte_carlo import MonteCarloEngine
from analysis import MCAnalyser, THEORETICAL


# --------------------------------------------------------------------------- #
# Colour palette                                                               #
# --------------------------------------------------------------------------- #

COLOURS = {
    "Normal"      : "#4C72B0",
    "Log-Normal"  : "#DD8452",
    "Exponential" : "#55A868",
    "Poisson"     : "#C44E52",
    "neutral"     : "#8172B2",
    "ci"          : "rgba(100,160,230,0.20)",
}

TEMPLATE = "plotly_dark"


# --------------------------------------------------------------------------- #
# 1. Distribution histograms with theoretical PDF overlay                     #
# --------------------------------------------------------------------------- #

def plot_distributions(engine: MonteCarloEngine) -> go.Figure:
    """
    4-panel figure: histogram of simulated samples vs theoretical PDF/PMF
    for Normal, Log-Normal, Exponential, and Poisson distributions.
    """
    sims = [
        (engine.simulate_normal(mu=0.0, sigma=1.0),
         {"mu": 0.0, "sigma": 1.0}),
        (engine.simulate_lognormal(mu=0.0, sigma=0.5),
         {"mu": 0.0, "sigma": 0.5}),
        (engine.simulate_exponential(lam=2.0),
         {"lambda": 2.0}),
        (engine.simulate_poisson(lam=5.0),
         {"lambda": 5.0}),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[s.distribution for s, _ in sims],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (sim, params), (row, col) in zip(sims, positions):
        colour = COLOURS[sim.distribution]
        name   = sim.distribution

        # Subsample 50k for histogram speed
        rng       = np.random.default_rng(0)
        subsample = rng.choice(sim.samples, size=50_000, replace=False)

        # Histogram (normalised to density)
        fig.add_trace(
            go.Histogram(
                x           = subsample,
                histnorm    = "probability density",
                nbinsx      = 80,
                marker_color= colour,
                opacity     = 0.55,
                name        = f"{name} samples",
                showlegend  = False,
            ),
            row=row, col=col,
        )

        # Theoretical PDF overlay
        theory = THEORETICAL[name]
        if name == "Poisson":
            lam    = params["lambda"]
            ks     = np.arange(0, int(lam * 3 + 1))
            pmf    = stats.poisson.pmf(ks, lam)
            fig.add_trace(
                go.Bar(
                    x           = ks,
                    y           = pmf,
                    marker_color= colour,
                    opacity     = 0.9,
                    name        = f"{name} PMF",
                    showlegend  = False,
                ),
                row=row, col=col,
            )
        else:
            scipy_dist = theory["scipy_dist"](params)
            x_min, x_max = np.percentile(subsample, [0.1, 99.9])
            xs   = np.linspace(x_min, x_max, 400)
            pdf  = scipy_dist.pdf(xs)
            fig.add_trace(
                go.Scatter(
                    x           = xs,
                    y           = pdf,
                    mode        = "lines",
                    line        = dict(color=colour, width=2.5),
                    name        = f"{name} PDF",
                    showlegend  = False,
                ),
                row=row, col=col,
            )

        # Annotate mean & std
        fig.add_annotation(
            text=(f"μ={sim.mean:.3f}  σ={sim.std:.3f}<br>"
                  f"skew={sim.skewness:.3f}  kurt={sim.kurtosis:.3f}"),
            xref=f"x{'' if (row==1 and col==1) else (row-1)*2+col} domain",
            yref=f"y{'' if (row==1 and col==1) else (row-1)*2+col} domain",
            x=0.97, y=0.97,
            showarrow=False,
            align="right",
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.4)",
            borderpad=4,
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(
            text="<b>Monte Carlo Simulation — Distribution Analysis</b>"
                 f"<br><sup>n = {engine.n_paths:,} paths per distribution</sup>",
            x=0.5,
        ),
        template=TEMPLATE,
        height=620,
        bargap=0.05,
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. Convergence plot (mean ± CI vs number of paths)                          #
# --------------------------------------------------------------------------- #

def plot_convergence(analyser: MCAnalyser) -> go.Figure:
    """
    4-panel convergence plot: sample mean ± 95% CI vs n_paths (log scale).
    Illustrates the Law of Large Numbers and CLT error decay.
    """
    configs = [
        ("normal",      {"mu": 0.0, "sigma": 1.0},   0.0),
        ("lognormal",   {"mu": 0.0, "sigma": 0.5},   np.exp(0.0 + 0.5**2 / 2)),
        ("exponential", {"lam": 2.0},                 0.5),
        ("poisson",     {"lam": 5.0},                 5.0),
    ]

    dist_names = ["Normal", "Log-Normal", "Exponential", "Poisson"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=dist_names,
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for (dist, kwargs, true_mean), name, (row, col) in zip(
        configs, dist_names, positions
    ):
        colour = COLOURS[name]
        cr     = analyser.convergence(dist=dist, steps=60, **kwargs)
        ns, means, stds = cr.ns, cr.means, cr.std_errs
        ci_hi  = means + 1.96 * stds
        ci_lo  = means - 1.96 * stds

        # CI band
        fig.add_trace(
            go.Scatter(
                x    = np.concatenate([ns, ns[::-1]]),
                y    = np.concatenate([ci_hi, ci_lo[::-1]]),
                fill = "toself",
                fillcolor = COLOURS["ci"],
                line      = dict(width=0),
                showlegend= False,
                hoverinfo = "skip",
            ),
            row=row, col=col,
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x    = ns,
                y    = means,
                mode = "lines",
                line = dict(color=colour, width=2),
                name = name,
                showlegend=False,
            ),
            row=row, col=col,
        )
        # True mean dashed
        fig.add_trace(
            go.Scatter(
                x         = [ns[0], ns[-1]],
                y         = [true_mean, true_mean],
                mode      = "lines",
                line      = dict(color="white", width=1.2, dash="dot"),
                showlegend= False,
                hoverinfo = "skip",
            ),
            row=row, col=col,
        )
        # Annotate convergence rate
        fig.add_annotation(
            text=f"slope ≈ {cr.rate:.3f}<br>(CLT: −0.500)",
            xref=f"x{'' if (row==1 and col==1) else (row-1)*2+col} domain",
            yref=f"y{'' if (row==1 and col==1) else (row-1)*2+col} domain",
            x=0.05, y=0.97,
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.4)",
            borderpad=4,
            align="left",
            row=row, col=col,
        )

    # Log x-axis for all panels
    for i in range(1, 5):
        ax = "xaxis" if i == 1 else f"xaxis{i}"
        fig.layout[ax].type = "log"
        fig.layout[ax].title = "Number of paths"

    fig.update_layout(
        title=dict(
            text="<b>Monte Carlo Convergence Analysis</b>"
                 "<br><sup>Sample mean ± 95% CI vs number of paths (log scale)</sup>",
            x=0.5,
        ),
        template=TEMPLATE,
        height=620,
    )
    return fig


# --------------------------------------------------------------------------- #
# 3. European Call option price convergence                                   #
# --------------------------------------------------------------------------- #

def plot_option_convergence(engine: MonteCarloEngine) -> go.Figure:
    """
    Show how the MC option price converges to the Black-Scholes
    analytical value as n_paths grows.
    """
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # Black-Scholes analytical price
    d1  = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    bs  = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    ns      = np.logspace(2, np.log10(engine.n_paths), 60, dtype=int)
    prices  = []
    ci_hi   = []
    ci_lo   = []

    rng = np.random.default_rng(42)
    Z_full = rng.standard_normal(engine.n_paths)

    for n in ns:
        Z       = Z_full[:n]
        ST      = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoffs = np.maximum(ST - K, 0.0)
        disc    = np.exp(-r * T)
        p       = disc * payoffs.mean()
        se      = disc * payoffs.std() / np.sqrt(n)
        prices.append(p)
        ci_hi.append(p + 1.96 * se)
        ci_lo.append(p - 1.96 * se)

    prices = np.array(prices)
    ci_hi  = np.array(ci_hi)
    ci_lo  = np.array(ci_lo)

    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ns, ns[::-1]]),
        y=np.concatenate([ci_hi, ci_lo[::-1]]),
        fill="toself", fillcolor=COLOURS["ci"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # MC price
    fig.add_trace(go.Scatter(
        x=ns, y=prices, mode="lines",
        line=dict(color=COLOURS["neutral"], width=2),
        name="MC Price",
    ))
    # BS analytical
    fig.add_trace(go.Scatter(
        x=[ns[0], ns[-1]], y=[bs, bs],
        mode="lines", line=dict(color="#FFD700", width=1.5, dash="dot"),
        name=f"Black-Scholes = {bs:.4f}",
    ))

    fig.update_layout(
        title=dict(
            text="<b>European Call Option — MC Price Convergence</b>"
                 f"<br><sup>S₀=100, K=100, T=1, r=5%, σ=20% | "
                 f"BS analytical = {bs:.4f}</sup>",
            x=0.5,
        ),
        xaxis=dict(type="log", title="Number of paths"),
        yaxis=dict(title="Option Price"),
        template=TEMPLATE,
        height=450,
        legend=dict(x=0.75, y=0.15),
    )
    return fig


# --------------------------------------------------------------------------- #
# 4. Full HTML dashboard                                                       #
# --------------------------------------------------------------------------- #

def save_dashboard(engine: MonteCarloEngine, analyser: MCAnalyser,
                   path: str = "dashboard.html") -> None:
    """
    Combine all plots into a single self-contained HTML dashboard.

    Parameters
    ----------
    path : output file path
    """
    import plotly.io as pio

    f1 = plot_distributions(engine)
    f2 = plot_convergence(analyser)
    f3 = plot_option_convergence(engine)

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Monte Carlo Simulation Dashboard</title>",
        "<style>body{background:#111;font-family:sans-serif;color:#eee;"
        "margin:0;padding:20px}"
        "h1{text-align:center;letter-spacing:2px;font-size:1.4rem;"
        "color:#aac4ff;margin-bottom:30px}"
        ".section{margin-bottom:40px}</style></head><body>",
        "<h1>Monte Carlo Simulation &amp; Statistical Modeling — Dashboard</h1>",
        "<div class='section'>",
        pio.to_html(f1, full_html=False, include_plotlyjs="cdn"),
        "</div><div class='section'>",
        pio.to_html(f2, full_html=False, include_plotlyjs=False),
        "</div><div class='section'>",
        pio.to_html(f3, full_html=False, include_plotlyjs=False),
        "</div></body></html>",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"Dashboard saved → {path}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Initialising engine (1,000,000 paths)...")
    engine   = MonteCarloEngine(n_paths=1_000_000, seed=42)
    analyser = MCAnalyser(engine)

    print("Generating distribution plots...")
    plot_distributions(engine).write_html("dist_plot.html")

    print("Generating convergence plots...")
    plot_convergence(analyser).write_html("convergence_plot.html")

    print("Generating option convergence plot...")
    plot_option_convergence(engine).write_html("option_plot.html")

    print("Building full dashboard...")
    save_dashboard(engine, analyser, path="dashboard.html")

    print("\nAll plots saved:")
    print("  dist_plot.html")
    print("  convergence_plot.html")
    print("  option_plot.html")
    print("  dashboard.html  ← open this in your browser")
