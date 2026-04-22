import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from solver import PlotRequest


def _safe_expression_to_values(expression: str, xs: np.ndarray) -> np.ndarray:
    x = sp.symbols("x")
    expr = sp.sympify(expression)
    fn = sp.lambdify(x, expr, modules=["numpy"])
    ys = fn(xs)
    return np.array(ys, dtype=float)


def create_plot_images(plot_requests: list[PlotRequest]) -> list[Path]:
    paths: list[Path] = []

    for req in plot_requests:
        if req.x_max <= req.x_min:
            continue

        xs = np.linspace(req.x_min, req.x_max, 400)
        try:
            ys = _safe_expression_to_values(req.expression, xs)
        except Exception:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        ax.plot(xs, ys, linewidth=2)
        ax.set_title(req.title)
        ax.set_xlabel(req.x_label)
        ax.set_ylabel(req.y_label)
        ax.grid(True, alpha=0.3)

        out = Path(tempfile.gettempdir()) / f"plot_{abs(hash((req.title, req.expression)))}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)

    return paths
