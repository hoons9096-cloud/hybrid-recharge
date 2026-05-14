"""Unified publication-quality matplotlib style for paper figures.

Apply at the top of every figure-generating script:

    from evaluation.figure_style import apply_publication_style
    apply_publication_style()

Or, when the script is run directly without the package import path:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from figure_style import apply_publication_style
    apply_publication_style()
"""
from __future__ import annotations

import matplotlib


# DejaVu Sans is primary because it ships with Matplotlib and has full Unicode
# coverage (incl. superscript minus used in "mm yr⁻¹"). Arial / Helvetica are
# kept as fallbacks for systems that prefer them, but DejaVu's glyphs render
# nearly identically to Arial in journal templates.
_FONT_FAMILY = ["DejaVu Sans", "Arial", "Helvetica"]


def apply_publication_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _FONT_FAMILY,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
        "figure.dpi": 100,
    })
