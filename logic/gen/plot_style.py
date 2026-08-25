import matplotlib.pyplot as plt

def apply_civic_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"], # Safe fallbacks
        "axes.facecolor": "#fbfbfb",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#bac4ce",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.labelcolor": "#4a5568",
        "xtick.color": "#4a5568",
        "ytick.color": "#4a5568",
        "text.color": "#1a202c",
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "scatter.edgecolors": "white",
        "scatter.linewidths": 1.0,
    })

def create_civic_colors():
    # Cartographic / Civic palette
    return {
        "brand-route": "#0066cc",
        "brand-eco": "#00a859",
        "brand-warn": "#f26522",
        "muted": "#4a5568",
        "grid": "#e2e8f0"
    }
