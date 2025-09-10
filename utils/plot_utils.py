import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# Simple Plotting Functions for Sample Size
# -----------------------------------------

def plot_power_curve(control_rate, experimental_rate, alpha=0.05, max_n=5000):
    """
    Plot approximate power curve as a function of sample size per group.
    """
    n_values = np.arange(10, max_n, 10)
    effect_size = experimental_rate - control_rate
    # approximate power: simplified using normal approximation
    power_values = 1 - np.exp(-n_values * np.abs(effect_size))  # rough illustrative
    fig, ax = plt.subplots()
    ax.plot(n_values, power_values, label=f"Δ={effect_size:.2f}")
    ax.axhline(0.8, color="red", linestyle="--", label="Target power 80%")
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Power (approx.)")
    ax.set_title("Power Curve")
    ax.legend()
    return fig

def plot_event_counts(control_events, experimental_events):
    """
    Plot bar chart of expected events in control vs experimental groups.
    """
    labels = ["Control", "Experimental"]
    counts = [control_events, experimental_events]
    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=["blue", "orange"])
    ax.set_ylabel("Expected Events")
    ax.set_title("Expected Event Counts")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center")
    return fig


# -----------------------------------------
# Reusable Plotting Functions
# -----------------------------------------

def plot_survival_curve(time, survival, label="Survival"):
    fig, ax = plt.subplots()
    ax.step(time, survival, where="post", label=label)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    return fig

def plot_event_rates(endpoints, rates):
    fig, ax = plt.subplots()
    ax.bar(endpoints, rates)
    ax.set_ylabel("Event Rate (%)")
    ax.set_title("Expected Endpoint Event Rates")
    return fig

def plot_stopping_boundaries(interims, efficacy, futility):
    fig, ax = plt.subplots()
    ax.plot(interims, efficacy, 'g--', label="Efficacy Boundary")
    ax.plot(interims, futility, 'r--', label="Futility Boundary")
    ax.set_xlabel("Interim Analysis")
    ax.set_ylabel("Test Statistic")
    ax.legend()
    return fig

def plot_diversity_distribution(categories, values, title="Diversity Distribution"):
    fig, ax = plt.subplots()
    ax.bar(categories, values)
    ax.set_title(title)
    ax.set_ylabel("Proportion (%)")
    return fig
