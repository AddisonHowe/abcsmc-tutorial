import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .abcsmc import sample_from_priors

"""
Plotting Functions

"""

def plot_results(particle_history, weight_history, score_history, 
                 acceptance_rates, epsilon_history, prior_list, **kwargs):
    """
    Plot results of the ABC-SMC function.

    Args:
        ...

    """
    pidx1 = kwargs.get("pidx1", 0)
    pidx2 = kwargs.get("pidx2", 1)
    pname1 = kwargs.get("pname1", "$\\theta_1$")
    pname2 = kwargs.get("pname2", "$\\theta_2$")
    save = kwargs.get("save", False)
    imgdir = kwargs.get("imgdir", "figures")

    empirical_prior = sample_from_priors(prior_list, 1000)
    for iteridx in range(len(particle_history)):
        particles = particle_history[iteridx]
        nrows = 1
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        ax1, ax2, ax3 = axes

        ax1.scatter(
            empirical_prior[:,pidx1], empirical_prior[:,pidx2],
            s=1, c='y', alpha=0.15
        )
        sc = ax1.scatter(
            particles[:,pidx1], particles[:,pidx2],
            s=2, 
            c=weight_history[iteridx],
            cmap='jet_r',
            norm=LogNorm()
        )
        fig.colorbar(sc, ax=ax1)
        ax1.set_xlabel(pname1)
        ax1.set_ylabel(pname2)
        ax1.set_title("Posterior vs Prior")

        sc = ax2.scatter(
            particles[:,pidx1], particles[:,pidx2],
            s=2, 
            c=weight_history[iteridx],
            cmap='jet_r',
            norm=LogNorm()
        )
        fig.colorbar(sc, ax=ax2)
        ax2.set_xlabel(pname1)
        ax2.set_ylabel(pname2)
        ax2.set_title("Posterior by weights")

        sc = ax3.scatter(
            particles[:,pidx1], particles[:,pidx2],
            s=2, 
            c=score_history[iteridx],
            cmap='jet_r',
            norm=LogNorm()
        )
        fig.colorbar(sc, ax=ax3)
        ax3.set_xlabel(pname1)
        ax3.set_ylabel(pname2)
        ax3.set_title("Posterior by score")

        fig.suptitle(f"Iteration {iteridx}")

        if save:
            plt.savefig(f"{imgdir}/posterior_{iteridx}.png")

    ns = range(len(epsilon_history))
    
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.plot(ns, epsilon_history, '.-', linewidth=3, markersize=12)
    ax.set_xticks(ns)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$\epsilon$");
    ax.set_title("$\epsilon$ schedule")
    if save:
        plt.savefig(f"{imgdir}/epsilon_sched.png")

    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.plot(ns, acceptance_rates, '.-', linewidth=3, markersize=12)
    ax.set_xticks(ns)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$\\alpha$")
    ax.set_title("Acceptance rate")
    if save:
        plt.savefig(f"{imgdir}/acceptance_rate.png")
