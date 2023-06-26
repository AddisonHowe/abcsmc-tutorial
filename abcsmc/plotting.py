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

    #~~~~~ Process kwargs ~~~~~#
    pidx1 = kwargs.get("pidx1", 0)
    pidx2 = kwargs.get("pidx2", 1)
    pname1 = kwargs.get("pname1", "$\\theta_1$")
    pname2 = kwargs.get("pname2", "$\\theta_2$")
    true_param = kwargs.get("true_param", None)
    size = kwargs.get('size', 2)
    true_param_size = kwargs.get("true_param_size", 8)
    save = kwargs.get("save", False)
    imgdir = kwargs.get("imgdir", "figures")
    weight_norm = kwargs.get("weight_norm", LogNorm)
    score_norm = kwargs.get('score_norm', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
            s=size, 
            c=weight_history[iteridx],
            cmap='jet_r',
            norm=None if weight_norm is None else weight_norm()
        )
        fig.colorbar(sc, ax=ax1)
        ax1.set_xlabel(pname1)
        ax1.set_ylabel(pname2)
        ax1.set_title("Posterior vs Prior")

        sc = ax2.scatter(
            particles[:,pidx1], particles[:,pidx2],
            s=size, 
            c=weight_history[iteridx],
            cmap='jet_r',
            norm=None if weight_norm is None else weight_norm()
        )
        fig.colorbar(sc, ax=ax2)
        ax2.set_xlabel(pname1)
        ax2.set_ylabel(pname2)
        ax2.set_title("Posterior by weights")
        if true_param is not None:
            s = f"({','.join([pname1, pname2])}) " + \
                f"$=$ ({true_param[0]:.2g},{true_param[1]:.2g})"
            ax2.plot(true_param[pidx1],true_param[pidx2], 
                     'k*', label=s, markersize=true_param_size)
            ax2.legend()

        sc = ax3.scatter(
            particles[:,pidx1], particles[:,pidx2],
            s=size, 
            c=score_history[iteridx],
            cmap='jet_r',
            norm=None if score_norm is None else score_norm()
        )
        fig.colorbar(sc, ax=ax3)
        ax3.set_xlabel(pname1)
        ax3.set_ylabel(pname2)
        ax3.set_title("Posterior by score")
        if true_param is not None:
            s = f"({','.join([pname1, pname2])}) " + \
                f"$=$ ({true_param[0]:.2g},{true_param[1]:.2g})"
            ax3.plot(true_param[pidx1],true_param[pidx2], 
                     'k*', label=s, markersize=true_param_size)
            ax3.legend()

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


def plot_perturbation_sample(iteridx0, particle_history, sampling_idx_history, 
                             particle_idx_history, **kwargs):
    """
    """
    #~~~~~ Process kwargs ~~~~~#
    samp_idxs_to_plot = kwargs.get('samp_idxs_to_plot', None)
    col0 = kwargs.get('col0', 'b')
    col1 = kwargs.get('col1', 'r')
    size0 = kwargs.get('size0', 24)
    size1 = kwargs.get('size1', 12)
    linealpha = kwargs.get('linealpha', 0.25)
    save = kwargs.get('save', True)
    saveas = kwargs.get('saveas', "perturbation_history.png")
    imgdir = kwargs.get('imgdir', "figures")
    nsamp = kwargs.get('nsamp', 10)
    pidx1 = kwargs.get("pidx1", 0)
    pidx2 = kwargs.get("pidx2", 1)
    pname1 = kwargs.get("pname1", "$\\theta_1$")
    pname2 = kwargs.get("pname2", "$\\theta_2$")
    plot_background = kwargs.get('plot_background', True)
    background_alpha = kwargs.get('background_alpha', 0.2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    assert iteridx0 < len(particle_history) - 1, \
        "iteridx0={} invalid. Must be between 0 and {}".format(
        iteridx0, len(particle_history) - 1)

    iteridx1 = iteridx0 + 1

    nsamp = min(nsamp, len(np.unique(sampling_idx_history[iteridx1])))
    if samp_idxs_to_plot is None:
        samp_idxs_to_plot = np.random.choice(
            np.unique(sampling_idx_history[iteridx1]), nsamp, replace=False
        )
    else:
        samp_idxs_to_plot = np.array(samp_idxs_to_plot, dtype=int)

    # Get starting particles
    particles0 = particle_history[iteridx0, samp_idxs_to_plot]

    # Get all ending particles
    particles1 = particle_history[iteridx1]
    samp_idxs1 = sampling_idx_history[iteridx1]
    sdict = {}
    for sidx in samp_idxs_to_plot:
        sdict[sidx] = particles1[samp_idxs1 == sidx]
    
    fig, ax = plt.subplots(1, 1)

    if plot_background:
        ax.scatter(
            particle_history[iteridx0,:,pidx1], 
            particle_history[iteridx0,:,pidx2], 
            c=col0, alpha=background_alpha, s=1
        )
        ax.scatter(
            particle_history[iteridx1,:,pidx1], 
            particle_history[iteridx1,:,pidx2], 
            c=col1, alpha=background_alpha, s=1
        )
    
    for i, sidx in enumerate(samp_idxs_to_plot):
        root_particle = [particles0[i,pidx1], particles0[i,pidx2]]
        terminal_particles = sdict[sidx]
        sc1 = ax.scatter(
            *root_particle, 
            c=col0, s=size0,
        )
        for tp in terminal_particles:
            ax.plot(
                [tp[0], root_particle[0]], [tp[1], root_particle[1]], 
                'k', alpha=linealpha
            )
        sc2 = ax.scatter(
            terminal_particles[:,pidx1], terminal_particles[:,pidx2], 
            c=col1, s=size1
        )
    ax.legend([sc1, sc2], [f'Iter {iteridx0}', f'Iter {iteridx1}'])
    ax.set_xlabel(pname1)
    ax.set_ylabel(pname2)
    ax.set_title(f"Accepted perturbations: iter ${iteridx0}\\to{iteridx1}$")

    if save:
        plt.savefig(f"{imgdir}/{saveas}")


def plot_all_perturbation_sample(
        iteridx0, particle_history, sampling_idx_history, particle_idx_history,
        all_particle_history, all_sampling_idx_history, **kwargs):
    """
    """
    #~~~~~ Process kwargs ~~~~~#
    samp_idxs_to_plot = kwargs.get('samp_idxs_to_plot', None)
    col0 = kwargs.get('col0', 'b')
    col1 = kwargs.get('col1', 'r')
    size0 = kwargs.get('size0', 24)
    size1 = kwargs.get('size1', 12)
    linealpha = kwargs.get('linealpha', 0.25)
    save = kwargs.get('save', True)
    saveas = kwargs.get('saveas', "all_perturbation_history.png")
    imgdir = kwargs.get('imgdir', "figures")
    nsamp = kwargs.get('nsamp', 10)
    pidx1 = kwargs.get("pidx1", 0)
    pidx2 = kwargs.get("pidx2", 1)
    pname1 = kwargs.get("pname1", "$\\theta_1$")
    pname2 = kwargs.get("pname2", "$\\theta_2$")
    plot_background = kwargs.get('plot_background', True)
    background_alpha = kwargs.get('background_alpha', 0.2)
    acceptance_history = kwargs.get('acceptance_history', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    assert iteridx0 < len(particle_history) - 1, \
        "iteridx0={} invalid. Must be between 0 and {}".format(
        iteridx0, len(particle_history) - 1)

    iteridx1 = iteridx0 + 1

    nsamp = min(nsamp, len(np.unique(sampling_idx_history[iteridx1])))
    if samp_idxs_to_plot is None:
        samp_idxs_to_plot = np.random.choice(
            np.unique(all_sampling_idx_history[iteridx1]), nsamp, replace=False
        )
    else:
        samp_idxs_to_plot = np.array(samp_idxs_to_plot, dtype=int)

    # Get starting particles
    particles0 = particle_history[iteridx0, samp_idxs_to_plot]
    # part_idxs0 = particle_idx_history[iteridx0, samp_idxs_to_plot]

    # Get all ending particles
    particles1 = all_particle_history[iteridx1]
    # part_idxs1 = particle_idx_history[iteridx0][all_sampling_idx_history[iteridx1]]
    samp_idxs1 = all_sampling_idx_history[iteridx1]
    sdict = {}
    acc_dict = {}
    for sidx in samp_idxs_to_plot:
        sdict[sidx] = particles1[samp_idxs1 == sidx]
        acc_dict[sidx] = acceptance_history[iteridx1][samp_idxs1 == sidx]

    fig, ax = plt.subplots(1, 1)

    if plot_background:
        ax.scatter(
            particle_history[iteridx0,:,pidx1], 
            particle_history[iteridx0,:,pidx2], 
            c=col0, alpha=background_alpha, s=1
        )
        ax.scatter(
            particle_history[iteridx1,:,pidx1], 
            particle_history[iteridx1,:,pidx2], 
            c=col1, alpha=background_alpha, s=1
        )

    for i, sidx in enumerate(samp_idxs_to_plot):
        root_particle = [particles0[i,pidx1], particles0[i,pidx2]]
        terminal_particles = sdict[sidx]
        accepted_screen = acc_dict[sidx]
        sc1 = ax.scatter(
            *root_particle, 
            c=col0, s=size0,
        )
        for tp in terminal_particles:
            ax.plot(
                [tp[0], root_particle[0]], [tp[1], root_particle[1]], 
                'k', alpha=linealpha
            )
        sc2_acc = ax.scatter(
            terminal_particles[accepted_screen,pidx1], 
            terminal_particles[accepted_screen,pidx2], 
            c=col1, s=size1,
        )
        sc2_rej = ax.scatter(
            terminal_particles[~accepted_screen,pidx1], 
            terminal_particles[~accepted_screen,pidx2], 
            c=col1, s=size1, fc='none', ec=col1
        )
    ax.legend(
        [sc1, sc2_acc, sc2_rej], 
        [f'Iter {iteridx0}', 
         f'Iter {iteridx1} (accept)', 
         f'Iter {iteridx1} (reject)']
    )
    ax.set_xlabel(pname1)
    ax.set_ylabel(pname2)
    ax.set_title(f"All perturbations: iter ${iteridx0}\\to{iteridx1}$")

    if save:
        plt.savefig(f"{imgdir}/{saveas}")
