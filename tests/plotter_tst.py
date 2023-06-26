import numpy as np
from abcsmc.plotting import *

x0 = np.array([0, 0])
x1 = np.array([0, 1])
x2 = np.array([1, 0])
x3 = np.array([1, 1])

dx0 = 0
dx1 = 1
dx2 = 2
dx3 = 3

particle_history = np.array([
    [x0+dx0, x1+dx0, x2+dx0, x3+dx0],
    [x0+dx1, x1+dx1, x2+dx1, x3+dx1],
    [x0+dx2, x1+dx2, x2+dx2, x3+dx2],
    [x0+dx3, x1+dx3, x2+dx3, x3+dx3],
])
print(particle_history)

weight_history = np.array([
    [0.25, 0.25, 0.4, 0.1],
    [0.25, 0.25, 0.4, 0.1],
    [0.25, 0.25, 0.4, 0.1],
    [0.25, 0.25, 0.4, 0.1],
])

score_history = np.array([
    [1/0.25, 1/0.25, 1/0.4, 1/0.1],
    [1/0.25, 1/0.25, 1/0.4, 1/0.1],
    [1/0.25, 1/0.25, 1/0.4, 1/0.1],
    [1/0.25, 1/0.25, 1/0.4, 1/0.1],
])

acceptance_rates = np.array([
    0.9, 0.8, 0.5, 0.2
])

epsilon_history = np.array([
    3, 2, 1, 0.5
])

sampling_idx_history = np.array([
    [-1,-1,-1,-1],
    [3, 2, 0, 1],
    [3, 3, 3, 1],
    [1, 3, 0, 0],
])

particle_idx_history = np.array([
    [0,1,2,3],
    [3,2,0,1],
    [1,1,1,2],
    [1,2,1,1],
])


x4 = [5, 5]
x5 = [5, 6]
x6 = [6, 6]

all_particle_history = [
    np.array([x4, x0+dx0, x1+dx0, x2+dx0, x3+dx0]),
    np.array([x4, x0+dx1, x5, x1+dx1, x2+dx1, x3+dx1]),
    np.array([x4, x0+dx2, x5, x1+dx2, x2+dx2, x6, x3+dx2]),
    np.array([x4, x0+dx3, x5, x1+dx3, x2+dx3, x6, x3+dx3]),
]
print(all_particle_history)

all_sampling_idx_history = [
    np.array([-1, -1,-1,-1,-1], dtype=int),
    np.array([0, 3, 1, 2, 0, 1], dtype=int),
    np.array([1, 3, 0, 3, 3, 1, 1], dtype=int),
    np.array([0, 1, 1, 3, 0, 3, 0], dtype=int),
]

all_particle_acceptance_history = [
    np.array([False, True, True, True, True], dtype=bool),
    np.array([False, True, False, True, True, True], dtype=bool),
    np.array([False, True, False, True, True, False, True], dtype=bool),
    np.array([False, True, False, True, True, False, True], dtype=bool),
]


NSAMP = 20
# SAMP_IDXS = None
np.random.seed(3131)
# SAMP_IDXS = [0, 1, 3]

for iteridx in range(len(particle_history) - 1):
    plot_perturbation_sample(
        iteridx, particle_history, sampling_idx_history, particle_idx_history, 
        nsamp=NSAMP, 
        # samp_idxs_to_plot=SAMP_IDXS,
        save=False,
        # imgdir=imgdir, 
        # saveas=f"perturbation_{iteridx}_{iteridx+1}.png"
    )

for iteridx in range(len(particle_history) - 1):
    plot_all_perturbation_sample(
        iteridx, particle_history, sampling_idx_history, particle_idx_history, 
        all_particle_history, all_sampling_idx_history,
        acceptance_history=all_particle_acceptance_history,
        nsamp=NSAMP, 
        # samp_idxs_to_plot=SAMP_IDXS,
        save=False,
        # imgdir=imgdir, 
        # saveas=f"perturbation_{iteridx}_{iteridx+1}.png"
    )

plt.show()