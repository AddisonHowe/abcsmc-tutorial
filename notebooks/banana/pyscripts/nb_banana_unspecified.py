#!/usr/bin/env python
# coding: utf-8

# # Banana Model (Unspecified)
# 

# In[ ]:


import os
import argparse
import numpy as np

from abcsmc.abcsmc import abcsmc
from abcsmc.models import BananaModel
from abcsmc.priors import UniformPrior
import abcsmc.pl as pl


# In[ ]:


SCRIPT = not pl.is_notebook()
if SCRIPT:
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--outdir", type=str, default="../../out/images")
    parser.add_argument("--imgdir", type=str, default="")
    parser.add_argument('-k', "--kernel", type=str, required=True, 
                        choices=['locm', 'mvn', 'uniform', 'gaussian'])
    parser.add_argument('-n', "--nparticles", type=int, default=800)
    parser.add_argument('-i', "--niters", type=int,default=5)
    parser.add_argument('-e0', "--eps0", type=float,default=5)
    parser.add_argument('-ep', "--eps_percentile", type=float, default=0.15)
    parser.add_argument('-d', "--data", type=list, default=[0, 0], nargs='+')
    parser.add_argument('-v', "--cov_const", type=float, nargs=4,
                        default=[1, 0, 0, 0.5])
    args = parser.parse_args()


# In[ ]:


# ABC-SMC parameters
KERNEL_METHOD = 'locm'
NPARTICLES = 1000
NITERS = 5
EPS0 = 5
EPS_PERCENTILE = 0.15
COV_CONST = [1, 0, 0, 0.5]  # covariance flattened

# Image output
imgdir = f"../../out/images/nb_banana_unspecified_{KERNEL_METHOD}"


# In[ ]:


# Reset parameters with command line arguments if running as a script
if SCRIPT:
    KERNEL_METHOD = args.kernel
    NPARTICLES = args.nparticles
    NITERS = args.niters
    EPS0 = args.eps0
    EPS_PERCENTILE = args.eps_percentile
    COV_CONST = args.cov_const
    if args.imgdir:
        imgdir = args.imgdir
    else:
        imgdir = f"{args.outdir}/nb_banana_unspecified_{KERNEL_METHOD}"

# Make image output directory 
os.makedirs(imgdir, exist_ok=True)


# ### Generate observed data $X_0$
# 

# In[ ]:


# Observed data
if SCRIPT:
    data = np.array(args.data, dtype=float).reshape([-1, 2])
else:
    data = np.array([[0, 0]])
assert data.shape[1] == 2, f"Bad shape for data. Got: {data.shape} Exp: (n,2)"

ndraws = len(data)
cov_const = np.array(COV_CONST).reshape([2, 2])

# Parameter indices and names used for plotting
pidx1 = 0
pidx2 = 1
pname1 = "$\\theta_1$"
pname2 = "$\\theta_2$"

# Priors
prior_theta1 = UniformPrior(-20, 20)
prior_theta2 = UniformPrior(-20, 20)
prior_list = [prior_theta1, prior_theta2]
plot_range = [[-20, 20], [-20, 20]]

# Model object
model = BananaModel(0, 0, cov=cov_const, ndraws=ndraws)

# Simulation function
def f_sim(particle):
    m = BananaModel(particle[0], particle[1], cov=cov_const, ndraws=ndraws)
    return m.generate_data(ndraws)

# Distance function
def f_dist(x):
    d = np.linalg.norm(x - data)
    assert np.ndim(d) == 0, "Bad distance function"
    return d

# Log data
np.savetxt(f"{imgdir}/data.txt", data)
print("Data D:\n", data)


# ### Plot Analytic Posterior

# In[ ]:


logposterior = True
pl.plot_posterior(
    model, data, prior_list,
    gridn=400,
    xlims=plot_range[0], ylims=plot_range[1],
    pname1=pname1, pname2=pname2,
    logposterior=logposterior,
    saveas=f"{imgdir}/analytic_posterior_plot.png",
    markersize=3,
    show=not SCRIPT,
)

print("Observed data D\n", data)


# ## Run ABC-SMC

# In[ ]:


particles, weights, results_dict = abcsmc(
    nparticles=NPARTICLES, 
    nparams=2, 
    prior_list=prior_list, 
    niters=NITERS,
    sim_func=f_sim,
    dist_func=f_dist, 
    eps0=EPS0, 
    eps_percentile=EPS_PERCENTILE, 
    min_eps=0, 
    kernel_method=KERNEL_METHOD,
    track_all_perturbations=True,
    disable_pbars=SCRIPT,
)


# In[ ]:


particle_history = results_dict['particle_history']
weight_history = results_dict['weight_history']
score_history = results_dict['score_history']
acceptance_rates = results_dict['acceptance_rates']
epsilon_history = results_dict['epsilon_history']
sampling_idx_history = results_dict['sampling_index_history']
particle_idx_history = results_dict['particle_index_history']
all_particle_history = results_dict['all_particle_history']
all_sampling_idx_history = results_dict['all_sampling_index_history']
all_particle_acceptance_history = results_dict['all_particle_acceptance_history']


# ## Plot results

# In[ ]:


pl.plot_results(
    particle_history, weight_history, score_history, acceptance_rates,
    epsilon_history, prior_list, 
    pname1=pname1, pname2=pname2,
    show=not SCRIPT,
    save=True, imgdir=imgdir
)


# ## Perturbations

# In[ ]:


N_PERTRUB_SAMP = 20
for iteridx in range(len(particle_history) - 1):
    pl.plot_perturbation_sample(
        iteridx, particle_history, sampling_idx_history, particle_idx_history, 
        nsamp=N_PERTRUB_SAMP, 
        pname1=pname1, pname2=pname2,
        imgdir=imgdir, 
        show=not SCRIPT,
        saveas=f"accepted_perturbations_{iteridx}_{iteridx+1}.png"
    )


# In[ ]:


N_PERTRUB_SAMP = 20

for iteridx in range(len(particle_history) - 1):
    pl.plot_all_perturbation_sample(
        iteridx, particle_history, sampling_idx_history, particle_idx_history, 
        all_particle_history, all_sampling_idx_history, 
        acceptance_history=all_particle_acceptance_history,
        nsamp=N_PERTRUB_SAMP, 
        pname1=pname1, pname2=pname2,
        imgdir=imgdir, 
        show=not SCRIPT,
        saveas=f"perturbations_{iteridx}_{iteridx+1}.png"
    )


# ### Compare analytic and empirical posteriors

# In[ ]:


emp_dist = pl.plot_empirical_posterior(
    particles, weights,
    nsamps=10000,
    pname1=pname1, pname2=pname2,
    show=not SCRIPT,
    saveas=f"{imgdir}/empirical_posterior.png"
)

pl.plot_posterior(
    model, data, prior_list,
    gridn=400,
    xlims=plot_range[0], ylims=plot_range[1],
    # xlims=[0, 16], ylims=[-2, 10],
    pname1=pname1, pname2=pname2,
    logposterior=True,
    empirical_dist=emp_dist,
    # empirical_dist=emp_dist[0:100],
    show=not SCRIPT,
    saveas=f"{imgdir}/posterior_comparison.png",
    legend_loc='upper left',
    plot_max_posterior=False
)


# In[ ]:




