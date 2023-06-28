import time
import signal
import numpy as np
from tqdm.auto import tqdm, trange
from .kernels import UniformKernel, GaussianKernel, MVNKernel, LOCMKernel

"""
The ABC-SMC Algorithm

"""

def abcsmc(nparticles, nparams, prior_list, niters, sim_func, dist_func,
           eps0=1, eps_percentile=0.15, min_eps=0, eps_sched=None, 
           kernel_method='uniform', **kwargs):
    """
    Implements the ABC-SMC algorithm.

    Args:
        nparticles : int : number of parameter particles.
        nparams : int : number of parameters, i.e. dim of a parameter vector.
        prior_list : list[Prior] : priors for each parameter. Length nparams.
        niters : int : number of iterations.
        sim_func : func : function to simulate.
            Input is parameter vector of shape (nparams,).
            Output is data array of arbitrary shape S.
        dist_func : func : function to scores simulation results.
            Input is data array of shape S, as returned by sim_func.\
            Output is a real number.
        eps0 : float : starting value of epsilon.
        eps_percentile : float : epsilon reduction percentile.
        min_eps : float : minimum value of epsilon.
        eps_sched : list(float) : epsilon schedule.
        kernel_method : str : specifier for kernel.
    Returns:
        tuple ()
    """
    
    #~~~~~ Process kwargs ~~~~~#
    track_all_perturbations = kwargs.get('track_all_perturbations', False)
    track_limit = kwargs.get('track_limit', 1000)
    verbosity = kwargs.get('verbosity', 1)
    disable_pbars = kwargs.get('disable_pbars', False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~#

    assert len(prior_list) == nparams, "Wrong # of priors given."
    
    # Termination Signal Handler
    def terminate_signal(signalnum, handler):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate_signal)

    if eps_sched:
        niters = len(eps_sched)
        
    particle_history = np.zeros([niters, nparticles, nparams])
    weight_history = np.zeros([niters, nparticles])
    score_history = np.zeros([niters, nparticles])
    acceptance_rates = np.zeros(niters)
    epsilon_history = []
    
    sampling_index_history = np.zeros([niters, nparticles], dtype=int)
    particle_index_history = np.zeros([niters, nparticles], dtype=int)
    
    all_particle_history = None
    all_sampling_index_history = None
    all_particle_acceptance_history = None
    if track_all_perturbations:
        all_particle_history = [[] for _ in range(niters)]
        all_sampling_index_history = [[] for _ in range(niters)]
        all_particle_acceptance_history = [[] for _ in range(niters)]

    prev_particles = np.empty([nparticles, nparams])
    prev_scores = np.empty(nparticles)
    prev_particles[:] = np.nan
    prev_scores[:] = np.nan

    kept_particles = np.empty([nparticles, nparams])
    kept_scores = np.empty(nparticles)
    sampling_idxs = np.empty(nparticles, dtype=int)
    particle_idxs = np.empty(nparticles, dtype=int)
    prev_part_idxs = np.empty(nparticles, dtype=int)

    # weights = np.ones(nparticles) / nparticles

    def compile_results(iteridx=niters):
        return {
            'particle_history': particle_history[0:iteridx],
            'weight_history': weight_history[0:iteridx],
            'score_history': score_history[0:iteridx],
            'acceptance_rates': acceptance_rates[0:iteridx],
            'epsilon_history': epsilon_history[0:iteridx],
            'sampling_index_history': sampling_index_history[0:iteridx],
            'particle_index_history': particle_index_history[0:iteridx],
            'niters': iteridx,
            'all_sampling_index_history': all_sampling_index_history,
            'all_particle_history': all_particle_history,
            'all_particle_acceptance_history': all_particle_acceptance_history,
        }
    
    try:
        if verbosity:
            print("Running ABC-SMC...")
        iterpbar = trange(niters, desc="Iteration", disable=disable_pbars)
        for iteridx in iterpbar:
            itertime0 = time.time()
            
            # Determine next threshold epsilon
            if eps_sched:
                eps = eps_sched[iteridx]
            else:
                if iteridx == 0:
                    eps = eps0
                else:
                    eps = np.percentile(prev_scores, eps_percentile*100)
                    if eps <= min_eps:
                        em = "Reached eps={:.4g}<={:.4g} after iter {}".format(
                            eps, min_eps, iteridx-1)
                        em += "\nInterrupted Execution."
                        em += f"\nReturning (iter {iteridx-1})."
                        print(em)
                        results = compile_results(iteridx=iteridx)
                        iterpbar.colour = 'green'
                        pbar.close()
                        iterpbar.close()
                        return prev_particles, weights, results
            
            epsilon_history.append(eps)

            # Determine perturbation kernel parameters
            if iteridx == 0:
                pass
            elif kernel_method == 'uniform':
                kernel = UniformKernel.construct_from_data(prev_particles)
            elif kernel_method == 'gaussian':
                kernel = GaussianKernel.construct_from_data(
                    prev_particles, prev_scores, weights, eps
                )
            elif kernel_method == 'mvn':
                kernel = MVNKernel.construct_from_data(
                    prev_particles, prev_scores, weights, eps
                )
            elif kernel_method == 'locm':
                kernel = LOCMKernel.construct_from_data(
                    prev_particles, prev_scores, weights, eps
                )
            else:
                raise RuntimeError(f"Unknown kernel method `{kernel_method}`")
            
            # Reset arrays tracking kept particles
            kept_particles[:] = np.nan
            kept_scores[:] = np.nan
            
            if iteridx == 0:
                pbar = tqdm(total=nparticles, 
                            desc=f"Current count [Iter {iteridx}]",
                            disable=disable_pbars)
            else:
                pbar.set_description(f"Current count [Iter {iteridx}]")
                pbar.reset()

            if verbosity:
                print(f"Epsilon: {eps:.4g}")
            
            count = 0
            tries = 0
            num_tracked = 0  # for tracking all particles perturbed
            while count < nparticles:
                if iteridx == 0:
                    # Sample particle from prior
                    particle = sample_from_priors(prior_list)
                else:
                    # Sample particle from the previous distribution
                    found = False
                    while not found:
                        particle, idx = sample_from_distribution(
                            prev_particles, weights
                        )
                        assert particle.shape == (nparams,), \
                            "Wrong shape for sampled particle before perturb."
                        # Perturb sampled particles
                        particle = kernel.perturb(particle, idx=idx)
                        assert particle.shape == (nparams,), \
                            "Wrong shape for sampled particle after perturb."
                        prob_particle = eval_prior_prob(particle, prior_list)
                        if prob_particle > 0:
                            found = True
                
                # Simulate
                data = sim_func(particle)
                # Score to find distance
                score = dist_func(data)

                # Accept or reject the particle
                if score <= eps:
                    kept_particles[count] = particle
                    kept_scores[count] = score
                    if iteridx == 0:
                        sampling_idxs[count] = -1
                        particle_idxs[count] = count
                    else:
                        sampling_idxs[count] = idx
                        particle_idxs[count] = prev_part_idxs[idx]
                    count += 1
                    pbar.update(1)
                
                if track_all_perturbations and num_tracked < track_limit:
                    all_particle_history[iteridx].append(particle)
                    if iteridx == 0:
                        all_sampling_index_history[iteridx].append(-1)
                    else:
                        all_sampling_index_history[iteridx].append(idx)
                    all_particle_acceptance_history[iteridx].append(
                        score <= eps)
                    num_tracked += 1

                tries += 1
                tic1 = time.time()
            if iteridx == 0:
                weights = np.ones(nparticles) / nparticles
            else:
                weights = compute_weights(
                    kept_particles, sampling_idxs, prev_particles, weights, 
                    prior_list, kernel
                )
            tic1 = time.time()
            if verbosity:
                print(f"Iter {iteridx} finished in {tic1 - itertime0:.3g} secs")
            pbar.refresh()
            acceptance_rates[iteridx] = count / tries
            
            prev_particles[:] = kept_particles
            prev_scores[:] = kept_scores
            prev_part_idxs[:] = particle_idxs

            particle_history[iteridx] = kept_particles
            score_history[iteridx] = kept_scores
            weight_history[iteridx] = weights
            sampling_index_history[iteridx] = sampling_idxs
            particle_index_history[iteridx] = particle_idxs
            
            if track_all_perturbations:
                all_sampling_index_history[iteridx] = np.array(
                    all_sampling_index_history[iteridx], dtype=int
                )
                all_particle_history[iteridx] = np.array(
                    all_particle_history[iteridx]
                )
                all_particle_acceptance_history[iteridx] = np.array(
                    all_particle_acceptance_history[iteridx], dtype=bool
                )

        results = compile_results(iteridx=iteridx+1)
        pbar.close()
        iterpbar.close()
        return prev_particles, weights, results
    
    except KeyboardInterrupt:
        errm = "Caught KeyboardInterrupt Error"
        errm += "\nHalted during execution of iteration {}/{}".format(
            iteridx, niters)
        errm += f"\nCurrent count: {count}/{nparticles}"
        errm += f"\nCurrent epsilon: {eps:.8g}"
        if iteridx > 0:
            errm += f"\nReturning previous state (iteration {iteridx-1})."
        else:
            errm += "\nReturning (None, None)."
        if verbosity:
            print(errm)
        pbar.close()
        iterpbar.close()
        # Return results of last completed iteration
        results = compile_results(iteridx=iteridx)
        if iteridx > 0:            
            return prev_particles, weights, results
        else:
            return None, None, None
        
########################################################################
##########################  Helper Functions  ##########################
########################################################################

def eval_prior_prob(particle, prior_list):
    return np.prod([p.pdf(particle[i]) for i, p in enumerate(prior_list)])

def sample_from_priors(prior_list, n=None):
    if n:
        return np.array([p.rvs(n) for p in prior_list]).T
    return np.array([p.rvs() for p in prior_list])

def sample_from_distribution(particles, weights):
    sampidx = np.random.choice(len(particles), p=weights)    
    return particles[sampidx], sampidx

def compute_weights(particles, sampling_idxs, prev_particles, prev_weights, 
                    prior_list, kernel):
    new_weights = np.zeros(prev_weights.shape)
    for i in range(len(particles)):
        denom = np.sum(
            prev_weights * kernel.vpdf(
                particles[i][None,:], prev_particles, 
                idx=np.arange(len(prev_particles))
            ).flatten()
        )
        new_weights[i] = eval_prior_prob(particles[i], prior_list) / denom
    return new_weights / new_weights.sum()
