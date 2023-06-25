import time
import signal
import numpy as np
from tqdm.auto import tqdm, trange
from .kernels import UniformKernel, GaussianKernel, MVNKernel, LOCMKernel

"""
The ABC-SMC Algorithm

"""

def abcsmc(nparticles, nparams, prior_list, niters, sim_func, dist_func,
           eps0, eps_percentile, min_eps, eps_sched=None, 
           kernel_method='uniform', **kwargs):
    """
    Implements the ABC-SMC algorithm.

    Args:
        nparticles : int : number of parameter particles.
        nparams : int : number of parameters.
        prior_list : list[Prior] : priors for each parameter.
        niters : int : number of iterations.
        sim_func : func : function to simulate.
        dist_func : func : function to scores simulation results.
        eps0 : float : starting value of epsilon.
        eps_percentile : float : epsilon reduction percentile.
        min_eps : float : minimum value of epsilon.
        eps_sched : list(float) : epsilon schedule.
        kernel_method : str : specifier for kernel.
    Returns:
        tuple ()
    """
    
    # Termination Signal Handler
    def terminate_signal(signalnum, handler):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate_signal)
        
    particle_history = np.zeros([niters, nparticles, nparams])
    weight_history = np.zeros([niters, nparticles])
    score_history = np.zeros([niters, nparticles])
    acceptance_rates = np.zeros(niters)
    epsilon_history = []
    
    prev_particles = np.empty([nparticles, nparams])
    prev_scores = np.empty(nparticles)
    prev_particles[:] = np.nan
    prev_scores[:] = np.nan

    kept_particles = np.empty([nparticles, nparams])
    kept_scores = np.empty(nparticles)
    sampidxs = -np.ones(nparticles, dtype=int)

    weights = np.ones(nparticles) / nparticles

    def compile_results(iteridx=niters):
        return {
            'particle_history': particle_history[0:iteridx],
            'weight_history': weight_history[0:iteridx],
            'score_history': score_history[0:iteridx],
            'acceptance_rates': acceptance_rates[0:iteridx],
            'epsilon_history': epsilon_history[0:iteridx],
            'niters': iteridx
        }
    
    try:
        print("Running ABC-SMC...")
        iterpbar = trange(niters, desc="Iteration")
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
                        return prev_particles, prev_scores, results
            
            epsilon_history.append(eps)

            # Determine perturbation kernel parameters
            if iteridx == 0:
                pass
            elif kernel_method == 'uniform':
                dx = 0.5 * (np.max(prev_particles, axis=0) - \
                            np.min(prev_particles, axis=0))
                kernel = UniformKernel(nparams, dx=dx)
            elif kernel_method == 'gaussian':
                good_idxs = prev_scores <= eps
                good_particles = prev_particles[good_idxs]
                good_weights = weights[good_idxs]
                good_weights /= good_weights.sum()
                sigma = np.zeros(nparams)
                for pidx in range(nparams):
                    sigma[pidx] =  np.sum(
                        weights * np.sum(
                            good_weights * np.square(
                                prev_particles[:,pidx][:,None] - \
                                good_particles[:,pidx][None,:]
                            ), axis=1
                        )
                    )
                kernel = GaussianKernel(nparams, sigma)
            elif kernel_method == 'mvn':
                good_idxs = prev_scores <= eps
                good_particles = prev_particles[good_idxs]
                good_weights = weights[good_idxs]
                good_weights /= good_weights.sum()
                z = good_particles[None,:] - prev_particles[:,None]
                t0 = z[:,:,:,None] @ z[:,:,None,:]  
                t1 = np.sum(good_weights[None,:,None,None] * t0, axis=1)
                cov =  np.sum(weights[:,None,None] * t1, axis=0)
                kernel = MVNKernel(nparams, cov)
            elif kernel_method == 'locm':
                good_idxs = prev_scores <= eps
                good_particles = prev_particles[good_idxs]
                good_weights = weights[good_idxs]
                good_weights /= good_weights.sum()
                z = good_particles[None,:] - prev_particles[:,None]
                mus = prev_particles.copy()
                covs = np.sum((z[:,:,:,None] @ z[:,:,None,:]) * \
                              good_weights[None,:,None,None], axis=1)
                kernel = LOCMKernel(nparams, mus, covs)
            else:
                raise RuntimeError(f"Unknown kernel method `{kernel_method}`")
            
            if iteridx > 0:
                print(kernel)

            # Compute local covariances for perturbation
            if iteridx > 0:
                tic0 = time.time()
                # locms = ...
                tic1 = time.time()
            
            # Reset arrays tracking kept particles
            kept_particles[:] = np.nan
            kept_scores[:] = np.nan
            
            if iteridx == 0:
                pbar = tqdm(total=nparticles, 
                            desc=f"Current count [Iter {iteridx}]")
            else:
                pbar.set_description(f"Current count [Iter {iteridx}]")
                pbar.reset()

            print(f"Epsilon: {eps:.4g}")
            
            count = 0
            tries = 0
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
                        # Perturb sampled particles
                        particle = kernel.perturb(particle, idx)
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
                    if iteridx > 0:
                        sampidxs[count] = idx
                    count += 1
                    pbar.update(1)
                tries += 1
                tic1 = time.time()
            if iteridx == 0:
                weights = np.ones(nparticles) / nparticles
            else:
                weights = compute_weights(
                    kept_particles, sampidxs, prev_particles, weights, 
                    prior_list, kernel
                )
            tic1 = time.time()
            print(f"Iter {iteridx} finished in {tic1 - itertime0:.3g} secs")
            pbar.refresh()

            acceptance_rates[iteridx] = count / tries
            
            prev_particles = kept_particles.copy()
            prev_scores = kept_scores.copy()

            particle_history[iteridx] = prev_particles
            score_history[iteridx] = prev_scores
            weight_history[iteridx] = weights
                    
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
        print(errm)
        pbar.close()
        iterpbar.close()
        # Return results of last completed iteration
        results = compile_results(iteridx=iteridx)
        if iteridx > 0:            
            return prev_particles, prev_scores, results
        else:
            return None, None, None
        
########################################################################
##########################  Helper Functions  ##########################
########################################################################

def eval_prior_prob(particle, prior_list):
    return np.prod([p.pdf(particle) for p in prior_list])

def sample_from_priors(prior_list, n=None):
    if n:
        return np.array([p.rvs(n) for p in prior_list]).T
    return np.array([p.rvs() for p in prior_list])

def sample_from_distribution(particles, weights):
    sampidx = np.random.choice(len(particles), p=weights)    
    return particles[sampidx].copy(), sampidx

def compute_weights(particles, sampidxs, prev_particles, prev_weights, 
                    prior_list, kernel):
    new_weights = np.zeros(prev_weights.shape)
    for i in range(len(particles)):
        denom = np.sum(
            prev_weights * kernel.vpdf(
                prev_particles, particles[i], sampidxs[i]
            )
        )
        new_weights[i] = eval_prior_prob(particles[i], prior_list) / denom
    return new_weights / new_weights.sum()
