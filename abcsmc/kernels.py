from abc import ABC, abstractmethod
import numpy as np

"""
Perturbation Kernels

"""

class Kernel(ABC):

    @staticmethod
    @abstractmethod
    def construct_from_data(particles, scores=None, weights=None, eps=None):
        pass

    @abstractmethod
    def perturb(self, particle1, particle0=None, **kwargs):
        """Perturb particle1 according to K( _ | p0 )"""
        pass

    @abstractmethod
    def pdf(self, particle1, particle0=None, **kwargs):
        """Compute the density K( p1 | p0 )"""
        pass

    @abstractmethod
    def vpdf(self, particles1, particles0=None, **kwargs):
        """Compute the densities K( p1_i | p0_j )"""
        pass


class UniformKernel(Kernel):
    """
    Perturbs a particle component-wise, with each component theta_j drawn 
    independently from a uniform distribution over the interval 
    [theta_j - dx_j, theta_j + dx_j] with dx_j chosen according to the 
    scale of the previous population.
    """
    
    def __init__(self, dim, dx):
        self.dim = dim
        self.dx = np.array(dx) if np.ndim(dx) else dx * np.ones(dim)
        self.area = np.prod(2*self.dx)
        self.density = 1 / self.area

    @staticmethod
    def construct_from_data(particles, scores=None, weights=None, eps=None):
        n, d = particles.shape
        dx = 0.5 * (np.max(particles, axis=0) - np.min(particles, axis=0))
        return UniformKernel(d, dx=dx)
    
    def __str__(self):
        if np.ndim(self.dx) > 1:
            dxstr = f"[{', '.join(self.dx)}]"
        else:
            dxstr = str(self.dx)
        return f"Uniform Kernel: <dim={self.dim}, dx={dxstr}>"

    def perturb(self, particle1, particle0=None, **kwargs):
        dx = self.dx * (2 * np.random.random(self.dim) - 1)
        return particle1 + dx
    
    def pdf(self, particle1, particle0, **kwargs):
        dx = np.abs(particle1 - particle0)
        return self.density * np.all(dx <= self.dx)
    
    def vpdf(self, particles1, particles0, **kwargs):
        dx = np.abs(particles1[:,None,:] - particles0[None,:,:])
        densities = self.density * np.all(dx <= self.dx, axis=2)
        return densities
        

class GaussianKernel:
    """
    Perturbs a particle component-wise, with each component theta_j drawn 
    independently from a 1d Gaussian with mean theta_j and variance sigma_j^2, 
    with sigma_j given by the scale of the previous population.
    """
    
    def __init__(self, dim, sigmas):
        self.dim = dim
        if np.ndim(sigmas):
            self.sigmas = np.array(sigmas) 
        else:
            self.sigmas = sigmas * np.ones(dim)

    @staticmethod
    def construct_from_data(particles, scores, weights, eps):
        n, d = particles.shape
        good_idxs = scores <= eps
        good_particles = particles[good_idxs]
        good_weights = weights[good_idxs]
        good_weights /= good_weights.sum()
        sigmas = np.zeros(d)
        for pidx in range(d):
            sigmas[pidx] =  np.sum(
                weights * np.sum(
                    good_weights * np.square(
                        particles[:,pidx][:,None] - \
                        good_particles[:,pidx][None,:]
                    ), axis=1
                )
            )
        return GaussianKernel(d, sigmas)

    def __str__(self):
        if np.ndim(self.sigmas) > 1:
            sigmasstr = f"[{', '.join(self.sigmas)}]"
        else:
            sigmasstr = str(self.sigmas)
        return f"Gaussian Kernel: <dim={self.dim}, sigmas={sigmasstr}>"

    def perturb(self, particle1, particle0=None, **kwargs):
        dx = np.random.randn(self.dim) * self.sigmas
        return particle1 + dx

    def pdf(self, particle1, particle0, **kwargs):
        diffterm = (particle1 - particle0) / self.sigmas
        denoms = self.sigmas * np.sqrt(2*np.pi)
        return np.prod(np.exp(-0.5 * diffterm * diffterm) / denoms)
    
    def vpdf(self, particles1, particles0, **kwargs):
        diffterm = particles1[:,None,:] - particles0[None,:,:]
        diffterm /= self.sigmas[None,None,:]
        denoms = self.sigmas[None,None,:] * np.sqrt(2*np.pi)
        return np.prod(np.exp(-0.5 * diffterm * diffterm) / denoms, axis=2)


class MVNKernel:
    """
    Perturbs a particle according to a multivariate Gaussian, with a specified 
    covariance matrix determined by the previous particle population.
    """
    
    def __init__(self, dim, cov):
        self.dim = dim
        self.cov = np.array(cov)  # Sigma
        self.prec = np.linalg.inv(self.cov)  # Sigma^(-1)
        self.cov_chol = np.linalg.cholesky(self.cov)  # L: Sigma = LL'
        self.prec_chol = np.linalg.cholesky(self.prec)  # A: Sigma^-1 = AA'
        self.det_prec_chol = np.linalg.det(self.prec_chol)
        self._normfact = self.det_prec_chol / np.sqrt(2*np.pi)**self.dim

    @staticmethod
    def construct_from_data(particles, scores, weights, eps):
        n, d = particles.shape
        good_idxs = scores <= eps
        good_particles = particles[good_idxs]
        good_weights = weights[good_idxs]
        good_weights /= good_weights.sum()
        z = good_particles[None,:] - particles[:,None]
        t0 = z[:,:,:,None] @ z[:,:,None,:]  
        t1 = np.sum(good_weights[None,:,None,None] * t0, axis=1)
        cov =  np.sum(weights[:,None,None] * t1, axis=0)
        return MVNKernel(d, cov)
    
    def __str__(self):
        covstr = f"[{', '.join([str(f) for f in self.cov.flatten()])}]"
        return f"Multivariate Normal Kernel: <dim={self.dim}, sigma={covstr}>"

    def perturb(self, particle1, particle0=None, **kwargs):
        dx = self.cov_chol @ np.random.randn(self.dim)
        return particle1 + dx

    def pdf(self, particle1, particle0, **kwargs):
        diffterm = particle1 - particle0
        expterm = np.sum(np.square(diffterm[None,:] @ self.prec_chol))
        return np.exp(-0.5*expterm) * self._normfact
    
    def vpdf(self, particles1, particles0, **kwargs):
        diffterm = particles1[:,None,:] - particles0[None,:,:]
        expterm = np.sum(np.square(diffterm @ self.prec_chol), axis=2)
        return np.exp(-0.5*expterm) * self._normfact


class LOCMKernel:
    
    def __init__(self, dim, mus, covs):
        self.n = len(mus)
        self.dim = dim
        self.mus = mus
        self.covs = covs
        assert mus.shape == (self.n, self.dim), "Bad shape for mus."
        assert covs.shape == (self.n, self.dim, self.dim), "Bad shape for covs."
        self.precs = np.linalg.inv(self.covs)  # Sigma^(-1)
        self.cov_chols = np.linalg.cholesky(self.covs)  # L: Sigma = LL'
        self.prec_chols = np.linalg.cholesky(self.precs)  # A: Sigma^-1 = AA'
        self.det_prec_chols = np.linalg.det(self.prec_chols)
        self._normfact = 1 / (np.sqrt(2*np.pi)**self.dim)

    @staticmethod
    def construct_from_data(particles, scores, weights, eps):
        n, d = particles.shape
        good_idxs = scores <= eps
        good_particles = particles[good_idxs]
        good_weights = weights[good_idxs]
        good_weights /= good_weights.sum()
        z = particles[:,None] - good_particles[None,:]
        mus = particles.copy()
        covs = np.sum((z[:,:,:,None] @ z[:,:,None,:]) * \
                        good_weights[None,:,None,None], axis=1)
        return LOCMKernel(d, mus, covs)

    def __str__(self):
        return f"LOCM Kernel: <dim={self.dim}>"

    def perturb(self, particle1, particle0=None, **kwargs):
        idx = kwargs['idx']
        dx = self.cov_chols[idx] @ np.random.randn(self.dim) 
        return particle1 + dx
    
    def pdf(self, particle1, particle0, **kwargs):
        idx = kwargs['idx']
        diffterm = particle1 - particle0
        expterm = np.sum(np.square(diffterm[None,:] @ self.prec_chols[idx]))
        return np.exp(-0.5*expterm) * self.det_prec_chols[idx] * self._normfact
    
    def vpdf(self, particles1, particles0, **kwargs):
        idxs = kwargs['idx']
        diffterm = particles1[:,None,:] - particles0[None,:,:]
        expterm = np.sum(
            np.square(diffterm[:,:,None,:] @ self.prec_chols[idxs][None,:]), 
            axis=3
        ).squeeze(2)
        return np.exp(-0.5*expterm) * self.det_prec_chols[idxs][None,:] \
               * self._normfact
