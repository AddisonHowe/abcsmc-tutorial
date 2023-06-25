import numpy as np

"""
Perturbation Kernels

"""

class UniformKernel:
    
    def __init__(self, dim, dx):
        self.dim = dim
        self.dx =  np.array(dx) if np.ndim(dx) else dx * np.ones(dim)
        self.area = np.prod(2*self.dx)
        self.density = 1 / self.area
    
    def __str__(self):
        if np.ndim(self.dx) > 1:
            dxstr = f"[{', '.join(self.dx)}]"
        else:
            dxstr = str(self.dx)
        return f"Uniform Kernel: <dim={self.dim}, dx={dxstr}>"

    def perturb(self, particle, idx=None):
        dx = self.dx * (2 * np.random.random(self.dim) - 1)
        return particle + dx
    
    def pdf(self, particle1, particle0, idx=None):
        dx = np.abs(particle1 - particle0)
        return self.density * np.all(dx <= self.dx)
    
    def vpdf(self, particles, particle0, idx=None):
        dx = np.abs(particles - particle0)
        return self.density * np.all(dx <= self.dx, axis=1)
        

class GaussianKernel:
    
    def __init__(self, dim, sigmas):
        self.dim = dim
        if np.ndim(sigmas):
            self.sigmas = np.array(sigmas) 
        else:
            self.sigmas = sigmas * np.ones(dim)

    def __str__(self):
        if np.ndim(self.sigmas) > 1:
            sigmasstr = f"[{', '.join(self.sigmas)}]"
        else:
            sigmasstr = str(self.sigmas)
        return f"Gaussian Kernel: <dim={self.dim}, sigmas={sigmasstr}>"

    def perturb(self, particle, idx=None):
        dx = np.random.randn(self.dim) * self.sigmas
        return particle + dx

    def pdf(self, particle1, particle0, idx=None):
        diffterm = (particle1 - particle0) / self.sigmas
        denom = self.sigmas * np.sqrt(2*np.pi)
        return np.prod(np.exp(-0.5*diffterm**2) / denom)
    
    def vpdf(self, particles, particle0, idx=None):
        diffterm = (particles - particle0[None,:]) / self.sigmas[None,:]
        denom = self.sigmas[None,:] * np.sqrt(2*np.pi)
        return np.prod(np.exp(-0.5 * diffterm * diffterm) / denom, axis=1)


class MVNKernel:
    
    def __init__(self, dim, cov):
        self.dim = dim
        self.cov = np.array(cov)  # Sigma
        self.prec = np.linalg.inv(self.cov)  # Sigma^(-1)
        self.cov_chol = np.linalg.cholesky(self.cov)  # L: Sigma = LL'
        self.prec_chol = np.linalg.cholesky(self.prec)  # A: Sigma^-1 = AA'
        self.det_prec_chol = np.linalg.det(self.prec_chol)

    def __str__(self):
        covstr = f"[{', '.join([str(f) for f in self.cov.flatten()])}]"
        return f"Multivariate Normal Kernel: <dim={self.dim}, sigma={covstr}>"

    def perturb(self, particle, idx=None):
        dx = self.cov_chol @ np.random.randn(self.dim) 
        return particle + dx

    def pdf(self, particle1, particle0, idx=None):
        diffterm = particle1 - particle0
        expterm = np.sum(np.square(diffterm[None,:] @ self.prec_chol))
        denom = np.sqrt(2*np.pi)**self.dim
        return np.exp(-0.5*expterm) * self.det_prec_chol / denom
    
    def vpdf(self, particles, particle0, idx=None):
        diffterm = particles - particle0[None,:]
        expterm = np.sum(np.square(diffterm @ self.prec_chol), axis=1)
        denom = np.sqrt(2*np.pi)**self.dim
        return np.exp(-0.5*expterm) * self.det_prec_chol / denom


class LOCMKernel:
    
    def __init__(self, dim, mus, covs):
        self.dim = dim
        self.mus = mus
        self.covs = covs
        self.precs = np.linalg.inv(self.covs)  # Sigma^(-1)
        self.cov_chols = np.linalg.cholesky(self.covs)  # L: Sigma = LL'
        self.prec_chols = np.linalg.cholesky(self.precs)  # A: Sigma^-1 = AA'
        self.det_prec_chols = np.linalg.det(self.prec_chols)

    def __str__(self):
        return f"LOCM Kernel: <dim={self.dim}>"

    def perturb(self, particle, idx):
        dx = self.cov_chols[idx] @ np.random.randn(self.dim) 
        return particle + dx
    
    def pdf(self, particle1, particle0, idx):
        diffterm = particle1 - particle0
        expterm = np.sum(np.square(diffterm[None,:] @ self.prec_chols[idx]))
        denom = np.sqrt(2*np.pi)**self.dim
        return np.exp(-0.5*expterm) * self.det_prec_chols[idx] / denom
    
    def vpdf(self, particles, particle0, idx):
        diffterm = particles - particle0[None,:]
        expterm = np.sum(np.square(diffterm @ self.prec_chols[idx]), axis=1)
        denom = np.sqrt(2*np.pi)**self.dim
        return np.exp(-0.5*expterm) * self.det_prec_chols[idx] / denom
