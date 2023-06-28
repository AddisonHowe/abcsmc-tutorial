from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as st

"""
Priors

"""

class Prior(ABC):
    """Abstract Prior Class
    All Priors should have a `rvs` and `pdf` method.
    """

    @abstractmethod
    def rvs(self, shape=None):
        """Randomly generate an ndarray of the given shape."""
        pass

    @abstractmethod
    def pdf(self, x):
        """Compute the probability density of the given value."""
        pass


class UniformPrior:
    """A Uniform Prior
    Defined by its support [a, b].
    """

    def __init__(self, a, b, seed=None):
        self.a = a
        self.b = b
        self.uniform = st.uniform(a, b-a)
        if seed:
            self.seed(seed)

    def __eq__(self, other) -> bool:
        if not isinstance(other, UniformPrior):
            return False
        return self.a == other.a and self.b == other.b

    def rvs(self, shape=None):
        return self.uniform.rvs(shape)

    def pdf(self, x):
        return self.uniform.pdf(x)
    
    def seed(self, seed):
        if isinstance(seed, (int, np.random.Generator)):
            self.uniform.random_state = seed
        else:
            raise RuntimeError(f"Unknown seed type: {type(seed)}")


class GaussianPrior:
    """A Gaussian Prior
    Normal distribution defined by mean and variance: N(mu, var).
    """

    def __init__(self, mu, var, seed=None):
        self.mu = mu
        self.var = var
        self.norm = st.norm(mu, var)
        if seed:
            self.seed(seed)

    def __eq__(self, other) -> bool:
        if not isinstance(other, GaussianPrior):
            return False
        return self.mu == other.mu and self.var == other.var

    def rvs(self, shape=None):
        return self.norm.rvs(shape)
    
    def pdf(self, x):
        return self.norm.pdf(x)
    
    def seed(self, seed):
        if isinstance(seed, (int, np.random.Generator)):
            self.norm.random_state = seed
        else:
            raise RuntimeError(f"Unknown seed type: {type(seed)}")


class NegGammaPrior:
    """A Negative Gamma Prior
    Neg. Gamma distribution with parameters a and b.
    """

    def __init__(self, a, b, seed=None):
        self.a = a
        self.b = b
        self.gam = st.gamma(a=a, scale=b)
        if seed:
            self.seed(seed)

    def __eq__(self, other) -> bool:
        if not isinstance(other, NegGammaPrior):
            return False
        return self.a == other.a and self.b == other.b

    def rvs(self, shape=None):
        return -self.gam.rvs(shape)
    
    def pdf(self, x):
        return self.gam.pdf(-x)
    
    def seed(self, seed):
        if isinstance(seed, (int, np.random.Generator)):
            self.gam.random_state = seed
        else:
            raise RuntimeError(f"Unknown seed type: {type(seed)}")


class StepPrior:
    """A Step Function Prior
    The prior distribution is supported over the interval [a, b], with a jump 
    at j, with a<j<b. The PDF is such that a proportion p of the density is 
    above j, i.e. a sampled point x has P(x>j) = p.
    """

    def __init__(self, a, b, j, p, seed=None):
        self.a = a
        self.b = b
        self.j = j
        self.p = p
        self.u1 = st.uniform(a, j-a)
        self.u2 = st.uniform(j, b-j)
        self.u3 = st.uniform()
        if seed:
            self.seed(seed)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StepPrior):
            return False
        return self.a == other.a and self.b == other.b \
               and self.j == other.j and self.p == other.p
        
    def rvs(self, shape=None):
        s1 = self.u1.rvs(shape)  # sample from u1
        s2 = self.u2.rvs(shape)  # sample from u2
        use2 = self.u3.rvs(shape) < self.p  # flip coins to decide which side
        return s1*(~use2) + s2*use2
    
    def pdf(self, x):
        return self.u1.pdf(x) * (1-self.p) + self.u2.pdf(x) * self.p
    
    def seed(self, seed):
        if isinstance(seed, int):
            self.u1.random_state = seed
            self.u2.random_state = seed + 10
            self.u3.random_state = seed + 20
        elif isinstance(seed, np.random.Generator):
            self.u1.random_state = seed
            self.u2.random_state = seed
            self.u3.random_state = seed
        else:
            raise RuntimeError(f"Unknown seed type: {type(seed)}")
