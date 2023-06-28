import numpy as np
import scipy.stats as st
from abc import ABC, abstractmethod

"""
Data-Generating Parameterized Models

"""

class Model(ABC):
    """
    Abstract Model Class
    A model should have a `generate_data` function, that will yield an ndarray
    of a given amount of data, each element in the array being a single draw 
    from the model. A model is parameterized by a vector `theta`.
    """

    def __init__(self):
        super().__init__(self)

    @abstractmethod
    def generate_data(self, n):
        pass
    
    @abstractmethod
    def loglikelihood(self, x, theta):
        pass
    
    def logposterior(self, theta, x, prior_list):
        return self.loglikelihood(x, theta) + self.logprior(theta, prior_list)

    def logprior(self, theta, priors):
        return np.sum(
            [np.log(p.pdf(theta[:,i])) for i, p in enumerate(priors)],
            axis=0
        )


class GaussianModel(Model):
    """A Gaussian Model
    """

    def __init__(self, mu, var, ndraws=None):
        self.ndraws = ndraws
        self.mu = mu
        self.var = var
        self.sigma = np.sqrt(var)
    
    def generate_data(self, n):
        if self.ndraws:
            return self.mu + self.sigma * np.random.randn([n, self.ndraws])
        return self.mu + self.sigma * np.random.randn(n)
    
    def loglikelihood(self, x, theta):
        """Return the log likelihood P(x|theta).
        Args:
            x : float or ndarray of shape (ndraws,) : observed data.
            theta: ndarray of shape (n, 2) : parameters.
        Returns:
            ll : ndarray of shape (n)
        """
        _, dim_theta = theta.shape
        assert dim_theta == 2, "Wrong dimension for theta."
        mus = theta[:,0]
        sig = np.sqrt(theta[:,1])
        if self.ndraws:
            assert x.shape == (self.ndraws,), "Wrong dimension for x."
            x = np.array(x).reshape([-1, 1])
        else:
            assert np.ndim(x) == 0, "Wrong dimension for x."
            x = np.array(x).reshape([1, -1])
        ll = -0.5 * ((x - mus[None,:]) / sig[None,:])**2 - np.log(sig[None,:]) \
             - 0.5 * np.log(2*np.pi)
        return np.sum(ll, axis=0)
    

class RingModel(Model):

    def __init__(self, theta1, theta2, var=0.5, ndraws=None):
        self.ndraws = ndraws
        self.mu = theta1*theta1 + theta2*theta2
        self.var = var
        self.sigma = np.sqrt(var)

    def generate_data(self, n):
        return self.mu + self.sigma * np.random.randn(n)
        
    def loglikelihood(self, x, theta):
        """Return the log likelihood P(x|theta).
        Args:
            x : float or ndarray of shape (ndraws,) : observed data.
            theta: ndarray of shape (n, 2) : parameters.
        Returns:
            ll : ndarray of shape (n)
        """
        _, dim_theta = theta.shape
        assert dim_theta == 2, "Wrong dimension for theta."
        mus = theta[:,0] * theta[:,0] + theta[:,1] * theta[:,1]
        sig = np.sqrt(self.var)
        if self.ndraws:
            assert x.shape == (self.ndraws,), "Wrong dimension for x."
            x = np.array(x).reshape([-1, 1])
        else:
            assert np.ndim(x) == 0, "Wrong dimension for x."
            x = np.array(x).reshape([1, -1])
        ll = -0.5 * ((x - mus[None,:]) / sig)**2 - np.log(sig) \
             - 0.5 * np.log(2*np.pi)
        return np.sum(ll, axis=0)


class EllipsoidModel(Model):

    def __init__(self, theta1, theta2, var=0.5, ndraws=None):
        self.ndraws = ndraws
        self.theta1 = theta1
        self.theta2 = theta2
        self.mu = (theta1 - 2*theta2)**2 + (theta2 - 4)**2
        self.var = var
        self.sigma = np.sqrt(var)

    def generate_data(self, n):
        return self.mu + self.sigma * np.random.randn(n)
    
    def loglikelihood(self, x, theta):
        assert theta.shape[1] == 2, "Wrong shape for theta."    
        _, dim_theta = theta.shape
        assert dim_theta == 2, "Wrong dimension for theta."
        mus = (theta[:,0] - 2*theta[:,1])**2 + (theta[:,1] - 4)**2
        sig = np.sqrt(self.var)
        if self.ndraws:
            assert x.shape == (self.ndraws,), "Wrong dimension for x."
            x = np.array(x).reshape([-1, 1])
        else:
            assert np.ndim(x) == 0, "Wrong dimension for x."
            x = np.array(x).reshape([1, -1])
        ll = -0.5 * ((x - mus[None,:]) / sig)**2 - np.log(sig) \
             - 0.5 * np.log(2*np.pi)
        return np.sum(ll, axis=0)
        

class BananaModel(Model):
    
    def  __init__(self, theta1, theta2, cov=[[1, 0],[0, 0.5]], ndraws=None):
        self.ndraws = ndraws
        self.theta1 = theta1
        self.theta2 = theta2
        self.cov = np.array(cov)
        self.prec_chol = np.linalg.cholesky(np.linalg.inv(cov))
        self.mu = np.array([theta1, theta1 + theta2*theta2])
        self.mvn = st.multivariate_normal(self.mu, self.cov)
        self._normfact = np.log(np.linalg.det(self.prec_chol)) - np.log(2*np.pi)

    def generate_data(self, n):
        return self.mvn.rvs(n)
    
    def loglikelihood(self, x, theta):
        """Return the log likelihood P(x|theta).
        Args:
            x : ndarray of shape (2,) or (ndraws,2) : observed data.
            theta: ndarray of shape (n, 2) : parameters.
        Returns:
            ll : ndarray of shape (n)
        """
        _, dim_theta = theta.shape
        assert dim_theta == 2, "Wrong dimension for theta."
        mus = np.array([theta[:,0], theta[:,0] + theta[:,1] * theta[:,1]]).T
        if self.ndraws:
            assert x.shape == (self.ndraws, 2), "Wrong dimension for x."
            x = np.array(x).reshape([-1, 2])
        else:
            assert np.ndim(x) == 1 and len(x)==2, "Wrong dimension for x."
            x = np.array(x).reshape([1, 2])    
        diffterm = (x[:,None,:] - mus[None,:,:])[:,:,None,:] @ self.prec_chol
        expterm = np.sum(np.square(diffterm), axis=3).squeeze(2)
        ll = -0.5*expterm + self._normfact
        return ll.sum(axis=0)