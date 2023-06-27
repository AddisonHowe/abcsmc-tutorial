import numpy as np
import scipy.stats as st
from abc import ABC, abstractmethod

"""
Data-Generating Parameterized Models

"""

class Model(ABC):
    """Abstract Model Class
    All Models should be able to generate a given amount of data.
    """

    def __init__(self):
        super().__init__(self)

    @abstractmethod
    def generate_data(self, n):
        pass


class GaussianModel(Model):

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
        self.sigma = np.sqrt(var)
    
    def generate_data(self, n):
        return self.mu + self.sigma * np.random.randn(n)
    
    
class RingModel(Model):

    def __init__(self, theta1, theta2, var):
        self.mu = theta1*theta1 + theta2*theta2
        self.var = var
        self.sigma = np.sqrt(var)

    def generate_data(self, n):
        return self.mu + self.sigma * np.random.randn(n)


class BananaModel(Model):
    
    def  __init__(self, theta1, theta2, cov=[[1, 0],[0, 0.5]]):
        self.theta1 = theta1
        self.theta2 = theta2
        self.cov = np.array(cov)
        self.mu = np.array([theta1, theta1 + theta2*theta2])
        self.mvn = st.multivariate_normal(self.mu, self.cov)

    def generate_data(self, n):
        return self.mvn.rvs(n)
