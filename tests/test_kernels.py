import pytest
import numpy as np
import scipy.stats as st
from abcsmc.kernels import UniformKernel, GaussianKernel, MVNKernel, LOCMKernel

"""
Test Script

"""

class TestUniform:

    @pytest.mark.parametrize('dim, dx, particle1, particle0, density_exp', [
        [2, 1, [-0.10, 0], [1, 0], 0.00],
        [2, 1, [ 0.10, 0], [1, 0], 0.25],
    ])
    def test_pdf(self, dim, dx, particle1, particle0, density_exp):
        kernel = self._get_kernel(dim, dx)
        d_exp = kernel.pdf(np.array(particle1), np.array(particle0))
        assert d_exp == density_exp, f"Expected {d_exp}. Got {density_exp}."

    @pytest.mark.parametrize('dim, dx, particles, particle0, density_exp', [
        [2, 1, [[-0.10, 0], [ 0.10, 0]], [1, 0], [0.00, 0.25]],
    ])
    def test_vpdf(self, dim, dx, particles, particle0, density_exp):
        kernel = self._get_kernel(dim, dx)
        d_exp = kernel.vpdf(np.array(particles), np.array(particle0))
        assert np.all(d_exp == density_exp), \
            f"Expected {d_exp}. Got {density_exp}."

    def _get_kernel(self, dim, dx):
        return UniformKernel(dim, dx)


class TestGaussian:

    @pytest.mark.parametrize('dim, sigmas, particle1, particle0', [
        [2, [1, 2], [-0.10, 0], [1, 0]],
        [2, [1, 2], [ 0.10, 0], [1, 0]],
    ])
    def test_pdf(self, dim, sigmas, particle1, particle0):
        kernel = self._get_kernel(dim, sigmas)
        d_exp = kernel.pdf(np.array(particle1), np.array(particle0))
        norm = st.multivariate_normal(mean=particle0, cov=np.diag(sigmas)**2)
        d_act = norm.pdf(particle1)
        assert np.allclose(d_exp, d_act), f"Expected {d_exp}. Got {d_act}."

    @pytest.mark.parametrize('dim, sigmas, particles, particle0', [
        [2, [1, 2], [[-0.10, 0], [ 0.10, 0]], [1, 0]],
    ])
    def test_vpdf(self, dim, sigmas, particles, particle0):
        kernel = self._get_kernel(dim, sigmas)
        d_exp = kernel.vpdf(np.array(particles), np.array(particle0))
        norm = st.multivariate_normal(mean=particle0, cov=np.diag(sigmas)**2)
        d_act = [norm.pdf(p) for p in particles]
        assert np.allclose(d_exp, d_act), f"Expected {d_exp}. Got {d_act}."

    def _get_kernel(self, dim, sigmas):
        return GaussianKernel(dim, sigmas)
    

class TestMVN:

    @pytest.mark.parametrize('dim, cov, particle1, particle0', [
        [2, [[1, 0.8], [0.8, 4]], [-0.10, 0], [1, 0]],
        [2, [[1, 0.8], [0.8, 4]], [ 0.10, 0], [1, 0]],
    ])
    def test_pdf(self, dim, cov, particle1, particle0):
        kernel = self._get_kernel(dim, cov)
        d_exp = kernel.pdf(np.array(particle1), np.array(particle0))
        norm = st.multivariate_normal(mean=particle0, cov=cov)
        d_act = norm.pdf(particle1)
        assert np.allclose(d_exp, d_act), f"Expected {d_exp}. Got {d_act}."
    
    @pytest.mark.parametrize('dim, cov, particles, particle0', [
        [2, [[1, 0.8], [0.8, 4]], [[-0.10, 0], [ 0.10, 0]], [1, 0]],
    ])
    def test_vpdf(self, dim, cov, particles, particle0):
        kernel = self._get_kernel(dim, cov)
        d_exp = kernel.vpdf(np.array(particles), np.array(particle0))
        norm = st.multivariate_normal(mean=particle0, cov=cov)
        d_act = [norm.pdf(p) for p in particles]
        assert np.allclose(d_exp, d_act), f"Expected {d_exp}. Got {d_act}."

    def _get_kernel(self, dim, cov):
        return MVNKernel(dim, cov)
