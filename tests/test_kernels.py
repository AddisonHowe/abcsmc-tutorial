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
        density = kernel.pdf(np.array(particle1), np.array(particle0))
        assert density == density_exp, f"Expected {density_exp}. Got {density}."

    @pytest.mark.parametrize('dim, dx, particles1, particles0, density_exp', [
        [2, 1, [[-0.10, 0], [ 0.10, 0]], [[1, 0]], [[0.00], [0.25]]],
    ])
    def test_vpdf(self, dim, dx, particles1, particles0, density_exp):
        kernel = self._get_kernel(dim, dx)
        density = kernel.vpdf(np.array(particles1), np.array(particles0))
        assert np.all(density == np.array(density_exp)), \
            f"Expected {density_exp}. Got {density}."

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

    @pytest.mark.parametrize('dim, sigmas, particles1, particles0', [
        [2, [1, 2], [[-0.10, 0], [ 0.10, 0]], [[1, 0]]],
    ])
    def test_vpdf(self, dim, sigmas, particles1, particles0):
        kernel = self._get_kernel(dim, sigmas)
        density = kernel.vpdf(np.array(particles1), np.array(particles0))
        d_exp = np.zeros([len(particles1), len(particles0)])
        for j, p0 in enumerate(particles0):
            norm = st.multivariate_normal(mean=p0, cov=np.diag(sigmas)**2)
            for i, p1 in enumerate(particles1):
                d_exp[i,j] = norm.pdf(p1)
        assert d_exp.shape == density.shape, "Wrong shape"                 
        assert np.allclose(d_exp, density), f"Expected {d_exp}. Got {density}."

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
    
    @pytest.mark.parametrize('dim, cov, particles1, particles0', [
        [2, [[1, 0.8], [0.8, 4]], [[-0.10, 0], [ 0.10, 0]], [[1, 0]]],
    ])
    def test_vpdf(self, dim, cov, particles1, particles0):
        kernel = self._get_kernel(dim, cov)
        density = kernel.vpdf(np.array(particles1), np.array(particles0))
        d_exp = np.zeros([len(particles1), len(particles0)])
        for j, p0 in enumerate(particles0):
            norm = st.multivariate_normal(mean=p0, cov=cov)
            for i, p1 in enumerate(particles1):
                d_exp[i,j] = norm.pdf(p1)
        assert d_exp.shape == density.shape, "Wrong shape"                 
        assert np.allclose(d_exp, density), f"Expected {d_exp}. Got {density}."

    def _get_kernel(self, dim, cov):
        return MVNKernel(dim, cov)
    

class TestLOCM:

    @pytest.mark.parametrize('dim, cov, particle1, particle0', [
        [2, [[1, 0.8], [0.8, 4]], [-0.10, 0], [1, 0]],
        [2, [[1, 0.8], [0.8, 4]], [ 0.10, 0], [1, 0]],
    ])
    def test_pdf(self, dim, cov, particle1, particle0):
        particles0 = np.array(particle0)[None,:]
        covs = np.array(cov)[None,:]
        kernel = self._get_kernel(dim, particles0, covs)
        density = kernel.pdf(np.array(particle1), np.array(particle0), idx=0)
        d_exp = st.multivariate_normal(mean=particle0, cov=cov).pdf(particle1)
        assert np.allclose(density, d_exp), f"Expected {d_exp}. Got {density}."
    
    @pytest.mark.parametrize('dim, particles0, covs, particles1, idxs', [
        [2, 
         [[1, 0]], 
         [[[1, 0.8], [0.8, 4]]], 
         [[-0.10, 0], [0.10, 0]],
         [0, 0]
        ],
    ])
    def test_vpdf(self, dim, particles0, covs, particles1, idxs):
        kernel = self._get_kernel(dim, particles0, covs)
        density = kernel.vpdf(
            np.array(particles1), np.array(particles0), idx=idxs
        )        
        d_exp = np.zeros([len(particles1), len(idxs)])
        for j, idx in enumerate(idxs):
            norm = st.multivariate_normal(mean=particles0[idx], cov=covs[idx])
            for i, p1 in enumerate(particles1):
                d_exp[i,j] = norm.pdf(p1)
        assert d_exp.shape == density.shape, "Wrong shape"                 
        assert np.allclose(d_exp, density), f"Expected {d_exp}. Got {density}."

    def _get_kernel(self, dim, particles0, covs):
        return LOCMKernel(dim, np.array(particles0), np.array(covs))
