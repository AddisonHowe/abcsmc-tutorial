import pytest
import numpy as np
import scipy.stats as st
from abcsmc.models import GaussianModel, RingModel, EllipsoidModel, BananaModel

"""
Test Script

"""

class TestGaussianModel:

    @pytest.mark.parametrize("x, theta", [
        [0, [[0, 1]]],
        [0, [[0, 1]]],
        [0, [[0, 1], [0, 2], [1, 1]]],
    ])
    def test_loglikelihood_singleton(self, x, theta):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            ll_exp[i] = st.norm(th[0], np.sqrt(th[1])).logpdf(x)
        g = GaussianModel(0, 0)
        ll = g.loglikelihood(x, np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"

    @pytest.mark.parametrize("x, theta", [
        [[0, 1, 2], [[0, 1]]],
        [[0, 1, 2], [[0, 1], [0, 2], [1, 1]]],
        [[0, 2], [[0, 1], [0, 2], [1, 1]]],
        [[0], [[0, 1], [0, 2], [1, 1]]],
    ])
    def test_loglikelihood_vector(self, x, theta):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            ll_exp[i] = np.sum(st.norm(th[0], np.sqrt(th[1])).logpdf(x))
        g = GaussianModel(0, 0, ndraws=len(x))
        ll = g.loglikelihood(np.array(x), np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"


class TestRingModel:

    @pytest.mark.parametrize("x, theta", [
        [0, [[0, 1]]],
        [0, [[0, 1]]],
        [0, [[0, 1], [0, 2], [1, 1]]],
    ])
    @pytest.mark.parametrize('var', [2, 3])
    def test_loglikelihood_singleton(self, x, theta, var):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            mu = th[0] * th[0] + th[1] * th[1]
            sig = np.sqrt(var)
            ll_exp[i] = np.sum(st.norm(mu, sig).logpdf(x))
        g = RingModel(0, 0, var=var)
        ll = g.loglikelihood(x, np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"

    @pytest.mark.parametrize("x, theta", [
        [[0, 1, 2], [[0, 1]]],
        [[0, 1, 2], [[0, 1], [0, 2], [1, 1]]],
        [[0, 2], [[0, 1], [0, 2], [1, 1]]],
        [[0], [[0, 1], [0, 2], [1, 1]]],
    ])
    @pytest.mark.parametrize('var', [2, 3])
    def test_loglikelihood_vector(self, x, theta, var):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            mu = th[0] * th[0] + th[1] * th[1]
            sig = np.sqrt(var)
            ll_exp[i] = np.sum(st.norm(mu, sig).logpdf(x))
        g = RingModel(0, 0, var=var, ndraws=len(x))
        ll = g.loglikelihood(np.array(x), np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"


class TestEllipsoidModel:

    @pytest.mark.parametrize("x, theta", [
        [0, [[0, 1]]],
        [0, [[0, 1]]],
        [0, [[0, 1], [0, 2], [1, 1]]],
    ])
    @pytest.mark.parametrize('var', [2, 3])
    def test_loglikelihood_singleton(self, x, theta, var):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            mu = (th[0] - 2*th[1])**2 + (th[1] - 4)**2
            sig = np.sqrt(var)
            ll_exp[i] = np.sum(st.norm(mu, sig).logpdf(x))
        g = EllipsoidModel(0, 0, var=var)
        ll = g.loglikelihood(x, np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"

    @pytest.mark.parametrize("x, theta", [
        [[0, 1, 2], [[0, 1]]],
        [[0, 1, 2], [[0, 1], [0, 2], [1, 1]]],
        [[0, 2], [[0, 1], [0, 2], [1, 1]]],
        [[0], [[0, 1], [0, 2], [1, 1]]],
    ])
    @pytest.mark.parametrize('var', [2, 3])
    def test_loglikelihood_vector(self, x, theta, var):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            mu = (th[0] - 2*th[1])**2 + (th[1] - 4)**2
            sig = np.sqrt(var)
            ll_exp[i] = np.sum(st.norm(mu, sig).logpdf(x))
        g = EllipsoidModel(0, 0, var=var, ndraws=len(x))
        ll = g.loglikelihood(np.array(x), np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"


class TestBananaModel:

    @pytest.mark.parametrize("x, theta", [
        [[2, 3], [[0, 1]]],
        [[2, 3], [[0, 2]]],
        [[2, 3], [[0, 1], [0, 2], [1, 1]]],
    ])
    @pytest.mark.parametrize('cov', [[[1, 0],[0, 0.5]]])
    def test_loglikelihood_singleton(self, x, theta, cov):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            mu = np.array([th[0], th[0] + th[1] * th[1]])
            ll_exp[i] = np.sum(st.multivariate_normal(mu, cov).logpdf(x))
        g = BananaModel(0, 0, cov=cov)
        ll = g.loglikelihood(x, np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"

    @pytest.mark.parametrize("x, theta", [
        [[[0, 1], [2, 1], [3, 1]],  [[0, 1]]],
        [[[0, 1], [2, 1], [3, 1]],  [[0, 1], [0, 2], [1, 1]]],
        [[[0, 1], [2, 1]],          [[0, 1], [0, 2], [1, 1]]],
        [[[0, 1]],                  [[0, 1], [0, 2], [1, 1]]],
    ])
    @pytest.mark.parametrize('cov', [[[1, 0],[0, 0.5]]])
    def test_loglikelihood_vector(self, x, theta, cov):
        n = len(theta)
        ll_exp = np.zeros(n)
        for i, th in enumerate(theta):
            mu = np.array([th[0], th[0] + th[1] * th[1]])
            ll_exp[i] = np.sum(st.multivariate_normal(mu, cov).logpdf(x))
        g = BananaModel(0, 0, cov=cov, ndraws=len(x))
        ll = g.loglikelihood(np.array(x), np.array(theta))
        assert np.allclose(ll, ll_exp), f"Expected:\n{ll_exp}\nGot:\n{ll}"
