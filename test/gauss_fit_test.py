from example_gauss_fit.example_gauss_fit import gauss_fit, gauss_eval
import pytest
import numpy as np


def test_gauss_fit():
    def gaussian(x, sigma, mu):
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-mu)/sigma, 2))
    mu = 2
    sigma = 3
    x = np.linspace(-10, 10, 10000)
    y = gaussian(x, sigma, mu)
    res = gauss_fit(x, y)

    assert abs(res['sigma'] - sigma) < 0.01 and abs(res['mu'] - mu) < 0.01


def test_unequal_length():
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 9)
    with pytest.raises(ValueError):
        _ = gauss_fit(x, y)


def test_short_arrays():
    x = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    with pytest.raises(ValueError):
        _ = gauss_fit(x, y)


def test_evaluation():
    def gaussian(x, sigma, mu):
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-mu)/sigma, 2))
    mu = -1
    sigma = 2
    x = np.linspace(-10, 10, 10)
    y_compare = gaussian(x, sigma, mu)
    y = gauss_eval(x, {'sigma': sigma, 'mu': mu})
    assert (y == y_compare).all()


def test_evaluation_without_sigma():
    x = np.linspace(-10, 10, 10)
    with pytest.raises(ValueError):
        _ = gauss_eval(x, {'mu': 1})
