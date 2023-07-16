from scipy.optimize import curve_fit
import numpy as np
from numpy.typing import ArrayLike


def _gaussian(x: ArrayLike, sigma: float, mu: float) -> ArrayLike:
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-mu)/sigma, 2))


def gauss_fit(x: ArrayLike, y: ArrayLike) -> dict:
    if len(x) != len(y):
        raise ValueError('x and y need to be of equal length')
    if len(x) < 3:
        raise ValueError('The amount of datapoints must be larger \
            than the two fit parameters')
    fit, _ = curve_fit(_gaussian, x, y)
    return {'sigma': fit[0], 'mu': fit[1]}


def gauss_eval(x: list, fit: dict) -> list:
    if 'mu' not in fit or 'sigma' not in fit:
        raise ValueError('mu and sigma are required in the fit dictionary')
    return _gaussian(x, fit['sigma'], fit['mu'])
