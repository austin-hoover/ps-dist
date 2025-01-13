import numpy as np
import scipy.optimize

from ..hist import Histogram
from ..hist import Histogram1D


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (y_fit, slope, intercept) from linear fit."""

    def function(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        return slope * x + intercept

    popt, pcov = scipy.optimize.curve_fit(function, x, y)
    slope, intercept = popt
    y_fit = function(x, slope, intercept)
    return (y_fit, slope, intercept)


def fit_normal(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float, float, float, float]:
    """Return (yfit, sigma, mu, amplitude, offset) from Gaussian fit."""

    def function(x, sigma, mu, amplitude, offset):
        amplitude = amplitude / (sigma * np.sqrt(2.0 * np.pi))
        return offset + amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    popt, pcov = scipy.optimize.curve_fit(function, x, y)
    sigma, mu, amplitude, offset = popt
    y_fit = function(x, slope, intercept)
    return (y_fit, sigma, mu, amplitude, offset)
