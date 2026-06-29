"""Uniform B-spline curves"""

import numpy as np
from numpy.typing import NDArray

# blending matrices for uniform splines
BM = {
    1: np.array([[1, 0], [-1, 1]]),
    2: np.array([[1, 1, 0], [-2, 2, 0], [1, -2, 1]]) / 2,
    3: np.array([[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]) / 6,
}

# matrices for curve derivative (direction) for powers(d-1)
BMdt = {
    1: np.array([[-1, 1]]),
    2: np.array([[-1, 1, 0], [1, -2, 1]]),
    3: np.array([[-1, 0, 1, 0], [2, -4, 2, 0], [-1, 3, -3, 1]]) / 2,
}


def powers_t(d: int, t: float):
    exps = np.arange(d + 1)
    return t**exps


def powers_tt(d: int, t: NDArray):
    exps = np.arange(d + 1)
    return t[..., None] ** exps[None, ...]


def splines_t(d: int, t: float):
    """[B_i-d ... B_i] = [0, 1, ... t^d] M"""
    return powers_t(d, t) @ BM[d]


def splines_tt(d: int, tt: NDArray):
    """[B_i-d ... B_i] = [0, 1, ... t^d] M"""
    return powers_tt(d, tt) @ BM[d]


def controls_i(d: int, points: NDArray, i: int):
    """Slice of control points for i'th span of t
    i.e. for a spline starting at t_i
    """
    assert i >= d and i < len(points)
    return points[i - d : i + 1]


def curve_t(d: int, controls: NDArray, t: float):
    """A point on curve at t, with 0 < t < 1"""
    assert controls.shape[0] == d + 1
    return powers_t(d, t) @ BM[d] @ controls


def curve_tt(d: int, controls: NDArray, tt: NDArray):
    """A segment of curve over tt, with 0 < t < 1"""
    assert controls.shape[0] == d + 1
    return powers_tt(d, tt) @ BM[d] @ controls


def megacurve(d: int, points: NDArray, tt: NDArray):
    wm = powers_tt(d, tt) @ BM[d]
    return np.concat([wm @ controls_i(d, points, i) for i in range(d, len(points))])


def flow_t(d: int, controls: NDArray, t: float):
    """A direction of curve at t, with 0 < t < 1"""
    assert controls.shape[0] == d + 1
    return powers_t(d - 1, t) @ BMdt[d] @ controls


def megaflow(d: int, points: NDArray, tt: NDArray):
    wm = powers_tt(d - 1, tt) @ BMdt[d]
    return np.concat([wm @ controls_i(d, points, i) for i in range(d, len(points))])
