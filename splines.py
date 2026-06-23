"""Uniform B-spline curves"""

import numpy as np
from numpy.typing import NDArray

M2 = np.array([[1, 1, 0], [-2, 2, 0], [1, -2, 1]]) / 2

M3 = np.array([[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]) / 6


def powers(d: int, t: NDArray):
    """Power vector: [1, t, ... t^2]"""
    exps = np.arange(d + 1)
    return t[..., None] ** exps[None, ...]


def segm_points(d: int, points: NDArray, i: int):
    """Points for a segment: C_(i-d)^n"""
    assert i >= d and i < len(points)
    return points[i - d : i + 1]


def segm_curve(d: int, M: NDArray, points: NDArray, i: int, tt: NDArray):
    """Curve for a segment: P M C"""
    return powers(d, tt) @ M @ segm_points(d, points, i)


def mega_curve(d, M, points, tt):
    """Curve for all along"""
    pm = powers(d, tt) @ M
    return np.concat([pm @ segm_points(d, points, j) for j in range(d, len(points))])
