"""
Eventually, will contain solver of a least-squares function minimisation.

For now, just snippets for approximating the ∇²f via the residual matrix and equation solving.
"""

from __future__ import annotations

from typing import Callable, Union

import jax
from jax import numpy as jnp

jfloat = Union[float, jnp.ndarray]

# lsf is a least squares function, ie, ½⋅Σr²_i(x) where the r_i are called residuals


def lsf_vects(residuals: list[Callable], point: jfloat) -> jfloat:
    return jnp.array([jnp.power(r(point), 2) for r in residuals])


def lsf_value(residuals: list[Callable], point: jfloat) -> jfloat:
    return lsf_vects(residuals, point).sum() * 0.5


def lsf_jacob(residuals: list[Callable], point: jfloat) -> jfloat:
    return jnp.array([jax.grad(r)(point) for r in residuals])


def lsf_deriv(residuals: list[Callable], point: jfloat) -> jfloat:
    return jnp.dot(lsf_jacob(residuals, point).transpose(), lsf_vects(residuals, point))


def lsf_ahess(residuals: list[Callable], point: jfloat) -> jfloat:
    jacob = lsf_jacob(residuals, point)
    return jnp.matmul(jacob.transpose(), jacob)


def lsf_descd(residuals: list[Callable], point: jfloat) -> jfloat:
    return jnp.linalg.solve(lsf_ahess(residuals, point), -lsf_deriv(residuals, point))
