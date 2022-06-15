"""
Implements the Nelder-Mead optimisation heuristics.
For more info, see https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

The main objects of this module are:
 * `nm(function_to_minimise, initial_simplex, params)` -- the Nelder-Mead algorithm entry point
 * `NMResult` -- representation of the algorithm's result
 * `default_initial_simplex(initial_point)` -- a method to provide the initial simplex if no better choice

The function `_demonstrate`, executed if this module is run directly, showcases the algorithm on x**2 and
Himmelblau's function.
"""

from typing import Callable, NamedTuple, Union

import jax
from jax import lax
from jax import numpy as jnp

jfloat = Union[float, jnp.ndarray]
jint = Union[int, jnp.ndarray]
jbool = Union[bool, jnp.ndarray]


class _NMState(NamedTuple):
    x: jfloat  # vertices of the simplex; assumed to be sorted wrt f, ie, f(x[0]) < f(x[1]) < ... < f(x[x.shape[0]-1])
    f: jfloat  # function values of the vertices
    i: jint  # iterations


class NMResult(NamedTuple):
    result_x: jfloat  # the final point found by the method
    result_f: jfloat  # the function value of the final point
    success: jbool
    # false if the algorithm terminated by iteration limit exceeded
    # true if by stddev of func values of simplex being small
    last_state: _NMState


# geometrical primitives
centroid = lambda v: v.sum(axis=0) / v.shape[0]
line = lambda x1, x2, t: x1 + t * (x2 - x1)  # for t in [0, 1] is line between x1 and x2


def reorder(state: _NMState) -> _NMState:
    """Maintains the invariant for NM that vertices of simplex are ordered according to their func value"""
    new_order = jnp.argsort(state.f)
    nx = state.x[new_order]
    nf = state.f[new_order]
    return state._replace(x=nx, f=nf)


def replace(state: _NMState, new_point: jfloat, new_point_val: jfloat) -> _NMState:
    """Replaces the vertex of the simplex with the highest func value with the new point, and re-orders."""
    nx = state.x.at[state.x.shape[0] - 1].set(new_point)
    nf = state.f.at[state.x.shape[0] - 1].set(new_point_val)
    return reorder(state._replace(x=nx, f=nf))


def shrink(state: _NMState, f: Callable) -> _NMState:
    """The shrkining step: replace all xi by .5*(x1+xi) and re-order."""
    nx = (state.x + state.x[0]) * 0.5
    nf = lax.map(f, state.x)
    return reorder(state._replace(x=nx, f=nf))


def nm_simplex_transform(state: _NMState, f: Callable) -> _NMState:
    """The core of the NM algorithm."""
    n = state.x.shape[0]
    fb = state.f[0]
    xw = state.f[n - 1]  # worst simplex point
    fw = state.f[n - 1]  # worst simplex point value
    fs = state.f[n - 2]  # second worst simplex point value
    xc = centroid(state.x)
    xr = line(xc, xw, -1.0)  # reflection of worst point around centroid
    fr = f(xr)

    # beware :) the following should be read from the end, ie, the declaration order is reverse of the decision tree order

    # else if rf ≥ f(sx) # must hold at this stage
    #   if f(sx) ≤ rf < f(wx) → ox = line(centroid, wx, -.5), if of < rf → replace wx by ox, re-order, continue iterating; else shrink
    #   else → ix = line(centroid, wx, .5), if if < wf → replace wx by ix, re-order, continue iterating; else shrink
    attempt_contracting = lambda state: lax.cond(
        ((fr <= fs) & (fr < fw)).all(),
        lambda state: lax.cond(
            (f(line(xc, xw, -0.5)) < fr).all(),
            lambda state: replace(state, line(xc, xw, -0.5), f(line(xc, xw, -0.5))),
            lambda state: shrink(state, f),
            state,
        ),
        lambda state: lax.cond(
            (f(line(xc, xw, 0.5)) < fw).all(),
            lambda state: replace(state, line(xc, xw, 0.5), f(line(xc, xw, 0.5))),
            lambda state: shrink(state, f),
            state,
        ),
        state,
    )

    # else if rf < f(x1) → ex = line(centroid, wx, -2), if ef < rf → replace wx by ef else by rf; re-order, continue iterating
    attempt_expanding = lambda state: lax.cond(
        (fr < fb).all(),
        lambda state: lax.cond(
            (f(line(xc, xw, -2.0)) < fr).all(),
            lambda state: replace(state, line(xc, xw, -2.0), f(line(xc, xw, -2.0))),
            lambda state: replace(state, xr, fr),
            state,
        ),
        attempt_contracting,
        state,
    )
    # if rf between f(x1) and f(sx) → replace wx by rx, re-order, continue iterating
    state = lax.cond(
        ((fr < fs) & (fr >= fb)).all(),
        lambda state: replace(state, xr, fr),
        attempt_expanding,
        state,
    )
    return state._replace(i=state.i + 1)


def nm(function: Callable, initial_simplex: jfloat, stddev_termination_tolerance: float, max_iter: int) -> jfloat:
    cond_rslt = lambda state: jnp.std(state.f) < stddev_termination_tolerance
    cond_iter = lambda state: state.i < max_iter
    cond_term = lambda state: ((~cond_rslt(state)) & cond_iter(state)).all()

    state0 = _NMState(initial_simplex, lax.map(function, initial_simplex), jnp.array([0]))

    nm_simplex_transform_f = lambda state: nm_simplex_transform(state, function)
    stateT = lax.while_loop(cond_term, nm_simplex_transform_f, state0)
    return NMResult(stateT.x[0], stateT.f[0], cond_rslt(stateT).all(), stateT)


def default_initial_simplex(x0: jfloat, step: float = 1.0) -> jfloat:
    """First vertex is the x0, others are x0 + e_i * step"""
    n = x0.shape[0]
    return lax.concatenate([x0.reshape(1, n), step * jnp.eye(n) + x0], dimension=0)


def _demonstrate():
    f1 = lambda x: jnp.power(x[0], 2)
    x1 = jnp.array([-1.0])
    r1 = nm(f1, default_initial_simplex(x1, 2.0), 0.1, 100)
    print(r1)  # this fails funnily because the function is symmetrical
    r1a = nm(f1, default_initial_simplex(x1, -1.0), 1.0e-10, 100)
    print(r1a)

    f2 = lambda x: jnp.power(jnp.power(x[0], 2) + x[1] - 11, 2) + jnp.power(x[0] + jnp.power(x[1], 2) - 7, 2)
    x2 = jnp.array([1.0, 1.0])
    r2 = nm(f2, default_initial_simplex(x2, 5.0), 1.0e-10, 100)
    print(r2)  # kinda failure, it collapses into a non-optimal point
    r2a = nm(f2, jnp.array([[2.0, 1.5], [3.5, 2.25], [3.25, 2.5]]), 1.0e-10, 100)
    print(r2a)  # a bit better, but still fails to find the optimum


if __name__ == "__main__":
    _demonstrate()
