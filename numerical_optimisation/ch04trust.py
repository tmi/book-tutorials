from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jlinalg
from jax import grad, lax
from jax.scipy.optimize import minimize as jinimize

key = jax.random.PRNGKey(0)
key, key1 = jax.random.split(key)
key, key2 = jax.random.split(key)
idn = lambda x: x
hessian = lambda f: jax.jacfwd(jax.jacrev(f))
cube = lambda v: v * v * v
jfloat = Union[float, jnp.ndarray]
jbool = Union[bool, jnp.ndarray]
jint = Union[int, jnp.ndarray]

_TRState(NamedTuple):
    x: jfloat
    Δ: jfloat
    i: jint


def compute_cauchy_point(f, B, x, Δ):
    # B is hessian of f or appx thereof
    g = grad(f)(x0)
    gn = jlinalg.norm(g)
    ps = -g * Δ / gn
    e = jnp.dot(g, jnp.matmul(B, g))
    t = 1 if e <= 0 else min(1, cube(gn) / (Δ * e))
    return t * ps

def trust_region_iteration(f, B, m, compute_p, Δmax, ν):

    def body(state: _TRState) -> _TRState:
        p = compute_p(f, B, state.x, state.Δ)
        act_increase = f(state.x) - f(state.x + p)
        prd_increase = m(state.x) - m(state.x + p)
        ρ = act_increase / prd_increase
        if ρ < 0.25:
            Δn = state.Δ*0.25
        elif ρ > 0.75 and jlinalg.norm(p) == state.Δ:
            Δn = min(2*state.Δ, Δmax)
        else:
            Δn = state.Δ
        if ρ > ν:
            xn = state.x + p
        else:
            xn = state.x
        return state._replace(Δ=Δn, x=xn, i=state.i+1)
        
