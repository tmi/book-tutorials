from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import grad, lax
from jax.scipy.optimize import minimize as jinimize

# utils

key = jax.random.PRNGKey(0)
idn = lambda x: x
hessian = lambda f: jax.jacfwd(jax.jacrev(f))
cube = lambda v: v * v * v
jfloat = Union[float, jnp.ndarray]
jbool = Union[bool, jnp.ndarray]
jint = Union[int, jnp.ndarray]


def random_psd(dim: int, key):
    mtx = jax.random.normal(key, shape=(dim, dim))
    return jnp.matmul(mtx, mtx.transpose())


def random_vec(dim: int, key):
    return jax.random.normal(key, shape=(dim,))


# test case

m0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
m0_x_true = jnp.array([1.0, 1.0])
b0 = jnp.array([1.0, 1.0])

key, key1 = jax.random.split(key)
m1 = random_psd(10, key1)
key, key2 = jax.random.split(key)
m1_x_true = random_vec(10, key1)
b1 = jnp.matmul(m1, m1_x_true)

# methods
class Result(NamedTuple):
    x: jfloat
    i: jint
    s: jbool


def minimise_conjugate_direction(m, b, x0):
    conjugate_set = jla.eigh(m)[1].transpose()
    r = lambda x: jnp.matmul(m, x) - b

    def body(xk, i):
        pk = conjugate_set[i]
        α = -jnp.dot(r(xk), pk) / jnp.dot(pk, jnp.matmul(m, pk))
        xkp = xk + α * pk
        return xkp, None

    iterations = jnp.arange(0, b.shape[0])
    result = lax.scan(body, x0, iterations)[0]
    return Result(x=result, i=b.shape[0], s=jnp.array([True]).all())


def minimise_conjugate_descent(m, b, x0):
    class _CDState(NamedTuple):
        xk: jfloat
        pk: jfloat
        iter: jint

    r = lambda x: jnp.matmul(m, x) - b
    p0 = -r(x0)
    state0 = _CDState(x0, p0, jnp.array([0]))
    # in theory only n iters should suffice, but in practice does not seem to -- exactness of arithmetics?
    max_iter = b.shape[0] * 2

    cond_rslt = lambda state: jnp.abs(r(state.xk)) > 0.0000001
    cond_iter = lambda state: state.iter < max_iter
    cond = lambda state: (cond_rslt(state) & cond_iter(state)).all()

    def body(state: _CDState) -> _CDState:
        α = -jnp.dot(r(state.xk), state.pk) / jnp.dot(state.pk, jnp.matmul(m, state.pk))
        xkp = state.xk + α * state.pk
        βkp = jnp.dot(r(xkp), jnp.matmul(m, state.pk)) / jnp.dot(state.pk, jnp.matmul(m, state.pk))
        pkp = -r(xkp) + βkp * state.pk
        return state._replace(xk=xkp, pk=pkp, iter=state.iter + 1)

    stateE = lax.while_loop(cond, body, state0)
    return Result(stateE.xk, stateE.iter, cond_iter(stateE).all())


def minimise_conjugate_descent_v2(m, b, x0):
    # it differns from the original in formulae for βk and αk... but there must be a bug as the results are wrong
    class _CDState(NamedTuple):
        xk: jfloat
        pk: jfloat
        iter: jint

    r = lambda x: jnp.matmul(m, x) - b
    p0 = -r(x0)
    state0 = _CDState(x0, p0, jnp.array([0]))
    max_iter = b.shape[0] * 2

    cond_rslt = lambda state: jnp.abs(r(state.xk)) > 0.0000001
    cond_iter = lambda state: state.iter < max_iter
    cond = lambda state: (cond_rslt(state) & cond_iter(state)).all()

    def body(state: _CDState) -> _CDState:
        α = -jnp.dot(r(state.xk), r(state.xk)) / jnp.dot(state.pk, jnp.matmul(m, state.pk))
        xkp = state.xk + α * state.pk
        βkp = jnp.dot(r(xkp), r(xkp)) / jnp.dot(r(state.xk), r(state.xk))
        pkp = -r(xkp) + βkp * state.pk
        return state._replace(xk=xkp, pk=pkp, iter=state.iter + 1)

    stateE = lax.while_loop(cond, body, state0)
    return Result(stateE.xk, stateE.iter, cond_iter(stateE).all())


# test suite
def test_suite():
    compare = lambda x_true, x_pred: print(
        f"{x_true =}\n{x_pred.x =}\ndifference: {jla.norm(x_true - x_pred.x)}\n{x_pred.i = }\n"
    )

    print("conjudage direction")
    x0i = minimise_conjugate_direction(m0, b0, jnp.zeros(b0.shape[0]))
    compare(m0_x_true, x0i)
    x1i = minimise_conjugate_direction(m1, b1, jnp.zeros(b1.shape[0]))
    compare(m1_x_true, x1i)
    print("\n---\n")

    print("conjudage descent")
    x0e = minimise_conjugate_descent(m0, b0, jnp.zeros(b0.shape[0]))
    compare(m0_x_true, x0e)
    x1e = minimise_conjugate_descent(m1, b1, jnp.zeros(b1.shape[0]))
    compare(m1_x_true, x1e)

    print("conjudage descent v2")
    x0v = minimise_conjugate_descent_v2(m0, b0, jnp.zeros(b0.shape[0]))
    compare(m0_x_true, x0v)
    x1v = minimise_conjugate_descent_v2(m1, b1, jnp.zeros(b1.shape[0]))
    compare(m1_x_true, x1v)


if __name__ == "__main__":
    test_suite()
