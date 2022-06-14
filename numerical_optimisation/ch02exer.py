import jax
import jax.numpy as jnp

import functions


def ex2_1():
    """Compute Gradient and Hessian of Rosenbrock function (functions.f3).
    Show that (1, 1) is the only local minimizer, and that the Hessian at that point is positive definite."""

    f = functions.f3
    gf = jax.grad(f)
    # instead of explicit hessian, we can compute a vector product with hessian; this is suff in many cases
    h_vp = lambda x0, v: grad(lambda x: jnp.vdot(gf(x), v))(x0)
    h_pd_test = lambda x0, v: jnp.vdot(v, h_vp(x0, v)) > 0  # test a single vector xMx > 0

    # full hessian
    h_full = jax.jacfwd(jax.jacrev(f))
    is_sym = lambda x: (h_full(x) == h_full(x).transpose()).all()
    is_pd = lambda x: (jnp.linalg.eigh(h_full(0))[0] > 0).all()

    v1 = jnp.array([1.0, 1.0])
    v0 = jnp.array([0.0, 0.0])

    # v1 is a strict local minimiser
    first = (gf(v1) == v0).all()
    second = jnp.logical_and(is_sym(v1), is_pd(v1))

    # uniqueness -- thats harder :)


def ex2_2():
    """Show that 8x1 + 12x2 + x1^2 - 2x2^2 has a single saddlepoint that is neither max nor min."""
    f = lambda x: 8 * x[0] + 12 * x[1] + jnp.pow(x[0], 2) - 2 * jnp.pow(x[1], 2)
    # no idea how to do that numerically...


def ex2_4():
    """Write second-order Taylor for cos(1/x) and third-order Taylor for cos(x). Eval for x = 1"""
    h_vp = lambda f, x0, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x0)
    taylor2 = lambda f, x0, p: f(x0) + jnp.dot(jax.grad(f)(x0), p) + 0.5 * h_vp(f, x0, p)


def ex2_9():
    """Show that for (x1 + x2^2)^2 at point (1,0) is the direction (-1,1) a minimizer"""
    f = lambda x: jnp.power(x[0] + jnp.power(x[1], 2), 2)
    g = jax.grad(jnp.array([1.0, 0.0]))  # 2, 0
    t = jnp.array([-1.0, 1.0])
    d1 = -g / jnp.linalg.norm(-g)
    d2 = t / jnp.linalg.norm(t)
    angle = jnp.arccos(d1, d2)
    jnp.abs(angle) < jnp.pi / 2  # still points in the correct half of the tangent hyperplane

    """Find all minimisers of min f(x + alpha dir) for the above"""
    f1 = lambda x, a: f(jnp.array([1.0, 0.0]) + a * jnp.array([-1.0, 1.0]))
