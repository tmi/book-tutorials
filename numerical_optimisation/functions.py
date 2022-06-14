import jax
import jax.numpy as jp
import jax.random as jandom

# signal
# f = x_1 + x_2 * e ^ {-(x_3 - t)^2 / x_4} + x_5 cos(x_6 t) -- the model
# y = y_t -- the observatinos
# r(t_i) = y - f -- the residuals
# min \sum r^2(t_i)


def f1_f(seed, n):
    """A function (factory) for fitting a model to noisy observations.
    Jax is currently failing on this (for n>2) due to line search failing to zoom.
    For small n, this is quite degenerate -- the noise trumps the seed solution, and the optimisation overfits."""

    f_model = lambda x: (lambda t: x[0] + x[1] * jp.exp(-jp.power(x[2] - t, 2) / x[3]) + x[4] * jp.cos(x[5] * t))
    key = jandom.PRNGKey(seed)
    key1, key2 = jandom.split(key)
    solution_points = jandom.randint(key2, shape=(6,), minval=3, maxval=6)
    solution_func = f_model(solution_points)
    time = jp.linspace(1, n, n)
    obser_noise = jandom.normal(key1, shape=(n,)) / 100.0
    obser_vals = solution_func(time) + obser_noise
    residuals = lambda x: jp.sum(jp.power(obser_vals - f_model(x)(time), 2))
    return residuals, solution_points


f1, _ = f1_f(0, 100)


def f2(x):
    """The global minimiser is non-isolated minimiser (in particular, this is not convex)
    Local algorithms are not expected to find a global minimum"""
    return (jp.power(x, 4) * (jp.cos(1.0 / x) + 2))[0]  # todo continuation at 0?? (func undefined, grad nan...)


def f3(x):
    """Rosenbrok function, 100(x_2 - x_1^2)^2 + (1-x_1)^2."""
    return 100 * jp.power(x[1] - jp.power(x[0], 2), 2) + jp.power(1 - x[0], 2)
