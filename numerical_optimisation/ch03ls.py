from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import grad, lax
from jax.scipy.optimize import minimize as jinimize

key = jax.random.PRNGKey(0)
key, key1 = jax.random.split(key)
key, key2 = jax.random.split(key)
idn = lambda x: x


def random_psd(dim: int, key):
    mtx = jax.random.normal(key, shape=(dim, dim))
    return jnp.matmul(mtx, mtx.transpose())


def random_vec(dim: int, key):
    return jax.random.normal(key, shape=(dim,))


m1 = random_psd(5, key1)
v1 = random_vec(5, key2)

test_cases = [
    (lambda x: jnp.sum(jnp.power(x, 2)), jnp.array([11.0, 0.001]), "x^2"),
    (lambda x: jnp.sum(jnp.power(x, 2) + jnp.cos(x / 3.0) + x), jnp.array([2.0, 3.4, 0.5]), "x^2 + x + cos(x/3)"),
    (
        lambda x: jnp.sum(jnp.exp(x) * jnp.power(x, 4) + jnp.sin(1 / (x + 3))),
        jnp.array([7.0]),
        "x^4*e^x + sin(1/(x+3))",
    ),
    (
        lambda x: 0.5 * jnp.matmul(x, jnp.matmul(m1, x)) - jnp.matmul(v1, x),
        jnp.array([1.0, 1024.0, 0.001, 2.4, 34.0]),
        "xQx - bx",
        {"q": m1, "b": v1},
    ),
]


class _LsState(NamedTuple):
    itrs: Union[int, jnp.ndarray]
    alpha: Union[float, jnp.ndarray]


class _LsResult(NamedTuple):
    next: Union[float, jnp.ndarray]
    succ: Union[bool, jnp.ndarray]
    itrs: Union[int, jnp.ndarray]


def ls_backtracking(f, x0, direction, alpha_base, ro, c, max_itrs):
    cond_conv = lambda alpha: f(x0 + alpha * direction) > f(x0) + c * alpha * jnp.dot(grad(f)(x0), direction)
    cond_itrs = lambda itrs: itrs <= max_itrs
    cond = lambda state: (cond_conv(state.alpha) & cond_itrs(state.itrs)).all()
    body = lambda state: state._replace(itrs=state.itrs + 1, alpha=state.alpha * ro)

    state0 = _LsState(jnp.array([0]), jnp.array([alpha_base]))
    stateE = lax.while_loop(cond, body, state0)

    next = x0 + stateE.alpha * direction
    succ = (~cond_conv(stateE.alpha)).all()
    return _LsResult(next, succ, stateE.itrs)


def ls_quad_analytical(f, q, b, x0, drc):
    # assumes f = 0.5⋅xQx - bx
    step = -jnp.dot(grad(f)(x0), drc) / jnp.dot(drc, jnp.dot(q, drc))
    next = x0 + step * drc
    return _LsResult(next, jnp.array([True]).all(), 0)


class _CuInState(NamedTuple):
    app: Union[float, jnp.ndarray]
    ap: Union[float, jnp.ndarray]
    itrs: Union[int, jnp.ndarray]


def ls_cubic_interpolation(f, x0, direction, c, a0, max_itrs):
    f_uv = lambda a: f(x0 + a * direction)
    f_uv_gr = grad(f_uv)(0.0)
    cond = lambda a: (f_uv(a) <= f_uv(0.0) + c * a * f_uv_gr).all()

    quad_inter = lambda ap: -1 * f_uv_gr * ap * ap / (2 * (f_uv(ap) - f_uv(0.0) - f_uv_gr * ap))

    step1 = lax.cond(cond(a0), idn, quad_inter, a0)  # if a0 works then a0, else a1

    def cube_inter(state: _CuInState):
        ap, app = state.ap[0], state.app[0]
        aps = ap * ap
        apps = app * app
        coef = 1 / (apps * aps * (ap - app))
        mat1 = jnp.array([[apps, -aps], [-apps * app, aps * ap]])
        vec1 = jnp.array([f_uv(ap) - f_uv(0) - f_uv_gr * ap, f_uv(app) - f_uv(0) - f_uv_gr * app])
        cons = coef * jnp.matmul(mat1, vec1)
        cona, conb = cons[0], cons[1]
        anext = (-conb + jnp.power(conb * conb - 3 * cona * f_uv_gr, 0.5)) / (3 * cona)
        return _CuInState(state.ap, jnp.array([anext]), state.itrs + 1)

    condI = lambda state: ((~cond(state.ap)) & (state.itrs <= max_itrs)).all()

    # if either a1 or a0 works then thats the answer, else cube inter iteration
    # note if not cond(step1), then step1=a1
    # in case of a1 or a0 works, iterations are 0 which is bit confusing
    stepI = lax.cond(
        cond(step1),
        lambda x: _CuInState(jnp.array([a0]), jnp.array([x]), jnp.array([0])),
        lambda x: lax.while_loop(condI, cube_inter, _CuInState(jnp.array([a0]), jnp.array([x]), jnp.array([0]))),
        step1,
    )
    next = stepI.ap * direction + x0
    succ = cond(stepI.ap).all()
    return _LsResult(next=next, succ=succ, itrs=stepI.itrs)


def ls_strong_wolfe(f, x0, direction, c1, c2, alpha_max, max_itrs):
    # max_itrs is essentially considered twice -- in both the former growing and the latter zoom phase
    Φ = lambda α: f(x0 + α * direction)
    Φd = grad(Φ)

    class _ZoomState(NamedTuple):
        itrs: Union[int, jnp.ndarray]
        alow: Union[float, jnp.ndarray]
        ahig: Union[float, jnp.ndarray]
        succ: Union[bool, jnp.ndarray]

    def zoom_body(state: _ZoomState) -> _ZoomState:
        state = state._replace(itrs=state.itrs + 1)
        amid = (state.ahig + state.alow) / 2  # TODO instead of bisection, try quad/cub interp
        cond1 = ((Φ(amid) > Φ(0.0) + c1 * amid * Φd(0.0)) | (Φ(amid) >= Φ(state.alow))).all()
        brnc1 = lambda istate: istate._replace(ahig=amid)

        def brnc2(istate: _ZoomState) -> _ZoomState:
            istate = lax.cond(
                (jnp.abs(Φd(amid)) <= -c2 * Φd(0.0)).all(),
                lambda jstate: jstate._replace(ahig=amid, alow=amid, succ=jnp.array([True]).all()),
                idn,
                istate,
            )
            istate = lax.cond(
                (Φd(amid) * (istate.ahig - istate.alow) >= 0.0).all(),
                lambda jstate: jstate._replace(ahig=jstate.alow),
                idn,
                istate,
            )
            istate = istate._replace(alow=amid)
            return istate

        return lax.cond(cond1, brnc1, brnc2, state)

    zoom_cond = lambda zoom_state: ((~zoom_state.succ) & (zoom_state.itrs < max_itrs)).all()

    class _LsState(NamedTuple):
        itrs: Union[int, jnp.ndarray]
        a: Union[float, jnp.ndarray]
        ap: Union[float, jnp.ndarray]
        succ: Union[bool, jnp.ndarray]
        succ_zoom: Union[bool, jnp.ndarray]

    def state_zoom_to_ls(ahig, alow, ls_state):
        res = lax.while_loop(
            zoom_cond, zoom_body, _ZoomState(itrs=jnp.array([0]), succ=jnp.array([False]).all(), alow=alow, ahig=ahig)
        )
        return ls_state._replace(
            a=res.alow, succ_zoom=res.succ, succ=jnp.array([True]).all(), itrs=ls_state.itrs + res.itrs
        )

    ls_a_next = lambda ap: (ap + alpha_max) / 2  # TODO instead of bisection, try quad/cub interp

    def ls_body(state: _LsState) -> _LsState:
        state = state._replace(itrs=state.itrs + 1)
        cond1 = ((Φ(state.a) > Φ(0.0) + c1 * state.a * Φd(0.0)) | ((Φ(state.a) >= Φ(state.ap)) & state.itrs > 1)).all()
        brnc1 = lambda istate: state_zoom_to_ls(istate.ap, istate.a, istate)
        state = lax.cond(cond1, brnc1, idn, state)
        cond2 = ((~state.succ) & (jnp.abs(Φd(state.a)) <= -c2 * Φd(0.0))).all()
        brnc2 = lambda istate: istate._replace(succ=jnp.array([True]).all(), succ_zoom=jnp.array([True]).all())
        state = lax.cond(cond2, brnc2, idn, state)
        cond3 = ((~state.succ) & (Φd(state.a) >= 0)).all()
        brnc3 = lambda istate: state_zoom_to_ls(istate.a, istate.ap, istate)
        state = lax.cond(cond3, brnc3, idn, state)
        anext = ls_a_next(state.ap)
        cond4 = (~state.succ).all()
        state = lax.cond(cond4, lambda istate: istate._replace(ap=istate.a, a=anext), idn, state)
        return state

    ls_cond = lambda ls_state: ((~ls_state.succ) & (ls_state.itrs < max_itrs)).all()

    ls_init = _LsState(
        itrs=jnp.array([0]), a=ls_a_next(0.0), ap=0.0, succ=jnp.array([False]).all(), succ_zoom=jnp.array([False]).all()
    )
    ls_res = lax.while_loop(ls_cond, ls_body, ls_init)

    return _LsResult(next=ls_res.a * direction + x0, itrs=ls_res.itrs, succ=(ls_res.succ & ls_res.succ_zoom).all())


def opt_base(f, x0, dir_method, step_method="backtrack", **kwargs):
    class _OptState(NamedTuple):
        xgrd: Union[float, jnp.ndarray]
        itrs: Union[int, jnp.ndarray]
        ddir: Union[float, jnp.ndarray]
        xcur: Union[float, jnp.ndarray]
        itrs_lsc: Union[int, jnp.ndarray]
        lsrs: Union[bool, jnp.ndarray]

    class _OptResult(NamedTuple):
        x: Union[float, jnp.ndarray]
        succ: Union[bool, jnp.ndarray]
        succ_ls: Union[bool, jnp.ndarray]
        nit: Union[int, jnp.ndarray]
        nit_lsc: Union[int, jnp.ndarray]

    max_itrs_ds = 200
    conv_stop = 0.00001  # TODO how this properly?
    cond_conv = lambda xgrd: jnp.max(jnp.abs(xgrd)) > conv_stop
    cond_itrs = lambda itrs: itrs <= max_itrs_ds
    cond = lambda state: (cond_conv(state.xgrd) & cond_itrs(state.itrs) & state.lsrs).all()

    # params for line search
    # for newton and quasi-newton, alpha_base should always be 1
    # steepest descent and conqugate gradient should choose alpha_base adhoc -- eg so that α₀∇f⋅dir = α_prev∇f_prev⋅dir_prev
    # c is like 10^-4
    if step_method == "backtrack":
        alpha_base = 1.0
        ro = 0.5
        c = 0.0001
        max_itrs_ls = 60
        lsres_f = lambda x_k, direction: ls_backtracking(f, x_k, direction, alpha_base, ro, c, max_itrs_ls)
    elif step_method == "cubic_interpol":
        alpha_base = 1.0
        c = 0.0001
        max_itrs_ls = 60
        lsres_f = lambda x_k, direction: ls_cubic_interpolation(f, x_k, direction, c, alpha_base, max_itrs_ls)
    elif step_method == "quadratic_analytic":
        lsres_f = lambda x_k, direction: ls_quad_analytical(f, kwargs["q"], kwargs["b"], x_k, direction)
    elif step_method == "strong_wolfe":
        # 0 < c1 < c2 < 1, alpha max > 0
        c1 = 0.0001
        c2 = 0.9
        alpha_max = 2.0
        max_itrs_ls = 60
        lsres_f = lambda x_k, direction: ls_strong_wolfe(f, x_k, direction, c1, c2, alpha_max, max_itrs_ls)
    else:
        raise NotImplementedError(f"not implemented step method {step_method}")

    def body(state):
        lsres = lsres_f(state.xcur, state.ddir)
        state = state._replace(
            lsrs=lsres.succ, xcur=lsres.next, itrs=state.itrs + 1, itrs_lsc=state.itrs_lsc + lsres.itrs
        )
        state = state._replace(ddir=dir_method(f, state.xcur), xgrd=grad(f)(state.xcur))
        return state

    state0 = _OptState(grad(f)(x0), jnp.array([0]), dir_method(f, x0), x0, jnp.array([0]), jnp.array([True]).all())
    stateE = lax.while_loop(cond, body, state0)

    return _OptResult(stateE.xcur, ~cond_conv(stateE.xgrd), stateE.lsrs, stateE.itrs, stateE.itrs_lsc)


def opt_steepest_descent(f, x0, **kwargs):
    mdir = lambda f, x: -grad(f)(x)
    return opt_base(f, x0, mdir, **kwargs)


def opt_newton_method(f, x0, **kwargs):
    # dir = - inv(∇²f)∇f
    mdir = lambda f, x: -jnp.matmul(jnp.linalg.inv(jax.jacfwd(jax.jacrev(f))(x)), grad(f)(x))
    return opt_base(f, x0, mdir, **kwargs)


def opt_quasinewton(f, x0):
    # dir = -Bk^-1 ⋅ ∇fk, where Bk is psd and symmetric, updated at every iteration
    # bfgs, others...?
    # TODO
    pass


def run_test_suite():
    methods = [
        (opt_newton_method, "newton_backtrack"),
        (opt_steepest_descent, "steepest_backtrack"),
        (lambda f, x0: jinimize(f, x0, method="bfgs"), "jax_bfgs"),
        (lambda f, x0: opt_newton_method(f, x0, step_method="cubic_interpol"), "newton_cubic_interpol"),
        (lambda f, x0: opt_steepest_descent(f, x0, step_method="cubic_interpol"), "steepest_cubic_interpol"),
        (lambda f, x0: opt_newton_method(f, x0, step_method="strong_wolfe"), "newton_strong_wolfe"),
        (lambda f, x0: opt_steepest_descent(f, x0, step_method="strong_wolfe"), "steepest_strong_wolfe"),
    ]
    fmt_one = lambda name, res, tc: f"{name} -> res: {tc[0](res.x)}, iters: {res.nit}"

    for tc in test_cases:
        print(tc[2])
        for m in methods:
            print(fmt_one(m[1], m[0](tc[0], tc[1]), tc))

    qf = test_cases[3]
    print(f"analytical linesearch for quadratic function {qf[2]}")
    res_sd = opt_steepest_descent(qf[0], qf[1], step_method="quadratic_analytic", **qf[3])
    print(fmt_one("steepest desc", res_sd, qf))
    res_nm = opt_newton_method(qf[0], qf[1], step_method="quadratic_analytic", **qf[3])
    print(fmt_one("newton method", res_nm, qf))


if __name__ == "__main__":
    # run_test_suite()
    pass
