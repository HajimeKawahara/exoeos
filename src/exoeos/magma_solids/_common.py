"""Small shared polynomial and implicit-order operations for dry solids."""

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import R, _tp, _berman_gibbs


def berman(T, P, rows):
    """Rows: Href, Sref, Vref, Cp[4], EOS[4]; Vref is J/bar/mol."""
    T, P = _tp(T, P)
    a = jnp.asarray(rows).T
    return _berman_gibbs(T, P, a[0], a[1], a[3:7], a[2], a[7:11])


def lambda_anomaly(T, tc, latent, l1, l2):
    """Berman lambda Cp=T*(l1+l2*T)**2 and latent H at tc."""
    tr, u = 298.15, jnp.minimum(T, tc)
    h = l1*l1*(u*u-tr*tr)/2+2*l1*l2*(u**3-tr**3)/3+l2*l2*(u**4-tr**4)/4
    s = l1*l1*(u-tr)+l1*l2*(u*u-tr*tr)+l2*l2*(u**3-tr**3)/3
    return h-T*s+jnp.where(T >= tc, latent*(1-T/tc), 0.)


def xlogx(x):
    # Continuous energy and fixed-face T/P derivatives at empty sites.
    # Composition derivatives at those boundaries are outside the API contract.
    return x * jnp.log(jnp.where(x > 0, x, 1.))


def polynomial(terms, size):
    """Compile static sparse terms; products avoid 0**negative AD at edges."""
    degree = max((sum(p) for _, p in terms), default=0)
    indices = [[i for i, power in enumerate(p) for _ in range(power)]
               + [size]*(degree-sum(p)) for _, p in terms]
    indices = np.asarray(indices, dtype=int).reshape(len(terms), degree)
    coefficients = np.asarray([c for c, _ in terms])

    def evaluate(x):
        extended = jnp.concatenate((x, jnp.ones(1, dtype=x.dtype)))
        return jnp.sum(jnp.asarray(coefficients)*jnp.prod(extended[indices], axis=1))

    return evaluate


def expression(data):
    """Return unreferenced molar site G and its affine site occupations."""
    size = data['size']
    h, s, v = (polynomial(data[key], size) for key in ('H', 'S', 'V'))
    sites = tuple(polynomial(p, size) for _, p in data['sites'])
    multiplicities = np.array([m for m, _ in data['sites']])

    def occupations(q):
        return jnp.stack([f(q) for f in sites]) if sites else jnp.ones(1)

    def energy(T, P, q):
        ideal = jnp.sum(jnp.asarray(multiplicities)*xlogx(occupations(q))) if sites else 0.
        return h(q)-T*s(q)+(P/1e5-1)*v(q)+R*T*ideal

    return energy, occupations


def relax(energy, initial, free, sites, starts=None):
    """Solve site stationarity with feasible damped Newton steps.

    energy is dimensionless (G/RT). Inactive coordinates have an identity
    equation. Only converged, positive-curvature roots are returned; failure
    gives NaN. Multiple starts select the lowest converged branch, without a
    claim of global phase stability. AD uses the stationarity equation.
    """
    free = jnp.asarray(free)

    def objective(q):
        return energy(q)+jnp.sum(jnp.where(free, 0., (q-initial)**2/2))

    gradient = jax.grad(objective)

    def solve(fun, guess):
        def run(start):
            def step(_, q):
                g = fun(q)
                h = jax.jacfwd(fun)(q)
                shift = jnp.maximum(0., 1e-5-jnp.linalg.eigvalsh(h)[0])
                direction = -jnp.linalg.solve(h+shift*jnp.eye(q.size), g)
                direction = direction/jnp.maximum(1., jnp.max(jnp.abs(direction)))
                value = objective(q)

                def backtrack(_, alpha):
                    trial = q+alpha*direction
                    feasible = jnp.all(sites(trial) >= 0.)
                    decrease = objective(trial) <= value+1e-4*alpha*(g@direction)+1e-14
                    return jnp.where(feasible & decrease, alpha, alpha/2)

                alpha = jax.lax.fori_loop(0, 32, backtrack, jnp.array(1.))
                return jnp.where(jnp.max(jnp.abs(g)) < 2e-12, q, q+alpha*direction)

            q = jax.lax.fori_loop(0, 80, step, start)
            h = jax.jacfwd(fun)(q)
            valid = (jnp.max(jnp.abs(fun(q))) < 1e-8) & (jnp.linalg.eigvalsh(h)[0] > 1e-9)
            valid &= jnp.all(sites(q) >= -1e-14)
            return q, jnp.where(valid, objective(q), jnp.inf)

        if starts is None:
            q, value = run(guess)
        else:
            roots, values = jax.vmap(run)(starts)
            index = jnp.argmin(values)
            q, value = roots[index], values[index]
        return jnp.where(jnp.isfinite(value), q, jnp.nan)

    def tangent_solve(linear, rhs):
        matrix = jax.vmap(linear)(jnp.eye(rhs.size)).T
        return jnp.linalg.solve(matrix, rhs)

    return jax.lax.custom_root(gradient, initial, solve, tangent_solve)


def pure_reference(data, *, ordered_lower=None):
    """Build scalar references, optionally comparing symmetric order branches.

    ordered_lower brackets the outer stable root beyond its last inflection.
    Comparing its energy with u=1/2 handles a first-order pure transition.
    """
    functions = tuple(expression(d)[0] for d in data)

    def evaluate(T, P):
        values = []
        for d, fun in zip(data, functions):
            if not d['sites']:
                values.append(fun(T, P, jnp.array([.5])))
                continue
            # The bracket only locates a root; its clipped inflection points
            # must not enter AD. The root's response is implicit below.
            lower = jax.lax.stop_gradient(ordered_lower(T, P)) if ordered_lower is not None else 1e-14
            upper = 1.-1e-14
            slope = jax.grad(lambda u: fun(T, P, jnp.array([u]))/(R*T))

            def interior(_):
                def solve(f, initial):
                    def step(_, bounds):
                        lo, hi = bounds
                        mid = (lo+hi)/2
                        return jnp.where(f(mid) > 0, lo, mid), jnp.where(f(mid) > 0, mid, hi)
                    lo, hi = jax.lax.fori_loop(0, 64, step, (lower, upper))
                    return (lo+hi)/2
                return jax.lax.custom_root(slope, (lower+upper)/2, solve,
                                           lambda f, rhs: rhs/f(jnp.ones_like(rhs)))

            u = jax.lax.cond(slope(lower) >= 0., lambda _: jnp.array(.5 if ordered_lower is not None else lower),
                             interior, None)
            value = fun(T, P, jnp.array([u]))
            if ordered_lower is not None:
                value = jnp.minimum(value, fun(T, P, jnp.array([.5])))
            values.append(value)
        return jnp.stack(values)

    return evaluate


def extensive(T, P, n, size, molar):
    T, P = _tp(T, P)
    n = jnp.asarray(n)*1.
    if n.shape != (size,):
        raise ValueError(f'n must have shape ({size},) in COMPONENTS order.')
    return jax.lax.cond(jnp.all(n == 0), lambda a: jnp.zeros_like(T+P+a.sum()),
                        lambda a: a.sum()*molar(T, P, a/a.sum()), n)
