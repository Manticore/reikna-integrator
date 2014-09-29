r"""
This example applies different steppers to the problem of simulating
a bright soliton in a 1D BEC with attractive interaction.

For the GPE equation

.. math::

    i \hbar \frac{\partial \Psi(x, t)}{\partial t}
    = \left(
        - \frac{\hbar^2}{2m} \nabla^2
        + U |\Psi|^2
    \right) \Psi

there is an exact solution

.. math::

    \Psi(x, t)
    = \sqrt{n_0} \exp(-i \mu t / \hbar)
        \frac{1}{\cosh\left(
            \sqrt{2 m |\mu| / \hbar^2} x
        \right)},

where :math:`\mu = g n_0 / 2`.
The attractive interaction means that :math:`U < 0`.
In dimensionless variables
:math:`x = \hbar \tilde{x} / \sqrt{m g n_0}`,
:math:`t = \hbar \tilde{t} / \mu`,
the GPE has the form

.. math::

    i \frac{\partial \tilde{\Psi}(\tilde{x}, \tilde{t})}{\partial \tilde{t}}
    = \left(
        - \tilde{\nabla}^2
        - 2 |\tilde{\Psi}|^2 / \tilde{n}_0
    \right) \tilde{\Psi},

and the soliton solution is

.. math::

    \tilde{\Psi}(\tilde{x}, \tilde{t})
    = \sqrt{\tilde{n_0}} \exp(i \tilde{t}) \frac{1}{\cosh \tilde{x}},

We can introduce linear losses in this GPE as

.. math::

    i \frac{\partial \tilde{\Psi}(\tilde{x}, \tilde{t})}{\partial \tilde{t}}
    = \left(
        - \tilde{\nabla}^2
        - 2 |\tilde{\Psi}|^2 / \tilde{n}_0
    \right) \tilde{\Psi}
    - i \gamma \tilde{\Psi},

and in Wigner representation the equation will be

.. math::

    \frac{\partial \tilde{\Psi}(\tilde{x}, \tilde{t})}{\partial \tilde{t}}
    = -i \left(
        - \tilde{\nabla}^2
        - 2 (|\tilde{\Psi}|^2 - M / V) / \tilde{n}_0
    \right) \tilde{\Psi}
    - \gamma \tilde{\Psi}
    + \gamma dW(\tilde{x}, \tilde{t}),

where :math:`M` is the number of modes, :math:`V` is the box size,
and :math:`dW` is the standard Wiener process.

In this example we test that:

- without losses the numerical solution converges to the exact solution;
- the total atom number is preserved without losses;
- the total atom number decreases as :math:`N_0 \exp(-2 \gamma \tilde{t})`
  when the losses are enabled, both for the GPE and for the Wigner representation;
- visually check that the soliton stays in place and does not dissipate.
"""

from __future__ import print_function, division

import itertools

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
import reikna.cluda as cluda
from reikna.cluda import Module

from beclab.integrator import *


def get_drift(state_dtype, U, gamma, dx, wigner=False):
    return Drift(
        Module.create(
            """
            <%
                r_dtype = dtypes.real_for(s_dtype)
                s_ctype = dtypes.ctype(s_dtype)
                r_ctype = dtypes.ctype(r_dtype)
            %>
            INLINE WITHIN_KERNEL ${s_ctype} ${prefix}0(
                const int idx_x,
                const ${s_ctype} psi,
                ${r_ctype} t)
            {
                return ${mul_cc}(
                    COMPLEX_CTR(${s_ctype})(
                        -${gamma},
                        -(${U} * (${norm}(psi) - ${correction}))),
                    psi
                );
            }
            """,
            render_kwds=dict(
                s_dtype=state_dtype,
                U=U,
                gamma=gamma,
                mul_cc=functions.mul(state_dtype, state_dtype),
                norm=functions.norm(state_dtype),
                correction=1. / dx if wigner else 0
                )),
        state_dtype, components=1)


def get_diffusion(state_dtype, gamma):
    return Diffusion(
        Module.create(
            """
            <%
                r_dtype = dtypes.real_for(s_dtype)
                s_ctype = dtypes.ctype(s_dtype)
                r_ctype = dtypes.ctype(r_dtype)
            %>
            INLINE WITHIN_KERNEL ${s_ctype} ${prefix}0_0(
                const int idx_x,
                const ${s_ctype} psi,
                ${r_ctype} t)
            {
                return COMPLEX_CTR(${s_ctype})(${numpy.sqrt(gamma)}, 0);
            }
            """,
            render_kwds=dict(
                mul_cr=functions.mul(state_dtype, dtypes.real_for(state_dtype)),
                s_dtype=state_dtype,
                gamma=gamma)),
        state_dtype, components=1, noise_sources=1)


class PsiSampler(Sampler):

    def __init__(self):
        Sampler.__init__(self, no_mean=True, no_stderr=True)

    def __call__(self, psi, t):
        return psi.get()


class NSampler(Sampler):

    def __init__(self, dx, wigner=False, delta_modifier=None, stop_time=None):
        Sampler.__init__(self)
        self._wigner = wigner
        self._dx = dx
        self._stop_time = stop_time
        self._delta_modifier = delta_modifier

    def __call__(self, psi, t):
        psi = psi.get()

        # (components, trajectories, x_points)
        density = numpy.abs(psi) ** 2 - (self._delta_modifier / 2 if self._wigner else 0)

        N = density.sum(2) * self._dx

        sample = N[:,0]

        if self._stop_time is not None and t >= self._stop_time:
            raise StopIntegration(sample)
        else:
            return sample


class DensitySampler(Sampler):

    def __init__(self, dx, wigner=False):
        Sampler.__init__(self, no_stderr=True)
        self._wigner = wigner
        self._dx = dx

    def __call__(self, psi, t):
        psi = psi.get()

        # (components, trajectories, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._dx if self._wigner else 0)
        return density[:,0]


def run_test(thr, stepper_cls, integration, cutoff=False, no_losses=False, wigner=False):

    print()
    print(
        "*** Running " + stepper_cls.abbreviation +
        ", " + integration +
        ", cutoff=" + str(cutoff) +
        ", wigner=" + str(wigner) +
        ", no_losses=" + str(no_losses) + " test")
    print()

    # Simulation parameters
    lattice_size = 128 # spatial lattice points
    domain = (-7., 7.) # spatial domain
    paths = 128 # simulation paths
    interval = 0.5 # time interval
    samples = 100 # how many samples to take during simulation
    steps = samples * 100 # number of time steps (should be multiple of samples)
    gamma = 0.0 if no_losses else 0.2
    n0 = 100.0
    U = -2. / n0
    state_dtype = numpy.complex128
    seed = 1234

    # Lattice
    x = numpy.linspace(domain[0], domain[1], lattice_size, endpoint=False)
    dx = x[1] - x[0]

    # Initial state
    psi0 = numpy.sqrt(n0) / numpy.cosh(x)
    N0 = (numpy.abs(psi0) ** 2).sum() * dx

    shape, box = (lattice_size,), (domain[1] - domain[0],)
    if cutoff:
        ksquared_cutoff = get_padded_ksquared_cutoff(shape, box, pad=4)
        cutoff_mask = get_ksquared_cutoff_mask(shape, box, ksquared_cutoff=ksquared_cutoff)
        print("Using", cutoff_mask.sum(), "out of", lattice_size, "modes")

        # projecting and renormalizing the initial state
        kpsi0 = numpy.fft.fft(psi0)
        kpsi0 *= cutoff_mask
        psi0 = numpy.fft.ifft(kpsi0)
        N = (numpy.abs(psi0) ** 2).sum() * dx
        psi0 *= numpy.sqrt(N0 / N)
    else:
        cutoff_mask = get_ksquared_cutoff_mask(shape, box)
        ksquared_cutoff = None

    n_max = (numpy.abs(psi0) ** 2).max()
    print("N0 =", N0)

    # Initial noise
    if wigner:
        numpy.random.seed(seed)
        random_normals = (
            numpy.random.normal(size=(paths, 1, lattice_size)) +
            1j * numpy.random.normal(size=(paths, 1, lattice_size))) / 2
        fft_scale = numpy.sqrt(dx / lattice_size)
        psi0 = numpy.fft.ifft(random_normals * cutoff_mask) / fft_scale + psi0
    else:
        psi0 = numpy.tile(psi0, (1, 1, 1))

    psi_gpu = thr.to_device(psi0.astype(state_dtype))

    # Prepare integrator components
    drift = get_drift(state_dtype, U, gamma, dx, wigner=wigner)
    stepper = stepper_cls((lattice_size,), (domain[1] - domain[0],), drift,
        kinetic_coeff=1j, ksquared_cutoff=ksquared_cutoff,
        trajectories=paths if wigner else 1,
        diffusion=get_diffusion(state_dtype, gamma) if wigner else None)

    if wigner:
        wiener = Wiener(stepper.parameter.dW, 1. / dx, seed=seed)
    integrator = Integrator(
        thr, stepper,
        wiener=wiener if wigner else None,
        verbose=True, profile=True)

    # Integrate
    psi_sampler = PsiSampler()
    n_sampler = NSampler(
        dx, wigner=wigner,
        delta_modifier=cutoff_mask.sum() / (domain[1] - domain[0]),
        stop_time=interval if integration == 'adaptive-endless' else None)
    density_sampler = DensitySampler(dx, wigner=wigner)

    samplers = dict(psi=psi_sampler, N=n_sampler, density=density_sampler)

    if integration == 'fixed':
        result, info = integrator.fixed_step(
            psi_gpu, 0, interval, steps, samples=samples,
            samplers=samplers, weak_convergence=['N'], strong_convergence=['psi'])
    elif integration == 'adaptive':
        result, info = integrator.adaptive_step(
            psi_gpu, 0, interval / samples, t_end=interval,
            samplers=samplers, weak_convergence=dict(N=1e-6))
    elif integration == 'adaptive-endless':
        result, info = integrator.adaptive_step(
            psi_gpu, 0, interval / samples,
            samplers=samplers, weak_convergence=dict(N=1e-6))

    N_t, N_mean, N_err = result['N']['time'], result['N']['mean'], result['N']['stderr']
    density = result['density']['mean']
    N_exact = N0 * numpy.exp(-gamma * N_t * 2)

    suffix = (
        ('_cutoff' if cutoff else '') +
        ('_wigner' if wigner else '') +
        ('_no-losses' if no_losses else '') +
        '_' + stepper_cls.abbreviation +
        '_' + integration)

    # Plot density
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.imshow(density.T, interpolation='nearest', origin='lower', aspect='auto',
        extent=(0, interval) + domain, vmin=0, vmax=n_max)
    s.set_xlabel('$t$')
    s.set_ylabel('$x$')
    fig.savefig('soliton_density' + suffix + '.pdf')
    plt.close(fig)

    # Plot population
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(N_t, N_mean, 'b-')
    if wigner:
        s.plot(N_t, N_mean + N_err, 'b--')
        s.plot(N_t, N_mean - N_err, 'b--')
    s.plot(N_t, N_exact, 'k--')
    s.set_ylim(-N0 * 0.1, N0 * 1.1)
    s.set_xlabel('$t$')
    s.set_ylabel('$N$')
    fig.savefig('soliton_N' + suffix + '.pdf')
    plt.close(fig)

    # Compare with the analytical solution
    if not wigner and no_losses:
        psi = result['psi']['values']
        tt = numpy.linspace(0, interval, samples + 1, endpoint=True).reshape(samples + 1, 1, 1, 1)
        xx = x.reshape(1, 1, 1, x.size)
        psi_exact = numpy.sqrt(n0) / numpy.cosh(xx) * numpy.exp(1j * tt)
        diff = numpy.linalg.norm(psi - psi_exact) / numpy.linalg.norm(psi_exact)
        print("Difference with the exact solution:", diff)
        if not cutoff:
            assert diff < 1e-3

    # Check the population decay
    sigma = numpy.linalg.norm(N_err) if wigner else N_mean * 1e-6
    max_diff = (numpy.abs(N_mean - N_exact) / sigma).max()
    print("Maximum difference with the exact population decay:", max_diff, "sigma")
    assert max_diff < 1


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    cutoffs = [
        False,
        True,
    ]

    steppers = [
        CDIPStepper,
        CDStepper,
        RK4IPStepper,
        RK46NLStepper,
    ]

    integrations = [
        'fixed',
        'adaptive',
        'adaptive-endless',
    ]

    for stepper_cls, integration, cutoff in itertools.product(steppers, integrations, cutoffs):

        # FIXME: Currently not all steppers support cutoffs.
        if cutoff and stepper_cls not in (CDStepper, RK46NLStepper):
            continue

        run_test(thr, stepper_cls, integration, cutoff=cutoff, no_losses=True, wigner=False)
        run_test(thr, stepper_cls, integration, cutoff=cutoff, wigner=False)
        if integration == 'fixed':
            run_test(thr, stepper_cls, integration, cutoff=cutoff, wigner=True)
