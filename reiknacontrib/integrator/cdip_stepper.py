import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type

from reikna.fft import FFT
from reikna.algorithms import PureParallel

from .helpers import get_ksquared, get_kprop_trf, normalize_kinetic_coeffs
from .base import Stepper


def get_prop_iter(state_type, drift, iterations, diffusion=None, noise_type=None):

    if dtypes.is_complex(state_type.dtype):
        real_dtype = dtypes.real_for(state_type.dtype)
    else:
        real_dtype = state_type.dtype

    if diffusion is not None:
        noise_dtype = noise_type.dtype
    else:
        noise_dtype = real_dtype

    return PureParallel(
        [
            Parameter('output', Annotation(state_type, 'o')),
            Parameter('input', Annotation(state_type, 'i'))]
            + ([Parameter('dW', Annotation(noise_type, 'i'))] if diffusion is not None else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))],
        """
        <%
            coords = ", ".join(idxs[1:])
            trajectory = idxs[0]
            components = drift.components
            if diffusion is not None:
                noise_sources = diffusion.noise_sources
            psi_args = ", ".join("psi_" + str(c) + "_tmp" for c in range(components))

            if diffusion is None:
                dW = None
        %>

        %for comp in range(components):
        ${output.ctype} psi_${comp} = ${input.load_idx}(${trajectory}, ${comp}, ${coords});
        ${output.ctype} psi_${comp}_tmp = psi_${comp};
        ${output.ctype} dpsi_${comp};
        %endfor

        %if diffusion is not None:
        %for ncomp in range(noise_sources):
        ${dW.ctype} dW_${ncomp} = ${dW.load_idx}(${trajectory}, ${ncomp}, ${coords});
        %endfor
        %endif

        %for i in range(iterations):

        %for comp in range(components):
        dpsi_${comp} =
            ${mul_cr}(
                ${mul_cr}(${drift.module}${comp}(
                    ${coords}, ${psi_args}, ${t} + ${dt} / 2), ${dt})
                %if diffusion is not None:
                %for ncomp in range(noise_sources):
                + ${mul_cn}(${diffusion.module}${comp}_${ncomp}(
                    ${coords}, ${psi_args}, ${t} + ${dt} / 2), dW_${ncomp})
                %endfor
                %endif
                , 0.5);
        %endfor

        %for comp in range(components):
        psi_${comp}_tmp = psi_${comp} + dpsi_${comp};
        %endfor

        %endfor

        %for comp in range(components):
        ${output.store_idx}(${trajectory}, ${comp}, ${coords}, psi_${comp}_tmp + dpsi_${comp});
        %endfor
        """,
        guiding_array=(state_type.shape[0],) + state_type.shape[2:],
        render_kwds=dict(
            drift=drift,
            diffusion=diffusion,
            iterations=iterations,
            mul_cr=functions.mul(state_type.dtype, real_dtype),
            mul_cn=functions.mul(state_type.dtype, noise_dtype)))


class _CDIPStepperComp(Computation):
    """
    Central difference, interaction picture (split-step) stepper.
    """

    abbreviation = "CDIP"

    def __init__(self, shape, box, drift,
            trajectories=1, kinetic_coeffs=0.5j, diffusion=None, iterations=3, noise_type=None):

        real_dtype = dtypes.real_for(drift.dtype)
        state_type = Type(drift.dtype, (trajectories, drift.components) + shape)

        self._noise = diffusion is not None

        Computation.__init__(self,
            [Parameter('output', Annotation(state_type, 'o')),
            Parameter('input', Annotation(state_type, 'i'))]
            + ([Parameter('dW', Annotation(noise_type, 'i'))] if self._noise else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))])

        self._ksquared = get_ksquared(shape, box).astype(real_dtype)
        # '/2' because we want to propagate only to dt/2
        kprop_trf = get_kprop_trf(state_type, self._ksquared, kinetic_coeffs / 2, exp=True)

        self._fft = FFT(state_type, axes=range(2, len(state_type.shape)))
        self._fft_with_kprop = FFT(state_type, axes=range(2, len(state_type.shape)))
        self._fft_with_kprop.parameter.output.connect(
            kprop_trf, kprop_trf.input,
            output_prime=kprop_trf.output, ksquared=kprop_trf.ksquared, dt=kprop_trf.dt)

        self._prop_iter = get_prop_iter(
            state_type, drift, iterations,
            diffusion=diffusion, noise_type=noise_type)

    def _add_kprop(self, plan, output, input_, ksquared_device, dt):
        temp = plan.temp_array_like(output)
        plan.computation_call(self._fft_with_kprop, temp, ksquared_device, dt, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _build_plan(self, plan_factory, device_params, *args):

        if self._noise:
            output, input_, dW, t, dt = args
        else:
            output, input_, t, dt = args

        plan = plan_factory()

        ksquared_device = plan.persistent_array(self._ksquared)

        # psi_I = prop_L_half_dt(input_)
        psi_I = plan.temp_array_like(input_)
        self._add_kprop(plan, psi_I, input_, ksquared_device, dt)

        # psi_N = prop_iter(psi_I)
        psi_N = plan.temp_array_like(input_)
        if self._noise:
            plan.computation_call(self._prop_iter, psi_N, psi_I, dW, t, dt)
        else:
            plan.computation_call(self._prop_iter, psi_N, psi_I, t, dt)

        # output = prop_L_half_dt(psi_N)
        self._add_kprop(plan, output, psi_N, ksquared_device, dt)

        return plan


class _CDParallelStepperComp(Computation):
    """
    Central difference stepper with no interaction between elements in transverse direction.
    """

    abbreviation = "CDParallel"

    def __init__(self, shape, drift, trajectories=1, diffusion=None, iterations=3, noise_type=None):

        if dtypes.is_complex(drift.dtype):
            real_dtype = dtypes.real_for(drift.dtype)
        else:
            real_dtype = drift.dtype

        state_type = Type(drift.dtype, (trajectories, drift.components) + shape)

        self._noise = diffusion is not None

        Computation.__init__(self,
            [Parameter('output', Annotation(state_type, 'o')),
            Parameter('input', Annotation(state_type, 'i'))]
            + ([Parameter('dW', Annotation(noise_type, 'i'))] if self._noise else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))])

        self._prop_iter = get_prop_iter(
            state_type, drift, iterations,
            diffusion=diffusion, noise_type=noise_type)

    def _build_plan(self, plan_factory, device_params, *args):

        if self._noise:
            output, input_, dW, t, dt = args
        else:
            output, input_, t, dt = args

        plan = plan_factory()

        if self._noise:
            plan.computation_call(self._prop_iter, output, input_, dW, t, dt)
        else:
            plan.computation_call(self._prop_iter, output, input_, t, dt)

        return plan


class CDIPStepper(Stepper):
    """
    Central difference, interaction picture (split-step) stepper.

    :param shape: grid shape.
    :param box: the physical size of the grid.
    :param drift: a :py:class:`Drift` object providing the function :math:`D`.
    :param trajectories: the number of stochastic trajectories.
    :param kinetic_coeffs: the value of :math:`K` above (can be real or complex).
        If it is a scalar, the same value will be used for all components
        and the second power of Laplacian;
        if it is a 1D vector, its elements will be used with the corresponding components
        and the second power of Laplacian;
        if a dictionary ``{power: values}``, ``values`` will be used for corresponding
        powers of the Laplacian (only even powers are supported).
    :param diffusion: a :py:class:`Diffusion` object providing the function :math:`S`.
    :param ksquared_cutoff: if a positive real value, will be used as a cutoff threshold
        for :math:`k^2` in the momentum space.
        The modes with higher momentum will be projected out on each step.
    """

    abbreviation = "CDIP"

    def __init__(self, shape, box, drift,
            trajectories=1, kinetic_coeffs=0.5j,
            diffusion=None, iterations=3, ksquared_cutoff=None):

        if ksquared_cutoff is not None:
            raise NotImplementedError

        Stepper.__init__(self, shape, box, drift, trajectories=trajectories, diffusion=diffusion)

        kinetic_coeffs = normalize_kinetic_coeffs(kinetic_coeffs, drift.components)

        if kinetic_coeffs.nonzero():
            self._stepper_comp = _CDIPStepperComp(
                shape, box, drift,
                trajectories=trajectories,
                kinetic_coeffs=kinetic_coeffs, diffusion=diffusion, noise_type=self.noise_type,
                iterations=iterations)
        else:
            self._stepper_comp = _CDParallelStepperComp(
                shape, drift,
                trajectories=trajectories,
                diffusion=diffusion, noise_type=self.noise_type,
                iterations=iterations)

    def get_stepper(self, thread):
        return self._stepper_comp.compile(thread)
