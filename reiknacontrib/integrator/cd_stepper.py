from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type

from reikna.fft import FFT
from reikna.algorithms import PureParallel

from .helpers import get_ksquared, get_kprop_trf, get_project_trf
from .base import Stepper


def get_prop_iter(state_type, drift, diffusion=None, noise_type=None):

    real_dtype = dtypes.real_for(state_type.dtype)
    if diffusion is not None:
        noise_dtype = noise_type.dtype
    else:
        noise_dtype = real_dtype

    return PureParallel(
        [
            Parameter('output', Annotation(state_type, 'o')),
            Parameter('orig_input', Annotation(state_type, 'i')),
            Parameter('input', Annotation(state_type, 'i')),
            Parameter('kinput', Annotation(state_type, 'i'))]
            + ([Parameter('dW', Annotation(noise_type, 'i'))] if diffusion is not None else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype)),
            Parameter('dt_modifier', Annotation(real_dtype))],
        """
        <%
            components = drift.components
            if diffusion is not None:
                noise_sources = diffusion.noise_sources
            coords = ", ".join(idxs[1:])
            trajectory = idxs[0]
            psi_args = ", ".join("psi_" + str(c) for c in range(components))

            if diffusion is None:
                dW = None
        %>

        %for comp in range(components):
        ${output.ctype} psi_orig_${comp} = ${orig_input.load_idx}(
            ${trajectory}, ${comp}, ${coords});
        ${output.ctype} psi_${comp} = ${input.load_idx}(${trajectory}, ${comp}, ${coords});
        ${output.ctype} kpsi_${comp} = ${kinput.load_idx}(${trajectory}, ${comp}, ${coords});
        ${output.ctype} dpsi_${comp};
        %endfor

        %if diffusion is not None:
        %for ncomp in range(noise_sources):
        ${dW.ctype} dW_${ncomp} = ${dW.load_idx}(${trajectory}, ${ncomp}, ${coords});
        %endfor
        %endif

        %for comp in range(components):
        dpsi_${comp} =
            kpsi_${comp}
            + ${mul_cr}(
                + ${mul_cr}(${drift.module}${comp}(
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
        ${output.store_idx}(${trajectory}, ${comp}, ${coords},
            psi_orig_${comp} + ${mul_cr}(dpsi_${comp}, ${dt_modifier}));
        %endfor
        """,
        guiding_array=(state_type.shape[0],) + state_type.shape[2:],
        render_kwds=dict(
            drift=drift,
            diffusion=diffusion,
            mul_cr=functions.mul(state_type.dtype, real_dtype),
            mul_cn=functions.mul(state_type.dtype, noise_dtype)))


class _CDStepperComp(Computation):

    def __init__(self, shape, box, drift,
            trajectories=1, kinetic_coeff=0.5j, diffusion=None, noise_type=None,
            iterations=3, ksquared_cutoff=None):

        real_dtype = dtypes.real_for(drift.dtype)
        state_type = Type(drift.dtype, (trajectories, drift.components) + shape)

        self._iterations = iterations
        self._noise = diffusion is not None

        Computation.__init__(self,
            [Parameter('output', Annotation(state_type, 'o')),
            Parameter('input', Annotation(state_type, 'i'))]
            + ([Parameter('dW', Annotation(noise_type, 'i'))] if self._noise else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))])

        # '/2' because we want to propagate only to dt/2
        self._ksquared = get_ksquared(shape, box).astype(real_dtype)
        kprop_trf = get_kprop_trf(state_type, self._ksquared, -kinetic_coeff / 2)

        self._ksquared_cutoff = ksquared_cutoff
        if self._ksquared_cutoff is not None:
            project_trf = get_project_trf(state_type, self._ksquared, ksquared_cutoff)
            self._fft_with_project = FFT(state_type, axes=range(2, len(state_type.shape)))
            self._fft_with_project.parameter.output.connect(
                project_trf, project_trf.input,
                output_prime=project_trf.output, ksquared=project_trf.ksquared)

        self._fft = FFT(state_type, axes=range(2, len(state_type.shape)))
        self._fft_with_kprop = FFT(state_type, axes=range(2, len(state_type.shape)))
        self._fft_with_kprop.parameter.output.connect(
            kprop_trf, kprop_trf.input,
            output_prime=kprop_trf.output, ksquared=kprop_trf.ksquared, dt=kprop_trf.dt)

        self._prop_iter = get_prop_iter(
            state_type, drift, diffusion=diffusion, noise_type=noise_type)

    def _project(self, plan, output, temp, input_, ksquared_device):
        plan.computation_call(self._fft_with_project, temp, ksquared_device, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _kpropagate(self, plan, output, temp, input_, ksquared_device, dt):
        plan.computation_call(self._fft_with_kprop, temp, ksquared_device, dt, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _build_plan(self, plan_factory, device_params, *args):

        if self._noise:
            output, input_, dW, t, dt = args
        else:
            output, input_, t, dt = args

        plan = plan_factory()

        ksquared_device = plan.persistent_array(self._ksquared)
        kdata = plan.temp_array_like(output)
        result = plan.temp_array_like(output)

        data_out = input_

        for i in range(self._iterations):

            data_in = data_out
            if i == self._iterations - 1 and self._ksquared_cutoff is None:
                data_out = output
            else:
                data_out = result

            if i == self._iterations - 1:
                dt_modifier = 2.
            else:
                dt_modifier = 1.

            self._kpropagate(plan, kdata, kdata, data_in, ksquared_device, dt)

            if self._noise:
                plan.computation_call(
                    self._prop_iter, data_out, input_, data_in, kdata, dW, t, dt, dt_modifier)
            else:
                plan.computation_call(
                    self._prop_iter, data_out, input_, data_in, kdata, t, dt, dt_modifier)

            if self._ksquared_cutoff is not None:
                project_out = output if i == self._iterations - 1 else data_out
                self._project(plan, project_out, data_out, data_out, ksquared_device)

        return plan


class CDStepper(Stepper):
    """
    Split step, central difference stepper.

    :param shape: grid shape.
    :param box: the physical size of the grid.
    :param drift: a :py:class:`Drift` object providing the function :math:`D`.
    :param trajectories: the number of stochastic trajectories.
    :param kinetic_coeff: the value of :math:`K` above (can be real or complex).
    :param diffusion: a :py:class:`Diffusion` object providing the function :math:`S`.
    :param ksquared_cutoff: if a positive real value, will be used as a cutoff threshold
        for :math:`k^2` in the momentum space.
        The modes with higher momentum will be projected out on each step.
    """

    abbreviation = "CD"

    def __init__(self, shape, box, drift,
            trajectories=1, kinetic_coeff=0.5j, diffusion=None, iterations=3, ksquared_cutoff=None):

        Stepper.__init__(self, shape, box, drift, trajectories=trajectories, diffusion=diffusion)

        self._stepper_comp = _CDStepperComp(
            shape, box, drift,
            trajectories=trajectories,
            kinetic_coeff=kinetic_coeff, diffusion=diffusion, noise_type=self.noise_type,
            iterations=iterations, ksquared_cutoff=ksquared_cutoff)

    def get_stepper(self, thread):
        return self._stepper_comp.compile(thread)
