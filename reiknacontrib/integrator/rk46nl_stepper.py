import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type

from reikna.fft import FFT
from reikna.algorithms import PureParallel

from beclab.integrator.helpers import get_ksquared, get_kprop_trf, get_project_trf


def get_xpropagate(state_arr, drift, diffusion=None, dW_arr=None):

    real_dtype = dtypes.real_for(state_arr.dtype)
    if diffusion is not None:
        noise_dtype = dW_arr.dtype
    else:
        noise_dtype = real_dtype

    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('omega', Annotation(state_arr, 'io')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('kinput', Annotation(state_arr, 'i'))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if diffusion is not None else []) +
            [Parameter('ai', Annotation(real_dtype)),
            Parameter('bi', Annotation(real_dtype)),
            Parameter('ci', Annotation(real_dtype)),
            Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype)),
            Parameter('stage', Annotation(numpy.int32))],
        """
        <%
            coords = ", ".join(idxs[1:])
            trajectory = idxs[0]
            components = drift.components
            if diffusion is not None:
                noise_sources = diffusion.noise_sources
            psi_args = ", ".join("psi_" + str(c) for c in range(components))

            if diffusion is None:
                dW = None
        %>

        %for comp in range(components):
        ${output.ctype} omega_${comp};
        if (${stage} == 0)
        {
            omega_${comp} = ${dtypes.c_constant(0, output.dtype)};
        }
        else
        {
            omega_${comp} = ${omega.load_idx}(${trajectory}, ${comp}, ${coords});
        }
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
                + ${drift.module}${comp}(${coords}, ${psi_args}, ${t} + ${dt} * ${ci}),
                ${dt})
            %if diffusion is not None:
            %for ncomp in range(noise_sources):
            + ${mul_cn}(${diffusion.module}${comp}_${ncomp}(
                ${coords}, ${psi_args}, ${t} + ${dt} * ${ci}), dW_${ncomp})
            %endfor
            %endif
            ;
        %endfor

        ${output.ctype} new_omega, new_u;
        %for comp in range(components):
        new_omega = ${mul_cr}(omega_${comp}, ${ai}) + dpsi_${comp};
        new_u = psi_${comp} + ${mul_cr}(new_omega, ${bi});
        if (${stage} < 5)
        {
            ${omega.store_idx}(${trajectory}, ${comp}, ${coords}, new_omega);
        }
        ${output.store_idx}(${trajectory}, ${comp}, ${coords}, new_u);
        %endfor
        """,
        guiding_array=(state_arr.shape[0],) + state_arr.shape[2:],
        render_kwds=dict(
            drift=drift,
            diffusion=diffusion,
            mul_cr=functions.mul(state_arr.dtype, real_dtype),
            mul_cn=functions.mul(state_arr.dtype, noise_dtype)))


class RK46NLStepper(Computation):
    """
    6-step 4th order RK optimized for minimum dissipation and minimum temporary space.
    """

    abbreviation = "RK46NL"

    def __init__(self, shape, box, drift, trajectories=1, kinetic_coeff=0.5j, diffusion=None,
            ksquared_cutoff=None):

        real_dtype = dtypes.real_for(drift.dtype)

        if diffusion is not None:
            assert diffusion.dtype == drift.dtype
            assert diffusion.components == drift.components
            self._noise = True
            dW_dtype = real_dtype if diffusion.real_noise else drift.dtype
            dW_arr = Type(dW_dtype, (trajectories, diffusion.noise_sources) + shape)
        else:
            dW_arr = None
            self._noise = False

        state_arr = Type(drift.dtype, (trajectories, drift.components) + shape)

        Computation.__init__(self,
            [Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i'))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if self._noise else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))])

        self._ksquared = get_ksquared(shape, box).astype(real_dtype)
        kprop_trf = get_kprop_trf(state_arr, self._ksquared, -kinetic_coeff)

        self._ksquared_cutoff = ksquared_cutoff
        if self._ksquared_cutoff is not None:
            project_trf = get_project_trf(state_arr, self._ksquared, ksquared_cutoff)
            self._fft_with_project = FFT(state_arr, axes=range(2, len(state_arr.shape)))
            self._fft_with_project.parameter.output.connect(
                project_trf, project_trf.input,
                output_prime=project_trf.output, ksquared=project_trf.ksquared)

        self._fft = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop.parameter.output.connect(
            kprop_trf, kprop_trf.input,
            output_prime=kprop_trf.output, ksquared=kprop_trf.ksquared, dt=kprop_trf.dt)

        self._xpropagate = get_xpropagate(state_arr, drift, diffusion=diffusion, dW_arr=dW_arr)

        self._ai = numpy.array([
            0.0, -0.737101392796, -1.634740794341,
            -0.744739003780, -1.469897351522, -2.813971388035])
        self._bi = numpy.array([
            0.032918605146, 0.823256998200, 0.381530948900,
            0.200092213184, 1.718581042715, 0.27])
        self._ci = numpy.array([
            0.0, 0.032918605146, 0.249351723343,
            0.466911705055, 0.582030414044, 0.847252983783])

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
        omega = plan.temp_array_like(output)
        result = plan.temp_array_like(output)

        data_out = input_

        for stage in range(6):

            data_in = data_out
            if stage == 5 and self._ksquared_cutoff is None:
                data_out = output
            else:
                data_out = result

            self._kpropagate(plan, kdata, kdata, data_in, ksquared_device, dt)

            if self._noise:
                plan.computation_call(
                    self._xpropagate, data_out, omega, data_in, kdata, dW,
                    self._ai[stage], self._bi[stage], self._ci[stage], t, dt, stage)
            else:
                plan.computation_call(
                    self._xpropagate, data_out, omega, data_in, kdata,
                    self._ai[stage], self._bi[stage], self._ci[stage], t, dt, stage)

            if self._ksquared_cutoff is not None:
                project_out = output if stage == 5 else data_out
                self._project(plan, project_out, data_out, data_out, ksquared_device)

        return plan
