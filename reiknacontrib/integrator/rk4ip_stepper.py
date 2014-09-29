import numpy

from reikna.cluda import dtypes, Module, functions
from reikna.core import Computation, Parameter, Annotation, Type

from reikna.fft import FFT
from reikna.algorithms import PureParallel

from beclab.integrator.helpers import get_ksquared, get_kprop_exp_trf


def get_nonlinear_wrapper(state_dtype, grid_dims, drift, diffusion=None):

    real_dtype = dtypes.real_for(state_dtype)
    if diffusion is not None:
        noise_dtype = diffusion.dtype
    else:
        noise_dtype = real_dtype

    return Module.create(
        """
        <%
            components = drift.components
            idx_args = ["idx_" + str(dim) for dim in range(grid_dims)]
            psi_args = ["psi_" + str(comp) for comp in range(components)]
            if diffusion is not None:
                dW_args = ["dW_" + str(ncomp) for ncomp in range(diffusion.noise_sources)]
        %>
        %for comp in range(components):
        INLINE WITHIN_KERNEL ${s_ctype} ${prefix}${comp}(
            %for idx in idx_args:
            const int ${idx},
            %endfor
            %for psi in psi_args:
            const ${s_ctype} ${psi},
            %endfor
            %if diffusion is not None:
            %for dW in dW_args:
            const ${n_ctype} ${dW},
            %endfor
            %endif
            const ${r_ctype} t,
            const ${r_ctype} dt)
        {
            return
                ${mul_sr}(${drift.module}${comp}(
                    ${", ".join(idx_args)}, ${", ".join(psi_args)}, t), dt)
                %if diffusion is not None:
                %for ncomp in range(diffusion.noise_sources):
                + ${mul_sn}(${diffusion.module}${comp}_${ncomp}(
                    ${", ".join(idx_args)}, ${", ".join(psi_args)}, t), ${dW_args[ncomp]})
                %endfor
                %endif
                ;
        }
        %endfor
        """,
        render_kwds=dict(
            grid_dims=grid_dims,
            s_ctype=dtypes.ctype(state_dtype),
            r_ctype=dtypes.ctype(real_dtype),
            n_ctype=dtypes.ctype(noise_dtype),
            mul_sr=functions.mul(state_dtype, real_dtype),
            mul_sn=functions.mul(state_dtype, noise_dtype),
            drift=drift,
            diffusion=diffusion))


def get_nonlinear1(state_arr, nonlinear_wrapper, components, diffusion=None, dW_arr=None):

    real_dtype = dtypes.real_for(state_arr.dtype)

    # output = N(input)
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i'))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if diffusion is not None else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))],
        """
        <%
            if diffusion is None:
                dW = None

            coords = ", ".join(idxs[1:])
            trajectory = idxs[0]

            args = lambda prefix, num: list(map(lambda i: prefix + str(i), range(num)))
            dW_args = args('dW_', diffusion.noise_sources) if diffusion is not None else []
            n_args = ", ".join(idxs[1:] + args('psi_', components) + dW_args)
        %>
        %for comp in range(components):
        ${output.ctype} psi_${comp} = ${input.load_idx}(${trajectory}, ${comp}, ${coords});
        %endfor

        %if diffusion is not None:
        %for ncomp in range(diffusion.noise_sources):
        ${dW.ctype} dW_${ncomp} = ${dW.load_idx}(${trajectory}, ${ncomp}, ${coords});
        %endfor
        %endif

        %for comp in range(components):
        ${output.store_idx}(
            ${trajectory}, ${comp}, ${coords}, ${nonlinear}${comp}(${n_args}, ${t}, ${dt}));
        %endfor
        """,
        guiding_array=(state_arr.shape[0],) + state_arr.shape[2:],
        render_kwds=dict(
            components=components,
            nonlinear=nonlinear_wrapper,
            diffusion=diffusion))


def get_nonlinear2(state_arr, nonlinear_wrapper, components, diffusion=None, dW_arr=None):

    real_dtype = dtypes.real_for(state_arr.dtype)

    # k2 = N(psi_I + k1 / 2, t + dt / 2)
    # k3 = N(psi_I + k2 / 2, t + dt / 2)
    # psi_4 = psi_I + k3 (argument for the 4-th step k-propagation)
    # psi_k = psi_I + (k1 + 2(k2 + k3)) / 6 (argument for the final k-propagation)
    return PureParallel(
        [
            Parameter('psi_k', Annotation(state_arr, 'o')),
            Parameter('psi_4', Annotation(state_arr, 'o')),
            Parameter('psi_I', Annotation(state_arr, 'i')),
            Parameter('k1', Annotation(state_arr, 'i'))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if diffusion is not None else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))],
        """
        <%
            if diffusion is None:
                dW = None

            coords = ", ".join(idxs[1:])
            trajectory = idxs[0]

            args = lambda prefix, num: ", ".join(map(lambda i: prefix + str(i), range(num)))
            dW_args = (args('dW_', diffusion.noise_sources) + ",") if diffusion is not None else ""
        %>

        %for comp in range(components):
        ${psi_k.ctype} psi_I_${comp} = ${psi_I.load_idx}(${trajectory}, ${comp}, ${coords});
        ${psi_k.ctype} k1_${comp} = ${k1.load_idx}(${trajectory}, ${comp}, ${coords});
        %endfor

        %if diffusion is not None:
        %for ncomp in range(diffusion.noise_sources):
        ${dW.ctype} dW_${ncomp} = ${dW.load_idx}(${trajectory}, ${ncomp}, ${coords});
        %endfor
        %endif

        %for comp in range(components):
        ${psi_k.ctype} k2_${comp} = ${nonlinear}${comp}(
            ${coords},
            %for c in range(components):
            psi_I_${c} + ${div}(k1_${c}, 2),
            %endfor
            ${dW_args}
            ${t} + ${dt} / 2, ${dt});
        %endfor

        %for comp in range(components):
        ${psi_k.ctype} k3_${comp} = ${nonlinear}${comp}(
            ${coords},
            %for c in range(components):
            psi_I_${c} + ${div}(k2_${c}, 2),
            %endfor
            ${dW_args}
            ${t} + ${dt} / 2, ${dt});
        %endfor

        %for comp in range(components):
        ${psi_4.store_idx}(${trajectory}, ${comp}, ${coords}, psi_I_${comp} + k3_${comp});
        %endfor

        %for comp in range(components):
        ${psi_k.store_idx}(
            ${trajectory}, ${comp}, ${coords},
            psi_I_${comp} + ${div}(k1_${comp}, 6) + ${div}(k2_${comp}, 3) + ${div}(k3_${comp}, 3));
        %endfor
        """,
        guiding_array=(state_arr.shape[0],) + state_arr.shape[2:],
        render_kwds=dict(
            components=components,
            nonlinear=nonlinear_wrapper,
            diffusion=diffusion,
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


def get_nonlinear3(state_arr, nonlinear_wrapper, components, diffusion=None, dW_arr=None):

    real_dtype = dtypes.real_for(state_arr.dtype)

    # k4 = N(D(psi_4), t + dt)
    # output = D(psi_k) + k4 / 6
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('kprop_psi_k', Annotation(state_arr, 'i')),
            Parameter('kprop_psi_4', Annotation(state_arr, 'i'))]
            + ([Parameter('dW', Annotation(dW_arr, 'i'))] if diffusion is not None else []) +
            [Parameter('t', Annotation(real_dtype)),
            Parameter('dt', Annotation(real_dtype))],
        """
        <%
            if diffusion is None:
                dW = None

            coords = ", ".join(idxs[1:])
            trajectory = idxs[0]

            args = lambda prefix, num: list(map(lambda i: prefix + str(i), range(num)))
            dW_args = args('dW_', diffusion.noise_sources) if diffusion is not None else []
            k4_args = ", ".join(idxs[1:] + args('psi4_', components) + dW_args)
        %>

        %for comp in range(components):
        ${output.ctype} psi4_${comp} = ${kprop_psi_4.load_idx}(${trajectory}, ${comp}, ${coords});
        ${output.ctype} psik_${comp} = ${kprop_psi_k.load_idx}(${trajectory}, ${comp}, ${coords});
        %endfor

        %if diffusion is not None:
        %for ncomp in range(diffusion.noise_sources):
        ${dW.ctype} dW_${ncomp} = ${dW.load_idx}(${trajectory}, ${ncomp}, ${coords});
        %endfor
        %endif

        %for comp in range(components):
        ${output.ctype} k4_${comp} = ${nonlinear}${comp}(${k4_args}, ${t} + ${dt}, ${dt});
        %endfor

        %for comp in range(components):
        ${output.store_idx}(
            ${trajectory}, ${comp}, ${coords},
            psik_${comp} + ${div}(k4_${comp}, 6));
        %endfor
        """,
        guiding_array=(state_arr.shape[0],) + state_arr.shape[2:],
        render_kwds=dict(
            components=components,
            nonlinear=nonlinear_wrapper,
            diffusion=diffusion,
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


class RK4IPStepper(Computation):
    """
    The integration method is RK4IP taken from the thesis by B. Caradoc-Davies
    "Vortex Dynamics in Bose-Einstein Condensates" (2000),
    namely Eqns. B.10 (p. 166).
    """

    abbreviation = "RK4IP"

    def __init__(self, shape, box, drift, trajectories=1, kinetic_coeff=0.5j, diffusion=None,
            ksquared_cutoff=None):

        if ksquared_cutoff is not None:
            raise NotImplementedError

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

        ksquared = get_ksquared(shape, box)
        # '/2' because we want to propagate only to dt/2
        self._kprop = (-ksquared / 2).astype(real_dtype)
        kprop_trf = get_kprop_exp_trf(state_arr, self._kprop, kinetic_coeff)

        self._fft = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop.parameter.output.connect(
            kprop_trf, kprop_trf.input,
            output_prime=kprop_trf.output, kprop=kprop_trf.kprop, dt=kprop_trf.dt)

        nonlinear_wrapper = get_nonlinear_wrapper(
            state_arr.dtype, len(state_arr.shape) - 2, drift, diffusion=diffusion)
        self._N1 = get_nonlinear1(
            state_arr, nonlinear_wrapper, drift.components, diffusion=diffusion, dW_arr=dW_arr)
        self._N2 = get_nonlinear2(
            state_arr, nonlinear_wrapper, drift.components, diffusion=diffusion, dW_arr=dW_arr)
        self._N3 = get_nonlinear3(
            state_arr, nonlinear_wrapper, drift.components, diffusion=diffusion, dW_arr=dW_arr)

    def _add_kprop(self, plan, output, input_, kprop_device, dt):
        temp = plan.temp_array_like(output)
        plan.computation_call(self._fft_with_kprop, temp, kprop_device, dt, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _build_plan(self, plan_factory, device_params, *args):

        if self._noise:
            output, input_, dW, t, dt = args
            t_args = (dW, t, dt)
        else:
            output, input_, t, dt = args
            t_args = (t, dt)

        plan = plan_factory()

        kprop_device = plan.persistent_array(self._kprop)

        # psi_I = D(psi)
        psi_I = plan.temp_array_like(output)
        self._add_kprop(plan, psi_I, input_, kprop_device, dt)

        # k1 = D(N(psi, t))
        k1 = plan.temp_array_like(output)
        temp = plan.temp_array_like(output)
        plan.computation_call(self._N1, temp, input_, *t_args)
        self._add_kprop(plan, k1, temp, kprop_device, dt)

        # k2 = N(psi_I + k1 / 2, t + dt / 2)
        # k3 = N(psi_I + k2 / 2, t + dt / 2)
        # psi_4 = psi_I + k3 (argument for the 4-th step k-propagation)
        # psi_k = psi_I + (k1 + 2(k2 + k3)) / 6 (argument for the final k-propagation)
        psi_4 = plan.temp_array_like(output)
        psi_k = plan.temp_array_like(output)
        plan.computation_call(self._N2, psi_k, psi_4, psi_I, k1, *t_args)

        # k4 = N(D(psi_4), t + dt)
        # output = D(psi_k) + k4 / 6
        kprop_psi_k = plan.temp_array_like(output)
        self._add_kprop(plan, kprop_psi_k, psi_k, kprop_device, dt)
        kprop_psi_4 = plan.temp_array_like(output)
        self._add_kprop(plan, kprop_psi_4, psi_4, kprop_device, dt)
        plan.computation_call(self._N3, output, kprop_psi_k, kprop_psi_4, *t_args)

        return plan
