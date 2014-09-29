import numpy

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.core import Parameter, Annotation, Transformation


def get_ksquared(shape, box):
    ks = [
        2 * numpy.pi * numpy.fft.fftfreq(size, length / size)
        for size, length in zip(shape, box)]

    if len(shape) > 1:
        full_ks = numpy.meshgrid(*ks, indexing='ij')
    else:
        full_ks = ks

    return sum([full_k ** 2 for full_k in full_ks])


def get_ksquared_cutoff_mask(shape, box, ksquared_cutoff=None):
    if ksquared_cutoff is None:
        return numpy.ones(shape, numpy.int32)
    else:
        ksquared = get_ksquared(shape, box)
        return (ksquared <= ksquared_cutoff).astype(numpy.int32)


def get_padded_ksquared_cutoff(shape, box, pad=1):
    # pad=1 means tangential mode space.
    # pad=4 recommended to avoid aliasing.

    k_limits = []

    for size, length in zip(shape, box):
        ks = numpy.abs(2 * numpy.pi * numpy.fft.fftfreq(size, length / size))

        # ks has a 'triangle' shape: it starts from 0,
        # grows until the middle of the array, and then decreases back to near 0.
        limit_idx = min(int(ks.size / 2 / float(pad)), ks.size / 2 - 1)
        k_limits.append(ks[limit_idx])

    return min(k_limits) ** 2


def get_kprop_trf(state_arr, ksquared_arr, coeff):
    coeff_dtype = dtypes.detect_type(coeff)
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('ksquared', Annotation(ksquared_arr, 'i')),
            Parameter('dt', Annotation(ksquared_arr.dtype))],
        """
        ${ksquared.ctype} ksquared = ${ksquared.load_idx}(${', '.join(idxs[2:])});
        ${output.store_same}(${mul}(${input.load_same}, ${coeff}, ksquared * ${dt}));
        """,
        render_kwds=dict(
            coeff=dtypes.c_constant(coeff, coeff_dtype),
            mul=functions.mul(
                state_arr.dtype, coeff_dtype, ksquared_arr.dtype, out_dtype=state_arr.dtype)))


def get_project_trf(state_arr, ksquared_arr, ksquared_cutoff):
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('ksquared', Annotation(ksquared_arr, 'i'))],
        """
        ${ksquared.ctype} ksquared = ${ksquared.load_idx}(${', '.join(idxs[2:])});
        ${input.ctype} data;

        if (ksquared > ${dtypes.c_constant(ksquared_cutoff, ksquared.dtype)})
        {
            data = ${dtypes.c_constant(0, input.dtype)};
        }
        else
        {
            data = ${input.load_same};
        }

        ${output.store_same}(data);
        """,
        render_kwds=dict(
            ksquared_cutoff=ksquared_cutoff))


def get_kprop_exp_trf(state_arr, kprop_arr, kinetic_coeff):
    kcoeff_dtype = dtypes.detect_type(kinetic_coeff)
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('kprop', Annotation(kprop_arr, 'i')),
            Parameter('dt', Annotation(kprop_arr.dtype))],
        """
        ${kprop.ctype} kprop = ${kprop.load_idx}(${', '.join(idxs[2:])});
        ${output.ctype} kprop_exp = ${exp}(${mul_k}(kprop * ${dt}, ${kinetic_coeff}));
        ${output.store_same}(${mul}(${input.load_same}, kprop_exp));
        """,
        render_kwds=dict(
            kinetic_coeff=dtypes.c_constant(kinetic_coeff, kcoeff_dtype),
            mul_k=functions.mul(kprop_arr.dtype, kcoeff_dtype, out_dtype=state_arr.dtype),
            exp=functions.exp(state_arr.dtype),
            mul=functions.mul(state_arr.dtype, state_arr.dtype)))
