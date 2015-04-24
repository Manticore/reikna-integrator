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


class KineticCoeffs:

    def __init__(self, values, dtype):
        self.values = values
        self.dtype = dtype

    # backward compatibility with the old division
    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        return KineticCoeffs({pwr: value / other for pwr, value in self.values.items()}, self.dtype)


def normalize_kinetic_coeffs(kinetic_coeffs, drift_components):

    if kinetic_coeffs is None:
        values = {}
        dtype = None

    elif isinstance(kinetic_coeffs, dict):

        # get the dtype that will suite all the coefficients
        coeffs = [value for pwr, value in kinetic_coeffs.items()]
        dtype = numpy.asarray(coeffs).dtype

        values = {
            pwr:numpy.asarray(coeffs).reshape(drift_components).astype(dtype)
            for pwr, coeffs in kinetic_coeffs.items()}
    else:
        arr = numpy.asarray(kinetic_coeffs)
        dtype = arr.dtype

        if arr.ndim == 0:
        # 0: same coefficient for all components, applied to \nabla^2
            values = {2: numpy.tile(arr, drift_components)}
        elif arr.ndim == 1:
        # 1: separate coefficient for each component, applied to \nabla^2
            values = {2: arr}
        elif kinetic_coeffs_arr.ndim == 2:
        # separate coefficient for each component and for each power of \nabla
            values = {
                pwr: arr[:,pwr]
                for pwr in range(arr.shape[1])
                if (arr[:,pwr] != 0).any()}
        else:
            raise ValueError("Unsupported number of dimensions of the kinetic coefficients array")

    if any(pwr % 2 == 1 for pwr in values):
        raise NotImplementedError("Odd powers of \\nabla are not supported")

    return KineticCoeffs(values, dtype)


def get_kprop_trf(state_arr, ksquared_arr, coeffs, exp=False):
    compound_dtype = dtypes.result_type(coeffs.dtype, ksquared_arr.dtype)
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('ksquared', Annotation(ksquared_arr, 'i')),
            Parameter('dt', Annotation(ksquared_arr.dtype))],
        """
        %if max(coeffs.values) > 0:
        ${ksquared.ctype} ksquared = ${ksquared.load_idx}(${', '.join(idxs[2:])});
        %endif

        ${dtypes.ctype(compound_dtype)} compound_coeff = ${dtypes.c_constant(0, compound_dtype)};

        %for pwr, values in coeffs.values.items():
        {
            ${dtypes.ctype(coeffs.dtype)} value;

            %for comp in range(output.shape[1]):
            ${'if' if comp == 0 else 'elseif'} (${idxs[1]} == ${comp})
            {
                value = ${dtypes.c_constant(values[comp], coeffs.dtype)};
            }
            %endfor

            compound_coeff =
                compound_coeff
                + ${mul_kc}(
                    %if pwr == 0:
                    ${dt}
                    %elif pwr == 2:
                    -ksquared * ${dt}
                    %else:
                    pow(-ksquared, ${pwr // 2}) * ${dt}
                    %endif
                    ,
                    value
                    );
        }
        %endfor

        ${output.store_same}(${mul_ic}(
            ${input.load_same},
            %if exp is not None:
            ${exp}(compound_coeff)
            %else:
            compound_coeff
            %endif
            ));
        """,
        render_kwds=dict(
            coeffs=coeffs,
            compound_dtype=compound_dtype,
            mul_ic=functions.mul(state_arr.dtype, compound_dtype, out_dtype=state_arr.dtype),
            mul_kc=functions.mul(ksquared_arr.dtype, coeffs.dtype, out_dtype=compound_dtype),
            exp=functions.exp(compound_dtype) if exp else None))


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
