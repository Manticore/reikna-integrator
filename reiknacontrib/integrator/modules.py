from reikna.helpers import product
import reikna.cluda.dtypes as dtypes


class Drift:
    """
    The base class for drift (deterministic part of a stepper) modules.

    The module must contain functions with the names and signatures

    ::

        INLINE WITHIN_KERNEL ${s_ctype} ${prefix}${component}(
                const int idx_x, ...
                const ${s_ctype} y0, ...
                ${r_ctype} t)

    where ``s_ctype`` is the C type corresponding to the state vector dtype,
    and ``component`` is the number of component the function handles
    (starting from 0).
    First set of parameters is the index of the processed element in the grid
    (total number is therefore the number of grid dimensions),
    the second set is the value of the state vector with this index for every component.

    :param module: a Reikna ``Module`` object.
    :param dtype: the data type of a supported state vector.
    :param components: the number of components the module has functions for.
    """

    def __init__(self, module, dtype, components=1):
        self.module = module
        self.components = components
        self.dtype = dtypes.normalize_type(dtype)

    def __process_modules__(self, process):
        return Drift(process(self.module), self.dtype, components=self.components)


class Diffusion:
    """
    The base class for diffusion (stochastic part of a stepper) modules.

    The module must contain functions with the names and signatures

    ::

            INLINE WITHIN_KERNEL ${s_ctype} ${prefix}${component}_${noise_source}(
                const int idx_x, ...
                const ${s_ctype} y0, ...
                ${r_ctype} t)

    where ``s_ctype`` is the C type corresponding to the state vector dtype,
    ``component`` and ``noise_source`` are the number of component and the number of noise source
    the function handles, respectively (starting from 0).
    The parameters are the same as for :py:class:`Drift`.

    :param module: a Reikna ``Module`` object.
    :param dtype: the data type of a supported state vector.
    :param components: the number of components the module has functions for.
    :param noise_sources: the number of noise sources the module has functions for.
    :param real_noise: whether the stochastic term uses a real-valued Wiener process.
    """

    def __init__(self, module, dtype, components=1, noise_sources=1, real_noise=False):
        self.module = module
        self.components = components
        self.noise_sources = noise_sources
        self.real_noise = real_noise
        self.dtype = dtypes.normalize_type(dtype)

    def __process_modules__(self, process):
        return Diffusion(
            process(self.module), self.dtype,
            components=self.components, noise_sources=self.noise_sources,
            real_noise=self.real_noise)
