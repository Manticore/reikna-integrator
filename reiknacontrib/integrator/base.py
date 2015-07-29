"""
Some base classes, mainly for centralized documenting purposes.
"""
from reikna.cluda import dtypes
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.helpers import product

from .wiener import Wiener


class Filter:
    """
    The base class of a filter object used by :py:class:`Integrator`.
    """

    def __call__(self, data, t):
        """
        A callback method that will be invoked by the integrator
        when the object is passed in the ``filters`` list to
        :py:meth:`Integrator.fixed_step` or :py:meth:`Integrator.adaptive_step`.

        The ``data`` array is a Reikna ``Array`` and is supposed to be modified inplace.
        """
        raise NotImplementedError


class Stepper:
    r"""
    The base class of a stepper object used by :py:class:`Integrator`.
    Used to calculated the differential of the state vector given
    time, time step and (optionally) Wiener process differentials:

    .. math::

        S(x, y, t, dt, dW) = \sum_{n=0,2,...} K_n \nabla^n y + D(x, y, t) dt + S(x, y, t) dW,

    where :math:`D` is the deterministic (drift) term,
    and :math:`S` is the stochastic (diffusion) term.

    This class assumes that the state vector is defined on a uniform rectangular grid,
    and its shape is ``(trajectories, components, *shape)``, where
    the number of components is specified by ``drift`` and ``diffusion`` objects.

    :param shape: grid shape.
    :param box: the physical size of the grid.
    :param drift: a :py:class:`Drift` object providing the function :math:`D`.
    :param trajectories: the number of stochastic trajectories.
    :param diffusion: a :py:class:`Diffusion` object providing the function :math:`S`.
    :param ksquared_cutoff: if a positive real value, will be used as a cutoff threshold
        for :math:`k^2` in the momentum space.
        The modes with higher momentum will be projected out on each step.
    """

    def __init__(self, shape, box, drift, trajectories=1, diffusion=None):

        if diffusion is not None:
            assert diffusion.dtype == drift.dtype
            assert diffusion.components == drift.components

            if not diffusion.real_noise or dtypes.is_real(drift.dtype):
                noise_dtype = drift.dtype
            else:
                noise_dtype = dtypes.real_for(drift.dtype)

            self.noise_type = Type(noise_dtype, (trajectories, diffusion.noise_sources) + shape)
            self.noise = True

            cell_volume = product(box) / product(shape)
            self._noise_normalization = 1. / cell_volume
        else:
            self.noise_type = None
            self.noise = False

    def get_stepper(self, thread):
        raise NotImplementedError

    def get_wiener(self, thread, seed=None):
        if self.noise_type is not None:
            return Wiener(thread, self.noise_type, self._noise_normalization, seed=seed)
        else:
            return None
