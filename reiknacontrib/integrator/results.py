import sys
import numpy

from reikna.linalg import EntrywiseNorm
from reikna.algorithms import Reduce, predicate_sum


_range = xrange if sys.version_info[0] < 3 else range


class StopIntegration(Exception):
    """
    If raised by a :py:class:`Sampler` object, stops the :py:class:`Integrator`.
    """
    pass


class IntegrationError(Exception):
    """
    Raised by :py:class:`Integrator` in case of various integration-time errors.
    """
    pass


class Timings:
    """
    Contains timings for various parts of the integration (in seconds).
    Has an overloaded ``+`` operator.

    .. py:attribute:: samplers

        Time taken by samplers.

    .. py:attribute:: normal

        Time taken by normal step integration.

    .. py:attribute:: double

        Time taken by double step integration.

    .. py:attribute:: integration

        Total time taken by normal and double step integration.
    """

    def __init__(self, normal=0, double=0, samplers=0):
        self.integration = normal + double
        self.samplers = samplers
        self.normal = normal
        self.double = double

    def __add__(self, other):
        return Timings(
            normal=self.normal + other.normal,
            double=self.double + other.double,
            samplers=self.samplers + other.samplers)


class IntegrationInfo:
    """
    Contains auxiliary information about integration.

    .. py:attribute:: weak_errors

        A dictionary matching sampler names to weak convergence estimates.

    .. py:attribute:: strong_errors

        A dictionary matching sampler names to strong convergence estimates.

    .. py:attribute:: timings

        A :py:class:`Timings` object.

    .. py:attribute:: steps

        The number of steps used for the integration.
    """

    def __init__(self, timings, strong_errors, weak_errors, steps):
        self.weak_errors = weak_errors
        self.strong_errors = strong_errors
        self.timings = timings
        self.steps = steps


class Sampler:
    """
    A base class for a sampler.

    :param no_mean: if ``True``, mean values will not be collected.
    :param no_stderr: if ``True``, estimates of sampling errors will not be collected.
    :param no_values: if ``True``, per-trajectory values will not be collected.
    """

    def __init__(self, no_mean=False, no_stderr=False, no_values=False):
        self.no_mean = no_mean
        self.no_stderr = no_stderr
        self.no_values = no_values

    def __call__(self, data, t):
        """
        A callback method that will be invoked by the integrator
        at every sampling point and passed the current data array
        (**warning:** must not be modified; copy if necessary)
        and the current integration time ``t``.

        Must return a scalar or a ``numpy`` array
        (with the same dtype and dimensions for every call).
        The first dimension must be equal to the number of trajectories.
        Depending on the attributes set on creation,
        the integrator will collect ``.mean(0)``, ``.std(0) / sqrt(trajectories)``
        and the array itself as ``mean``, ``stderr`` and ``values`` correspondingly
        in the data structure returned from :py:meth:`Integrator.fixed_step` and
        :py:meth:`Integrator.adaptive_step`.
        """
        raise NotImplementedError


def sample(data, t, samplers):
    sample_dict = {}
    stop_integration = False

    for key, sampler in samplers.items():

        try:
            sample = sampler(data, t)
        except StopIntegration as e:
            sample = e.args[0]
            stop_integration = True

        sample_dict[key] = dict(trajectories=sample.shape[0], time=t)

        if isinstance(sample, numpy.ndarray):
            if not sampler.no_values:
                sample_dict[key]['values'] = sample.copy()
            if not sampler.no_mean:
                sample_dict[key]['mean'] = sample.mean(0)
            if not sampler.no_stderr:
                sample_dict[key]['stderr'] = sample.std(0) / numpy.sqrt(sample.shape[0])
        else:
            thr = sample.thread

            if not sampler.no_values:
                sample_dict[key]['values'] = sample.get()
            if not sampler.no_mean:
                sum_vals = Reduce(sample, predicate_sum(sample.dtype), axes=(0,)).compile(thr)
                sum_dev = thr.empty_like(sum_vals.parameter.output)
                sum_vals(sum_dev, sample)
                sample_dict[key]['mean'] = sum_dev.get() / sample.shape[0]
            if not sampler.no_stderr:
                norm2 = EntrywiseNorm(sample, order=2, axes=(0,)).compile(thr)
                n2_dev = thr.empty_like(norm2.parameter.output)
                norm2(n2_dev, sample)
                std = n2_dev.get() / numpy.sqrt(sample.shape[0])
                sample_dict[key]['stderr'] = std / numpy.sqrt(sample.shape[0])

    return sample_dict, stop_integration


def calculate_errors(sample_normal, sample_double, strong_keys, weak_keys):

    # FIXME: performance can be improved by calculating norms on GPU

    weak_errors = {}
    strong_errors = {}

    for key in weak_keys:
        mean_normal = sample_normal[key]['mean']
        mean_double = sample_double[key]['mean']
        error_norm = numpy.linalg.norm(mean_normal)
        if error_norm > 0:
            error = numpy.linalg.norm(mean_normal - mean_double) / error_norm
        else:
            error = 0
        weak_errors[key] = error

    for key in strong_keys:
        values_normal = sample_normal[key]['values']
        values_double = sample_double[key]['values']

        errors = []
        for i in _range(values_normal.shape[0]):
            value_normal = values_normal[i]
            value_double = values_double[i]

            error_norm = numpy.linalg.norm(value_normal)
            if error_norm > 0:
                error = numpy.linalg.norm(value_normal - value_double) / error_norm
            else:
                error = 0
            errors.append(error)

        strong_errors[key] = max(errors)

    return strong_errors, weak_errors


def transpose_results(results):
    new_results = {}
    for key in results[0]:
        new_results[key] = dict(trajectories=results[0][key]['trajectories'])
        for val_key in results[0][key]:
            if val_key != 'trajectories':
                new_results[key][val_key] = []

    for res in results:
        for key in res:
            for val_key in res[key]:
                if val_key != 'trajectories':
                    new_results[key][val_key].append(res[key][val_key])

    for key in new_results:
        for val_key in new_results[key]:
            if val_key != 'trajectories':
                new_results[key][val_key] = numpy.array(new_results[key][val_key])

    return new_results


def join_results(results_list):
    full_results = {}
    for key in results_list[0]:

        r0 = results_list[0][key]

        trajectories = numpy.array([results[key]['trajectories'] for results in results_list])

        full_results[key] = dict(
            time=r0['time'],
            trajectories=trajectories.sum())

        if 'values' in r0:
            full_results[key]['values'] = numpy.concatenate(
                [results[key]['values'] for results in results_list], axis=0)

        if 'mean' in r0:
            means = numpy.concatenate(
                [results[key]['mean'].reshape(1, *r0['mean'].shape)
                for results in results_list], axis=0)

            full_results[key]['mean'] = (
                (means * trajectories.reshape(
                    trajectories.size, *([1] * len(r0['mean'].shape)))).sum(0)
                / trajectories.sum())

        if 'stderr' in r0:

            stderrs = numpy.concatenate(
                [results[key]['stderr'].reshape(1, *r0['stderr'].shape)
                for results in results_list], axis=0)

            full_results[key]['stderr'] = (
                numpy.sqrt((stderrs**2 * trajectories.reshape(
                    trajectories.size, *([1] * len(r0['mean'].shape)))**2).sum(0))
                / trajectories.sum())

    return full_results
