from __future__ import print_function, division

import numpy
import sys
import time

from progressbar import Bar, ETA, Timer, Percentage, ProgressBar

from beclab.integrator.progress import StatefulLabel, HFill
from beclab.integrator.results import sample, transpose_results, calculate_errors, \
    Sampler, StopIntegration, IntegrationError, Timings, IntegrationInfo


_range = xrange if sys.version_info[0] < 3 else range


class Integrator:
    """
    A low-level integration class.
    Integrates an equation of the form

    .. math::

        dy(x, t) = S(x, y, t, dt, dW),

    where :math:`y(x)` is a multi-dimensional state vector with the primary dimension
    being the number of trajectories, and :math:`S` is a stepper function.

    :param thread: a Reikna ``Thread`` object.
    :param stepper: a :py:class:`Stepper` object providing the function :math:`S`.
    :param wiener: a :py:class:`Wiener` object that will be used to generate
        Wiener process differentials ``dW``.
    :param verbose: if ``True``, some information about integration progress
        will be printed to stdout.
    :param profile: if ``True``, CPU-GPU synchronization will be performed
        before each sample collection, allowing for accurate measurements
        of sampler time consumption.
    """

    def __init__(self, thread, stepper, wiener=None, verbose=True, profile=False):

        self.thr = thread
        self.verbose = verbose
        self.profile = profile

        self.stepper = stepper.compile(thread)
        # TODO: temporary dW array can be avoided if wiener-stepper is called in a computation
        if wiener is not None:
            self.noise = True
            self._w = wiener
            self.wiener = wiener.compile(thread)
            self.wiener_double = wiener.double_step().compile(thread)
            self.dW = self.thr.empty_like(wiener.parameter.dW)
            self.dW_state = self.thr.empty_like(wiener.parameter.state)
        else:
            self.noise = False

    def _sample(self, data, t, samplers):

        if self.profile:
            self.thr.synchronize()

        t1 = time.time()
        sample_dict, stop_integration = sample(data, t, samplers)
        t_samplers = time.time() - t1

        return sample_dict, stop_integration, t_samplers

    def _integrate(self, data_out, data_in, double_step, t_start, dt, steps, samples=None,
            samplers=None, verbose=False, filters=None):

        stepper = self.stepper

        # In case of double step we will be calling the Wiener noise generator twice,
        # keeping the time step the same as during the normal step,
        # thus allowing for strong error estimation.
        noise_dt = dt / 2 if double_step else dt

        results = []
        step = 0
        sample = 0
        t = t_start

        t_samplers = 0
        t_integration_start = time.time()

        if self.noise:
            wiener = self.wiener_double if double_step else self.wiener
            dW_state = self.dW_state
            dW = self.dW
            self.thr.to_device(
                numpy.zeros(wiener.parameter.state.shape, wiener.parameter.state.dtype),
                dest=dW_state)

        if verbose:
            title = 'Double step ' if double_step else 'Normal step '
            val_step = steps // 100 if steps >= 100 else 1
            pbar = ProgressBar(
                widgets=[title, Percentage(), Bar(), ETA()], maxval=steps - 1).start()

        for step in _range(steps):
            if step == 0:
                _data_in = data_in
            else:
                _data_in = data_out

            if self.noise:
                wiener(dW_state, dW, noise_dt)
                stepper(data_out, _data_in, dW, t, dt)
            else:
                stepper(data_out, _data_in, t, dt)

            t += dt

            # Calling every second step for normal step so that filters are executed
            # exactly at the same places, and the convergence can be measured.
            if double_step or (step + 1) % 2 == 0:
                for filter_ in filters:
                    filter_(data_out, t)

            if verbose and step % val_step == 0:
                pbar.update(step)

            if samples is not None:
                if (step + 1) % (steps // samples) == 0:
                    sample_dict, stop_integration, t_sampler = self._sample(data_out, t, samplers)
                    results.append(sample_dict)
                    t_samplers += t_sampler
                    if stop_integration:
                        break

        if verbose:
            pbar.finish()

        t_total = time.time() - t_integration_start

        if verbose and not double_step:
            print("Samplers time: {f:.1f}%".format(f=t_samplers / t_total * 100))

        return results, t_total - t_samplers, t_samplers

    def _check_convergence_keys(self, samplers, strong_convergence, weak_convergence):
        """
        Weak convergence can be estimated only for the samplers that collect mean values,
        and strong convergence can be estimated only for the samplers
        that collect per-trajectory values.
        This method checks that the convergence estimates requested by the user
        satisfy these conditions.
        """

        weak_keys = set(weak_convergence if weak_convergence is not None else [])
        strong_keys = set(strong_convergence if strong_convergence is not None else [])

        samplers_with_mean = set([key for key, sampler in samplers.items() if not sampler.no_mean])
        samplers_with_values = set([key for key, sampler in samplers.items() if not sampler.no_values])

        if not strong_keys.issubset(samplers_with_values):
            raise ValueError(
                "Samplers without all values cannot be used to estimate strong convergence:" +
                ", ".join(strong_keys - samplers_with_values))

        if not weak_keys.issubset(samplers_with_mean):
            raise ValueError(
                "Samplers without mean values cannot be used to estimate weak convergence:" +
                ", ".join(weak_keys - samplers_with_mean))

        return strong_keys, weak_keys

    def fixed_step(self, data_dev, t_start, t_end, steps,
            samples=1, samplers=None, filters=None,
            weak_convergence=None, strong_convergence=None):
        """
        Runs a fixed time step integration process.

        :param data_dev: a Reikna ``Array`` with the initial state vector
            (**Warning:** will be updated inplace).
            Should be the same as the one passed to the ``stepper`` object used to
            create this integrator.
        :param t_start: starting time.
        :param t_end: ending time.
        :param steps: number of steps to perform.
        :param samples: number of samples to collect (not including the sample at ``t = t_start``).
            ``steps`` must be divisible by ``samples``.
        :param samplers: a dictionary containing correspondences
            (``sampler_name``, ``sampler``), where ``sampler`` is a :py:class:`Sampler` object.
        :param filters: a list with :py:class:`Filter` objects that are applied to the data
            array after each two normal steps, or after each double step.
        :param weak_convergence: a list of sampler names which will be used to estimate
            weak convergence (a normalized 2-norm of the difference between the sampled mean values
            during integration with normal and double time steps).
        :param strong_convergence: a list of sampler names which will be used to estimate
            strong convergence (the maximum over trajectories of normalized 2-norms of differences
            between the sampled values during integration with normal and double time steps).
        :returns: a tuple ``(results, info)``, where ``info`` is a :py:class:`IntegrationInfo`
            object, and ``results`` is the data structure

            ::

                {sampler_name:
                    {trajectories: int,
                     time: array(samples + 1),
                     mean: array(samples + 1, *sample_shape),
                     values: array(samples + 1, trajectories, *sample_shape),
                     stderr: array(samples + 1, *sample_shape)}, ...}

            The existence of ``mean``, ``values`` and ``stderr`` keys depends on
            whether they were enabled by a particular sampler.
        """

        if samplers is None:
            samplers = {}
        if filters is None:
            filters = []

        strong_keys, weak_keys = self._check_convergence_keys(
            samplers, strong_convergence, weak_convergence)
        convergence_samplers = {key:samplers[key] for key in weak_keys | strong_keys}

        assert steps % samples == 0
        assert steps % 2 == 0
        dt = (t_end - t_start) / steps

        if self.verbose:
            print("Integrating from " + str(t_start) + " to " + str(t_end))

        # double step (to estimate the convergence)
        if len(convergence_samplers) > 0:
            data_double_dev = self.thr.copy_array(data_dev)
            results_double, t_double, t_samplers_double = self._integrate(
                data_double_dev, data_double_dev, True, t_start, dt * 2, steps // 2,
                samplers=convergence_samplers, samples=1, verbose=self.verbose, filters=filters)
        else:
            results_double = {}
            t_double = 0
            t_samplers_double = 0

        # actual integration
        sample_start, _, t_samplers_start = self._sample(data_dev, t_start, samplers)
        results, t_normal, t_samplers_normal = self._integrate(
            data_dev, data_dev, False, t_start, dt, steps,
            samples=samples, samplers=samplers, verbose=self.verbose, filters=filters)
        results = [sample_start] + results

        strong_errors, weak_errors = calculate_errors(
            results[-1], results_double[-1], strong_keys, weak_keys)
        if self.verbose:
            if len(strong_errors) > 0:
                print("Strong errors:", repr(strong_errors))
            if len(weak_errors) > 0:
                print("Weak errors:", repr(weak_errors))

        timings = Timings(
            normal=t_normal, double=t_double,
            samplers=t_samplers_normal + t_samplers_double + t_samplers_start)

        t_sample = (t_end - t_start) / samples
        ts_start = [t_start + i * t_sample for i in range(samples)]
        ts_end = [t_start + (i + 1 * t_sample) for i in range(samples)]
        steps_used = [
            (t_start, t_end, steps // samples)
            for t_start, t_end in zip(ts_start, ts_end)]

        info = IntegrationInfo(timings, strong_errors, weak_errors, steps_used)

        return transpose_results(results), info

    def adaptive_step(
            self, data_dev, t_start, t_sample,
            starting_steps=2,
            t_end=None,
            samplers=None,
            strong_convergence=None,
            weak_convergence=None,
            filters=None,
            display=None,
            steps_limit=16384):
        """
        Runs an adaptive step integration that varies the time step to satisfy
        the provided convergence criteria.

        :param data_dev: same as for :py:meth:`fixed_step`.
        :param t_start: same as for :py:meth:`fixed_step`.
        :param t_sample: time between samples.
        :param starting_steps: initial number of steps per sampling interval.
            Must be a power of 2 and greater than 1.
        :param t_end: ending time.
            If ``None``, the integration will continue until one of the samplers
            raises :py:class:`StopIntegration`.
        :param samplers: same as for :py:meth:`fixed_step`.
        :param filters: same as for :py:meth:`fixed_step`.
        :param weak_convergence: a dictionary of correspondences between sampler names
            and convergence thresholds.
            The convergence will be estimated after each sampling interval
            and the number of steps will be increased if it is above the threshold.
            Convergence is calculated the same as in :py:meth:`fixed_step`.
        :param strong_convergence: same as above, but for strong convergence.
        :param display: a list of sampler names or tuples ``(sampler_name, format_string)``
            specifying the values to be displayed during integration if ``verbose=True``.
            If none is specified, only the time will be shown.
            ``format_string`` is a format specification from ``format`` method of the standard
            string class.
            For example, ``('energy', '.3f')`` will result in printing the mean value from the
            ``'energy'`` sampler as ``'{energy:.3f'}.format(energy_mean)``.
        :param steps_limit: maximum number of steps per sampling interval.
            If the integrator cannot achieve the required convergence without exceeding
            this number, an :py:class:`IntegrationError` will be raised.
        :returns: same as for :py:meth:`fixed_step`.
        """

        if self.noise:
            # TODO: in order to support it we must somehow implement noise splitting
            # (so that the convergence improves steadily when we decrease the time step).
            # See Wilkie & Cetinbas, 2004 (doi:10.1016/j.physleta.2005.01.064).
            raise NotImplementedError("Adaptive step for stochastic equations is not supported")

        assert starting_steps % 2 == 0

        if samplers is None:
            samplers = {}
        if filters is None:
            filters = []

        strong_keys, weak_keys = self._check_convergence_keys(
            samplers, strong_convergence, weak_convergence)

        if len(strong_keys | weak_keys) == 0:
            raise ValueError("At least one convergence criterion must be specified")

        convergence_samplers = {key:samplers[key] for key in weak_keys | strong_keys}

        if self.verbose:
            label = StatefulLabel(display=display)

        if self.verbose:
            print(
                "Integrating from " + str(t_start) + " to " +
                (str(t_end) if t_end is not None else "infinity"))

        data_try = self.thr.empty_like(data_dev)
        data_double_try = self.thr.empty_like(data_dev)

        t = t_start
        steps = starting_steps

        sample_start, _, t_samplers = self._sample(data_dev, t, samplers)
        results = [sample_start]
        timings = Timings(samplers=t_samplers)
        steps_used = []
        if self.verbose:
            label.set(t, sample_start)

        if self.verbose:
            if t_end is None:
                widgets = [label, ' ', HFill(), Timer()]
                maxval = 1e10 # must be higher than any possible value of time
            else:
                widgets = [label, ' ', HFill(), Percentage(), ' ', ETA()]
                maxval = t_end - t_start

            pbar = ProgressBar(widgets=widgets, maxval=maxval).start()

        while True:
            dt = t_sample / steps
            _, t_double, _ = self._integrate(
                data_double_try, data_dev, True, t, dt * 2, steps // 2, verbose=False,
                filters=filters)
            _, t_normal, _ = self._integrate(
                data_try, data_dev, False, t, dt, steps, verbose=False,
                filters=filters)

            t += t_sample

            sample_normal, _, t_samplers_normal = self._sample(
                data_try, t, convergence_samplers)
            sample_double, _, t_samplers_double = self._sample(
                data_double_try, t, convergence_samplers)

            timings += Timings(
                normal=t_normal, double=t_double,
                samplers=t_samplers_normal + t_samplers_double)

            strong_errors, weak_errors = calculate_errors(
                sample_normal, sample_double, strong_keys, weak_keys)

            converged = (
                all(strong_errors[key] <= strong_convergence[key] for key in strong_keys) and
                all(weak_errors[key] <= weak_convergence[key] for key in weak_keys))

            if converged:
                self.thr.copy_array(data_try, dest=data_dev)

                sample_final, stop_integration, t_samplers_normal = self._sample(
                    data_try, t, samplers)

                results.append(sample_final)

                if self.verbose:
                    label.set(t, sample_final)
                    if t_end is None:
                        pbar.update(t - t_start)
                    else:
                        pbar.update(min(maxval, t - t_start))

                steps_used.append((t - t_sample, t, steps))

                if stop_integration or (t_end is not None and t >= t_end):
                    break

                if steps > 2:
                    steps //= 2
            else:
                t -= t_sample
                steps *= 2
                if steps > steps_limit:
                    raise IntegrationError("Number of steps per sample is greater than the limit")

        if self.verbose:
            pbar.finish()

        info = IntegrationInfo(timings, {}, {}, steps_used)

        return transpose_results(results), info
