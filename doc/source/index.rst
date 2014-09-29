*****************
reikna-integrator
*****************

A collection of SDE integration tools based on `Reikna <http://reikna.publicfields.net>`_.

Contents:

.. toctree::
   :maxdepth: 2


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


*************
API reference
*************

.. py:module:: reiknacontrib.integrator


Integrator
----------

.. autoclass:: Integrator
    :members:

.. autoclass:: Sampler
    :members: __call__

.. autoclass:: Filter
    :members: __call__


Steppers
--------

Available steppers:

.. py:class:: CDStepper

    Central difference semi-implicit stepper (`Werner and Drummond, 1997 <http://dx.doi.org/doi:10.1006/jcph.1996.5638>`_).

.. py:class:: CDIPStepper

    Central difference semi-implicit stepper in the interaction picture.
    Does not support momentum cutoffs at the moment.

.. py:class:: CDParallelStepper

    Central difference semi-implicit stepper for cases with no transverse derivatives.
    Supports real-valued state vectors.

.. py:class:: RK4IPStepper

    Standard Runge-Kutta 4th order in the interaction picture (`Caradoc-Davies, 2000 (PhD thesis) <http://www.physics.otago.ac.nz/research/jackdodd/resources/ResourceFiles/Caradoc-Davies_PhD_thesis.pdf>`_).

.. py:class:: RK46NLStepper

    Low-dissipation and low-dispersion 4th order Runge-Kutta (`Berland, Bogey and Bailly, 2006 <http://dx.doi.org/doi:10.1016/j.compfluid.2005.04.003>`_).

.. autoclass:: Stepper
    :members:

.. autoclass:: Drift
    :members:

.. autoclass:: Diffusion
    :members:


Wiener process
--------------

.. autoclass:: Wiener


Helper classes
--------------

.. autoclass:: IntegrationInfo

.. autoclass:: Timings


Exceptions
----------

.. autoclass:: StopIntegration

.. autoclass:: IntegrationError
