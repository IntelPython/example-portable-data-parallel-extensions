import numpy as np
import dpctl.tensor as dpt
from ._kde_sycl_ext import _kde


def _validate_inputs(poi, sample, h, expected_type):
    """
    Returns (poi.shape[0], samples.shape[0], poi.shape[1], h)
    """
    if not isinstance(poi, expected_type):
        raise TypeError(
            f"Expected first argument of type {expected_type}, got {type(poi)}"
        )
    if not isinstance(sample, expected_type):
        raise TypeError(
            f"Expected second argument of type {expected_type}, got {type(sample)}"
        )
    if not (sample.ndim == 2 and poi.ndim == 2):
        raise ValueError("Both input arrays must be two-dimensional")
    h = float(h)
    if not (h > 0):
        raise ValueError("KDE smoothing scale must be positive")
    m, d1 = poi.shape
    n, d2 = sample.shape
    if not (d1 == d2):
        raise ValueError(f"Dimensionality of inputs must be the same, but got {d1} and {d2}")
    return m, n, d1, h


def kde_dpctl(poi: dpt.usm_ndarray, sample: dpt.usm_ndarray, h: float) -> dpt.usm_ndarray:
    """Given a sample from underlying continuous distribution and
    a smoothing parameter `h`, evaluate density estimate at points of
    interest `poi`.
    """
    m, n, d, h = _validate_inputs(poi, sample, h, dpt.usm_ndarray)
    xp = poi.__array_namespace__()
    dm = xp.sum(xp.square(poi[:, xp.newaxis, ...] - sample[xp.newaxis, ...]), axis=-1)
    assert dm.shape == (m, n)
    two_pi = dpt.asarray(2*xp.pi, dtype=poi.dtype, device=poi.device)
    return xp.mean(xp.exp(dm/(-2*h*h)), axis=-1) * xp.pow(xp.sqrt(two_pi) * h, -d)


def kde_ext(poi: dpt.usm_ndarray, sample: dpt.usm_ndarray, h: float, mode=0) -> dpt.usm_ndarray:
    """Given a sample from underlying continuous distribution and
    a smoothing parameter `h`, evaluate density estimate at points of
    interest `poi`.
    """
    _, _, _, h = _validate_inputs(poi, sample, h, dpt.usm_ndarray)

    xp = poi.__array_namespace__()
    pdf = xp.empty_like(poi[:, 0])
    # Returns host-task event, and event associated with offloaded tasks
    ht_ev, impl_ev = _kde(poi=poi, sample=sample, pdf=pdf, h=h, mode=mode, depends=[])

    # wait for the events
    ht_ev.wait()
    impl_ev.wait()

    return pdf


def kde_numpy(poi: np.ndarray, sample: np.ndarray, h: float) -> np.ndarray:
    """Given a sample from underlying continuous distribution and
    a smoothing parameter `h`, evaluate density estimate at each point of
    interest `poi`.
    """
    _, _, d, h = _validate_inputs(poi, sample, h, np.ndarray)
    dm = np.sum(np.square(poi[:, np.newaxis, ...] - sample[np.newaxis, ...]), axis=-1)
    return np.mean(np.exp(dm/(-2*h*h)), axis=-1)/np.power(np.sqrt(2*np.pi) * h, d)
