from typing import NamedTuple

import dpctl.tensor as dpt
import dpctl.utils as du
from ._qr import _qr


class QRDecompositionResult(NamedTuple):
    Q: dpt.usm_ndarray
    R: dpt.usm_ndarray


def qr(x : dpt.usm_ndarray) -> tuple[dpt.usm_ndarray, dpt.usm_ndarray]:
    """
    Compute QR decomposition for a stack of matrices using 
    oneMKL interface library calls.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray, got {type(x)}"
        )
    if x.ndim < 2:
        raise ValueError(
            "Input must be a matrix, or a stack of matrices"
        )
    m, n = x.shape[-2:]
    q_shape = x.shape[:-2] + (m, m,)
    r_shape = x.shape
    if x.size == 0:
        return QRDecompositionResult(
            dpt.empty(q_shape, dtype=x.dtype, usm_type=x.usm_type, device=x.device),
            dpt.empty_like(x)
        )
    
    if x.ndim == 2:
        x = x[dpt.newaxis, ...]
    
    x_f = dpt.moveaxis(x, (-2, -1), (0, 1))
    # must make a copy, since LAPACK overwrites content of this array
    x_f = dpt.asarray(x_f, copy=True, order="F")
    x_f = dpt.reshape(x_f, (m, n, -1))

    q_f = dpt.empty(
        (m, m, x_f.shape[-1]), 
        dtype=x.dtype, 
        device=x.device, 
        usm_type=x.usm_type,
        order="F"
    )
    r_f = dpt.empty_like(x_f, order="F")

    # either synchronize, or get dependencies and pass them 
    # to _qr via depends = list_of_events
    if hasattr(du, "SequentialOrderManager"):
        _mgr = du.SequentialOrderManager[x.sycl_queue]
        deps = _mgr.submitted_events
        ht_ev, qr_ev = _qr(stack_of_as=x_f, stack_of_qs=q_f, stack_of_rs=r_f, depends=deps)
        _mgr.add_event_pair(ht_ev, qr_ev)
    else:
        x.sycl_queue.wait()
        ht_ev, _ = _qr(stack_of_as=x_f, stack_of_qs=q_f, stack_of_rs=r_f)
        ht_ev.wait()

    q_f = dpt.moveaxis(q_f, -1, 0)
    r_f = dpt.moveaxis(r_f, -1, 0)
    q_f = dpt.reshape(q_f, q_shape)
    r_f = dpt.reshape(r_f, r_shape)
    return QRDecompositionResult(q_f, r_f)
