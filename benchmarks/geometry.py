"""Pure NumPy geometry shared by benchmark and policy adapters."""
import math
import numpy as np


def quat_wxyz_to_axis_angle(quat):
    value = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError('Expected a finite, nonzero quaternion')
    value /= norm
    if value[0] < 0:
        value = -value
    w = float(np.clip(value[0], -1, 1))
    scale = math.sqrt(max(1 - w*w, 0))
    if scale < 1e-8:
        return np.zeros(3, np.float32)
    return (value[1:] * (2 * math.acos(w) / scale)).astype(np.float32)


def axis_angle_to_quat_wxyz(rotation):
    value = np.asarray(rotation, dtype=np.float64).reshape(3)
    if not np.isfinite(value).all():
        raise ValueError('Expected finite axis-angle rotation')
    angle = float(np.linalg.norm(value))
    if angle < 1e-8:
        return np.array([1, 0, 0, 0], np.float32)
    return np.concatenate(([math.cos(angle/2)], value/angle * math.sin(angle/2))).astype(np.float32)


def xyzw_to_wxyz(quat):
    value = np.asarray(quat, dtype=np.float32).reshape(4)
    return value[[3, 0, 1, 2]].copy()
