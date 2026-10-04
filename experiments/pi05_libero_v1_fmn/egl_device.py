"""Select the single accessible NVIDIA EGL device inside Slurm's device cgroup.

EGL enumeration is not assumed to match CUDA_VISIBLE_DEVICES ordinals. No
physical GPU ID is selected by the user; Slurm cgroup permissions are authoritative.
"""
import os
from slurm_runtime import require_slurm


def configure_egl():
    require_slurm()
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from mujoco.egl import egl_ext as EGL
    accessible = []
    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
        try:
            if display == EGL.EGL_NO_DISPLAY or not EGL.eglInitialize(display, None, None):
                continue
            vendor = EGL.eglQueryString(display, EGL.EGL_VENDOR)
            if isinstance(vendor, bytes): vendor = vendor.decode()
            if vendor and 'nvidia' in str(vendor).lower():
                accessible.append(index)
        except Exception:
            continue
        finally:
            if display != EGL.EGL_NO_DISPLAY:
                try: EGL.eglTerminate(display)
                except Exception: pass
    if len(accessible) != 1:
        raise RuntimeError(f'Expected one accessible NVIDIA EGL device in Slurm cgroup, found {accessible}')
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(accessible[0])
    print('SLURM_EGL_DEVICE', accessible[0], flush=True)
    return accessible[0]
