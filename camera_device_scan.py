"""
Cross-platform physical camera device scanning and opening.

This is OS-aware already (tries the right OpenCV backend for Windows vs
macOS/Linux automatically), so nothing here needs manual edits per
platform -- see get_platform_backends() below.

Used by camera_setup_ui.py (to list devices for the user to pick from in
the Settings window) and by live_detect.py (to open whichever device was
chosen there). Scanning/selecting a physical camera is a Settings-window
concern -- live_detect.py never prompts in the terminal for this.
"""

import contextlib
import os
import sys

import cv2

@contextlib.contextmanager
def _suppress_native_stderr():
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def get_backend_name(backend):
    if backend is None:
        return "default"
    backend_names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_DSHOW: "CAP_DSHOW",
        cv2.CAP_MSMF: "CAP_MSMF",
        cv2.CAP_VFW: "CAP_VFW",
    }
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backend_names[cv2.CAP_AVFOUNDATION] = "CAP_AVFOUNDATION"
    return backend_names.get(backend, str(backend))


def get_platform_backends():
    """Return backends appropriate for the current OS (Windows vs macOS/Linux)."""
    backends = [None]
    if os.name == "nt":
        backends += [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif hasattr(cv2, "CAP_AVFOUNDATION"):
        backends += [cv2.CAP_AVFOUNDATION]
    else:
        backends += [cv2.CAP_ANY]
    return backends


def try_open_camera(index, backend=None):
    """Attempt to open a camera silently. Returns (capture, frame) or None."""
    with _suppress_native_stderr():
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
    return cap, frame


def find_available_cameras(max_devices=5):
    """Scan for cameras, trying each backend supported by the platform.

    Returns a list of (index, backend, width, height) tuples.
    """
    discovered = []
    backends = get_platform_backends()

    for index in range(max_devices):
        for backend in backends:
            result = try_open_camera(index, backend)
            if result is None:
                continue
            cap, _frame = result
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            discovered.append((index, backend, width, height))
            break

    return discovered


def open_selected_camera(index, preferred_backend=None):
    """Open the chosen device, falling back to other platform backends."""
    backends = []
    if preferred_backend is not None:
        backends.append(preferred_backend)
    for backend in get_platform_backends():
        if backend not in backends:
            backends.append(backend)

    for backend in backends:
        result = try_open_camera(index, backend)
        if result is not None:
            cap, _frame = result
            return cap, backend

    return None, None


def format_device_label(index, backend, width, height):
    return f"Camera {index} — {width}x{height} ({get_backend_name(backend)})"