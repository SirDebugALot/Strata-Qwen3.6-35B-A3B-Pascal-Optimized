"""Selective v0.1.40 #691 port: process-local EcoQoS and unchanged job containment.

The upstream contained-child test is in:
https://github.com/Niko1221/Strata/blob/e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a/serve/test_winjob.py
These tests add Windows-10 fallback and failure-path checks, plus query the
actual execution-speed state on a short-lived child. No model/GPU is started.
Run: python -m unittest serve.test_winjob serve.test_winjob_power
"""
import ctypes
import os
import subprocess
import sys
import unittest
from unittest import mock

from serve import winjob


@unittest.skipUnless(os.name == "nt", "process power throttling is Windows-only")
class TestProcessPower(unittest.TestCase):
    def test_older_windows_retries_execution_speed_only(self):
        calls = []

        def set_info(handle, info_class, ptr, size):
            state = ctypes.cast(ptr, ctypes.POINTER(winjob._PowerThrottling)).contents
            calls.append((handle, info_class, size, state.Version, state.ControlMask, state.StateMask))
            return len(calls) == 2

        with mock.patch.object(winjob._k32, "SetProcessInformation", side_effect=set_info):
            self.assertTrue(winjob.no_throttle(123))
        self.assertEqual(calls, [(123, 4, 12, 1, 5, 0), (123, 4, 12, 1, 1, 0)])

    def test_rejected_power_request_does_not_break_containment(self):
        child = mock.Mock(_handle=123)
        with mock.patch.object(winjob, "_job", 456), \
                mock.patch.object(winjob._k32, "SetProcessInformation", return_value=False), \
                mock.patch.object(winjob._k32, "AssignProcessToJobObject", return_value=True) as assign:
            self.assertTrue(winjob.contain(child))
            assign.assert_called_once_with(456, 123)

    def test_contained_child_has_execution_throttling_disabled(self):
        from ctypes import wintypes

        # A venv launcher can add an intermediate process; use the actual interpreter.
        python = getattr(sys, "_base_executable", None) or sys.executable
        child = subprocess.Popen([python, "-c", "import time; time.sleep(30)"],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            self.assertTrue(winjob.no_throttle(int(child._handle)))
            self.assertTrue(winjob.contain(child))
            get_info = winjob._k32.GetProcessInformation
            get_info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
            get_info.restype = wintypes.BOOL
            state = winjob._PowerThrottling(1, 0, 0)
            self.assertTrue(get_info(int(child._handle), 4, ctypes.byref(state), ctypes.sizeof(state)))
            self.assertEqual(state.ControlMask & 1, 1)
            self.assertEqual(state.StateMask & 1, 0)
        finally:
            child.kill()
            child.wait(10)


if __name__ == "__main__":
    unittest.main()
