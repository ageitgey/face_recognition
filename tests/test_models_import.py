#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Importing `face_recognition` when `face_recognition_models` cannot be imported
must raise an ImportError that keeps the original cause, instead of printing a
message and calling quit() (which exits the interpreter with status 0).
"""

import subprocess
import sys
import textwrap
import unittest


class Test_models_import_error(unittest.TestCase):

    def run_snippet(self, code):
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
        )

    def test_missing_models_raises_import_error(self):
        result = self.run_snippet("""
            import sys
            sys.modules["face_recognition_models"] = None  # makes the import fail
            try:
                import face_recognition
            except ImportError as e:
                assert "face_recognition_models" in str(e), str(e)
                assert isinstance(e.__cause__, ImportError), repr(e.__cause__)
                print("IMPORT_ERROR_RAISED")
        """)
        self.assertIn("IMPORT_ERROR_RAISED", result.stdout, result.stdout + result.stderr)

    def test_missing_models_exits_with_failure_status(self):
        result = self.run_snippet("""
            import sys
            sys.modules["face_recognition_models"] = None
            import face_recognition
        """)
        self.assertNotEqual(result.returncode, 0)
