#!/usr/bin/env python
import unittest
import sys
import time
import traceback
import platform
import os


class PytestLikeResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        # Store (test, outcome, info) for each test:
        #   outcome ∈ { "passed", "failed", "error", "skipped" }
        #   info can hold the exception or skip reason, etc.
        self.results = []

    def startTestRun(self):
        """Called once before all tests."""
        self.start_time = time.time()
        # Print a header similar to pytest
        print("=" * 29 + " test session starts " + "=" * 29)
        # You can customize the platform/versions/paths as you like:
        print(
            f"platform {platform.system().lower()} -- Python {platform.python_version()}"
        )
        print(f"rootdir: {os.path.abspath(os.curdir)}")

    def stopTestRun(self):
        """Called once after all tests are done."""
        elapsed = time.time() - self.start_time
        print()  # blank line after final dot

        # Summarize
        ntests = len(self.results)
        print(f"collected {ntests} items")

        # Show details of failures/errors
        failures = [r for r in self.results if r[1] == "failed"]
        errors = [r for r in self.results if r[1] == "error"]

        if failures:
            print("\n" + "=" * 35 + " FAILURES " + "=" * 35)
            for test, outcome, err_info in failures:
                test = test.split("::")
                if len(test) == 3:
                    test = f"{test[1]}.{test[2]}"
                else:
                    test = f"{test[1]}"
                print(f"__________ {test} __________")
                self._print_traceback(err_info)
        if errors:
            print("\n" + "=" * 35 + " ERRORS " + "=" * 37)
            for test, outcome, err_info in errors:
                test = test.split("::")
                if len(test) == 3:
                    test = f"{test[1]}.{test[2]}"
                else:
                    test = f"{test[1]}"
                print(f"__________ {test} __________")
                self._print_traceback(err_info)

        # Optional short summary lines for passes/fails/skips
        print("\n==================== short test summary info ====================")
        for test, outcome, info in self.results:
            if outcome in ("passed", "failed", "error", "skipped"):
                print(f"{outcome.upper()} {test}")

        # Final one-line summary: “1 failed, 1 errors, 10 passed in 2.04s”
        passed_count = sum(1 for r in self.results if r[1] == "passed")
        failed_count = len(failures)
        error_count = len(errors)
        skipped_count = sum(1 for r in self.results if r[1] == "skipped")

        summary_bits = []
        if failed_count:
            summary_bits.append(f"{failed_count} failed")
        if error_count:
            summary_bits.append(f"{error_count} error{'s' if error_count != 1 else ''}")
        if passed_count:
            summary_bits.append(f"{passed_count} passed")
        if skipped_count:
            summary_bits.append(f"{skipped_count} skipped")

        summary_str = ", ".join(summary_bits) if summary_bits else "no tests run"
        print(
            f"=================== {summary_str} in {elapsed:.2f}s ==================="
        )

    def startTest(self, test):
        """Called right before each test method."""
        super().startTest(test)
        # Print a dot or some indicator
        print(".", end="", flush=True)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append((self._test_id(test), "passed", None))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append((self._test_id(test), "failed", err))

    def addError(self, test, err):
        super().addError(test, err)
        self.results.append((self._test_id(test), "error", err))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.results.append((self._test_id(test), "skipped", reason))

    def _print_traceback(self, err_info):
        """Helper to print the traceback like pytest does."""
        if isinstance(err_info, tuple) and len(err_info) == 3:
            exc_type, exc_value, tb = err_info
            traceback.print_exception(exc_type, exc_value, tb, file=sys.stdout)
        else:
            print(str(err_info))

    def _test_id(self, test):
        try:
            return f"{test.__class__.__module__}::{test.__class__.__name__}::{test._testMethodName}"
        except AttributeError:
            # Fallback for _ErrorHolder objects
            return f"{test.__class__.__module__}::{test.__class__.__name__}"


class PytestLikeRunner(unittest.TextTestRunner):
    """A custom TextTestRunner that uses the PytestLikeResult."""

    resultclass = PytestLikeResult

    def run(self, test):
        result = self._makeResult()
        # Overriding run to ensure we call startTestRun/stopTestRun
        result.startTestRun()
        test(result)
        result.stopTestRun()
        return result

    def tearDown(self):
        # Reset signal handlers
        from twisted.internet import reactor

        reactor.removeAll()
        reactor.stop()
        super().tearDown()


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(".", pattern="test_*.py")  # discover tests in r2e_tests/
    runner = PytestLikeRunner(verbosity=0)
    result = runner.run(suite)
    # Exit with code 0 if all tests passed/skipped, 1 if there were fails/errors
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
