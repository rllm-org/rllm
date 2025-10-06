"""
AppWorld Multithreading Signal Patch

AppWorld uses the signal module to implement timeout control, but signal can only be used in the main thread.
This patch replaces the signal-related functions to allow AppWorld to run in a multithreading environment.

Usage:
    Call apply_signal_patch() before importing AppWorld.
    from rllm.environments.appworld.signal_patch import apply_signal_patch
    apply_signal_patch()
    # Now it's safe to import and use AppWorld
    from appworld import AppWorld

Caveats:
    - This patch disables the timeout protection feature of signal
    - AppWorld's timeout check will no longer work
    - It is recommended to set timeout control at a higher level (like AgentExecutionEngine)
"""

import logging
import signal as _signal_module

logger = logging.getLogger(__name__)

# Global flag to prevent duplicate patch application
_patch_applied = False

# Save original function references
_original_functions = {}


def apply_signal_patch(verbose: bool = True):
    """
    Apply signal patch to allow AppWorld to run in a multithreading environment.

    Args:
        verbose: Whether to print patch application information

    Returns:
        bool: Whether the patch was successfully applied (returns False if already applied)
    """
    global _patch_applied, _original_functions

    if _patch_applied:
        if verbose:
            logger.info("Signal patch already applied, skipping")
        return False

    # Save original functions (for possible restoration)
    _original_functions["signal"] = _signal_module.signal
    _original_functions["getsignal"] = _signal_module.getsignal
    _original_functions["alarm"] = _signal_module.alarm
    if hasattr(_signal_module, "setitimer"):
        _original_functions["setitimer"] = _signal_module.setitimer

    # Define thread-safe alternative functions
    def _thread_safe_signal(signum, handler):
        """
        Thread-safe signal.signal() replacement.

        For SIGALRM, return None without setting a signal handler.
        For other signals, try using the original function, and if called in a non-main thread, catch the exception.
        """
        if signum == _signal_module.SIGALRM:
            return None
        try:
            return _original_functions["signal"](signum, handler)
        except ValueError as e:
            # ValueError: signal only works in main thread
            logger.debug(f"signal.signal() called in a non-main thread, ignored: {e}")
            return None

    def _thread_safe_getsignal(signum):
        """
        Thread-safe signal.getsignal() replacement.

        For SIGALRM, return None.
        For other signals, use the original function.
        """
        if signum == _signal_module.SIGALRM:
            return None
        return _original_functions["getsignal"](signum)

    def _thread_safe_alarm(seconds):
        """
        Thread-safe signal.alarm() replacement.

        Always return 0, indicating no previous alarm.
        Actually does not set any alarm.
        """
        return 0

    def _thread_safe_setitimer(which, seconds, interval=0):
        """
        Thread-safe signal.setitimer() replacement.

        Always return (0, 0), indicating no previous timer.
        Actually does not set any timer.
        """
        return (0, 0)

    # Apply the patch
    _signal_module.signal = _thread_safe_signal
    _signal_module.getsignal = _thread_safe_getsignal
    _signal_module.alarm = _thread_safe_alarm
    if hasattr(_signal_module, "setitimer"):
        _signal_module.setitimer = _thread_safe_setitimer

    _patch_applied = True

    if verbose:
        logger.info("Signal patch applied - AppWorld can now run in a multithreading environment")
        logger.warning("Warning: signal timeout protection is disabled, please set timeout control at a higher level")

    return True


def restore_signal_functions():
    """
    Restore the original signal functions.

    Warning: Only use this for testing or debugging. After restoration, AppWorld will no longer work in a multithreading environment.

    Returns:
        bool: Whether the signal functions were successfully restored (returns False if the patch was not applied)
    """
    global _patch_applied, _original_functions

    if not _patch_applied:
        logger.warning("Signal patch not applied, no need to restore")
        return False

    # Restore original functions
    _signal_module.signal = _original_functions["signal"]
    _signal_module.getsignal = _original_functions["getsignal"]
    _signal_module.alarm = _original_functions["alarm"]
    if "setitimer" in _original_functions:
        _signal_module.setitimer = _original_functions["setitimer"]

    _patch_applied = False
    _original_functions.clear()

    logger.info("Signal functions restored to original implementation")
    logger.warning("Warning: AppWorld will no longer work in a non-main thread")

    return True


def is_patch_applied():
    """
    Check if the signal patch has been applied.

    Returns:
        bool: Whether the patch has been applied
    """
    return _patch_applied


# Convenient context manager
class SignalPatch:
    """
    Context manager for signal patch.

    Usage example:
        with SignalPatch():
            from appworld import AppWorld
            world = AppWorld(task_id="test")
            # In this code block, the signal patch is activated
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.applied_by_me = False

    def __enter__(self):
        self.applied_by_me = apply_signal_patch(verbose=self.verbose)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.applied_by_me:
            restore_signal_functions()
        return False
